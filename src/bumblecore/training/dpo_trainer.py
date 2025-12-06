from typing import Optional, Mapping, Union
from contextlib import contextmanager,nullcontext
import copy

from typing_extensions import override
from transformers import AutoTokenizer
from transformers.utils import is_flash_attn_2_available
from transformers import AutoModelForCausalLM
import torch
import torch.distributed as dist
import deepspeed

from .base_trainer import BaseTrainer
from ..data_processing import DataFormatter, DPODataset, load_dpo_data, DPOCollator
from ..config import TrainConfig
from ..model_loss import DPOLoss,ForCausalLMLoss
from ..utils import LossPlotter


class DPOTrainer(BaseTrainer):
    def __init__(self, config: TrainConfig):
        self.config = config
        self.format_preprocess_fn = DataFormatter(self.config.training_stage)
        self.tokenizer, self.train_dataset = self._prepare_datasets()
        self.data_collator = DPOCollator(self.tokenizer)
        super().__init__(config, self.train_dataset, self.tokenizer, self.data_collator)
        self.reference_model = self._setup_reference_model()
        self._print_train_parameters()

        self.use_ref_model = True
    

    def _setup_reference_model(self):
        if self.config.finetuning_type == "lora":
            reference_model = None
            self._print_log("LoRA detected: using policy model with disabled adapter as reference.")
        else:
            reference_model = self._initialize_deepspeed_reference_engine()
            self._print_log("Using separate frozen reference model.")

        return reference_model


    def _get_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            use_fast=self.config.tokenizer_use_fast,
            trust_remote_code=self.config.trust_remote_code,
        )
    

    def _prepare_datasets(self):
        dataset = load_dpo_data(self.config.dataset_path)
        messages = self.format_preprocess_fn(dataset)

        tokenizer = self._get_tokenizer()
        train_dataset = DPODataset(
            messages,
            tokenizer,
            max_length=self.config.cutoff_len,
        )
        return tokenizer, train_dataset
    

    def _get_reference_model(self):
        return AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=self.config.trust_remote_code,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "eager",
            dtype=self.dtype,
        )


    def _initialize_deepspeed_reference_engine(self):
        reference_deepspeed_config = copy.deepcopy(self.deepspeed_config)
        zero_stage = reference_deepspeed_config.get("zero_optimization", {}).get("stage", 0)
        if zero_stage == 3:
            with deepspeed.zero.Init(config_dict_or_path=reference_deepspeed_config):
                reference_model = self._get_reference_model()
        else:
            reference_deepspeed_config["zero_optimization"] = {"stage": 0}
            reference_model = self._get_reference_model()

        reference_model_engine, _, _, _ = deepspeed.initialize(
            model=reference_model,
            config_params=reference_deepspeed_config,
        )
        reference_model_engine.eval()
        return reference_model_engine


    def nested_detach(
        self,
        tensors: Union["torch.Tensor", list["torch.Tensor"], tuple["torch.Tensor"], dict[str, "torch.Tensor"]],
        clone: bool = False,
    ):

        if isinstance(tensors, (list, tuple)):
            return type(tensors)(self.nested_detach(t, clone=clone) for t in tensors)
        elif isinstance(tensors, Mapping):
            return type(tensors)({k: self.nested_detach(t, clone=clone) for k, t in tensors.items()})

        if isinstance(tensors, torch.Tensor):
            if clone:
                return tensors.detach().clone()
            else:
                return tensors.detach()
        else:
            return tensors
            

    def concatenated_forward(
        self, 
        model, 
        batch: dict[str, torch.Tensor], 
        is_ref_model: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if self.use_ref_model:
            batch = self.nested_detach(batch, clone=True)

        all_logits = model(**batch, use_cache=False).logits.to(torch.float32)

        all_logps, valid_length = self.get_batch_logps(
            logits=all_logits, 
            labels=batch["labels"], 
            ld_alpha=(self.config.ld_alpha if not is_ref_model else None)
        )

        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        chosen_length, _ = valid_length.split(batch_size, dim=0)

        return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_logps / chosen_length
    
    @contextmanager
    def temporary_eval(self, model):
        was_training = model.module.training
        model.module.eval()
        try:
            yield model
        finally:
            if was_training:
                model.module.train()

    @torch.no_grad()
    def get_reference_logps(
        self,
        reference_model, 
        batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        if reference_model is None:
            model_to_use = self.model_engine
            ctx_adapter = self.model_engine.module.disable_adapter()
        else:
            model_to_use = reference_model
            ctx_adapter = nullcontext()

        with ctx_adapter, self.temporary_eval(model_to_use):
            reference_chosen_logps, reference_rejected_logps, *_ = self.concatenated_forward(
                model_to_use, batch, is_ref_model=True
            )

        return reference_chosen_logps, reference_rejected_logps
    

    def dpo_with_sft_loss(
        self,
        logits,
        labels,
        dpo_loss,
        sft_weight=0.0,
    ):
        sft_loss = ForCausalLMLoss(logits, labels, vocab_size=self.vocab_size)
        return dpo_loss + sft_weight * sft_loss

    @override
    def compute_loss(self, model_engine, batch):

        cat_batch  = {
            "input_ids": torch.cat([batch["chosen_input_ids"], batch["rejected_input_ids"]], dim=0),
            "attention_mask": torch.cat([batch["chosen_attention_mask"], batch["rejected_attention_mask"]], dim=0),
            "labels": torch.cat([batch["chosen_labels"], batch["rejected_labels"]], dim=0)
        }

        policy_chosen_logps, policy_rejected_logps, policy_chosen_logits, policy_rejected_logits, _ = self.concatenated_forward(model_engine, cat_batch)

        reference_chosen_logps, reference_rejected_logps = self.get_reference_logps(self.reference_model, cat_batch)

        losses, chosen_rewards, rejected_rewards = DPOLoss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=reference_chosen_logps,
            reference_rejected_logps=reference_rejected_logps,
            beta=self.config.pref_beta,
            label_smoothing=self.config.dpo_label_smoothing
        )
        loss = losses.mean()

        if self.config.sft_weight > 0:
            loss = self.dpo_with_sft_loss(
                logits = policy_chosen_logits,
                labels = batch["chosen_labels"],
                dpo_loss=loss,
                sft_weight=self.config.sft_weight
            )
        
        rewards_chosen_mean = chosen_rewards.mean()
        rewards_rejected_mean = rejected_rewards.mean()
        rewards_accuracies = (chosen_rewards > rejected_rewards).float().mean()
        rewards_margins = (chosen_rewards - rejected_rewards).mean()

        logps_chosen_mean = policy_chosen_logps.mean()
        logps_rejected_mean = policy_rejected_logps.mean()
        logits_chosen_mean = policy_chosen_logits.mean()
        logits_rejected_mean = policy_rejected_logits.mean()

        return dict(
            loss=loss,
            metrics=dict(
                rewards_chosen_mean=rewards_chosen_mean,
                rewards_rejected_mean=rewards_rejected_mean,
                rewards_accuracies=rewards_accuracies,
                rewards_margins=rewards_margins,
                logps_chosen_mean=logps_chosen_mean,
                logps_rejected_mean=logps_rejected_mean,
                logits_chosen_mean=logits_chosen_mean,
                logits_rejected_mean=logits_rejected_mean,
            )
        )
    
    @override
    def gather_scalar_for_log(self, computation_result):
        loss = computation_result["loss"]
        loss_tensor = loss.detach().clone()
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        computation_result["avg_loss"] = (loss_tensor / self.world_size).item()

        metrics = computation_result["metrics"]
        for key in metrics.keys():
            dist.all_reduce(metrics[key], op=dist.ReduceOp.SUM)
            metrics[key] = (metrics[key] / self.world_size).item()

        return computation_result     

    def get_batch_logps(
        self,
        logits: "torch.Tensor",
        labels: "torch.Tensor",
        label_pad_token_id: int = -100,
        ld_alpha: Optional[float] = None,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batchsize x seqlen) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id
        labels[labels == label_pad_token_id] = 0  
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        valid_length = loss_mask.sum(-1)
        if ld_alpha is not None:
            num_examples = labels.shape[0] // 2
            chosen_lengths = valid_length[:num_examples]
            rejected_lengths = valid_length[num_examples:]
            min_lengths = torch.min(chosen_lengths, rejected_lengths)
            start_positions = torch.argmax(loss_mask.int(), dim=1)
            public_lengths = start_positions + torch.cat([min_lengths, min_lengths], dim=0)

            seq_len = labels.shape[-1]
            position_ids = torch.arange(seq_len, device=per_token_logps.device).expand_as(per_token_logps)

            ld_mask = position_ids < public_lengths.unsqueeze(1)
            front_mask = (ld_mask * loss_mask).float()
            rear_mask = (~ld_mask * loss_mask).float()

            front_logps = (per_token_logps * front_mask).sum(-1)
            rear_logps = (per_token_logps * rear_mask).sum(-1)
            logps = front_logps + ld_alpha * rear_logps
        else:
            logps = (per_token_logps * loss_mask).sum(-1)

        return logps, valid_length
    
    @override
    def plot_training_loss(self):
        if self.rank != 0:
            return
        
        steps = [item["step"] for item in self.training_metrics]
        losses = [item["loss"] for item in self.training_metrics]
        accuracies = [item["rewards/accuracies"] for item in self.training_metrics]
        smooth_plot = (self.config.training_stage not in ["pretrain", "continue_pretrain"])
        LossPlotter.plot_loss(
            x_values=steps, 
            y_values=losses,
            xlabel='Step', 
            ylabel='Loss', 
            save_dir=self.config.output_dir,
            filename="training_loss",
            smooth_plot=smooth_plot
        )
        LossPlotter.plot_loss(
            x_values=steps, 
            y_values=accuracies,
            xlabel='Step', 
            ylabel='rewards/accuracies', 
            save_dir=self.config.output_dir,
            filename="training_rewards_accuracies",
            smooth_plot=smooth_plot
        )

        self._print_log(f"Training loss and training_rewards_accuracies saved to {self.config.output_dir}")
    

    @override
    def log_metrics(self, lr, grad_norm, epochs_done, global_step, computation_result):
        m = computation_result["metrics"]
        avg_loss = computation_result["avg_loss"]
        msg = {
            "step": global_step,
            "loss": avg_loss,
            "learning_rate": lr,
            "grad_norm": grad_norm,
            "epoch": epochs_done,
            "rewards/chosen": m["rewards_chosen_mean"],
            "rewards/rejected": m["rewards_rejected_mean"],
            "rewards/accuracies": m["rewards_accuracies"],
            "rewards/margins": m["rewards_margins"],
            "logps/chosen": m["logps_chosen_mean"],
            "logps/rejected": m["logps_rejected_mean"],
            "logits/chosen": m["logits_chosen_mean"],
            "logits/rejected": m["logits_rejected_mean"],
        }

        self.training_metrics.append(msg)

        log_str = (
            f"step: {global_step}/{self.max_steps} | "
            f"loss: {msg['loss']:.4f} | "
            f"lr: {lr:.2e} | "
            f"grad_norm: {grad_norm:.4f} | "
            f"acc: {m['rewards_accuracies']:.4f} | "
            f"margin: {m['rewards_margins']:.4f} | "
            f"epoch: {epochs_done:.2f} | "
        )
        prefix_len = len(f"step: {global_step}/{self.max_steps} | ")
        rewards = " " * prefix_len + f"{'→ Rewards':<10} | Chosen: {m['rewards_chosen_mean']:>10.4f} | Rejected: {m['rewards_rejected_mean']:>10.4f} |"
        probs   = " " * prefix_len + f"{'→ LogPs':<10} | Chosen: {m['logps_chosen_mean']:>10.4f} | Rejected: {m['logps_rejected_mean']:>10.4f} |"
        logits  = " " * prefix_len + f"{'→ Logits':<10} | Chosen: {m['logits_chosen_mean']:>10.4f} | Rejected: {m['logits_rejected_mean']:>10.4f} |"
        lines = [log_str, rewards, probs, logits]
        max_len = max(len(line) for line in lines)
        separator = "-" * max_len

        full_log = "\n".join(lines + [separator])
        self._print_log(full_log)

        if self.tb_writer is not None:
            writer = self.tb_writer
            writer.add_scalar("Train/Loss", avg_loss, global_step)
            writer.add_scalar("Train/LearningRate", lr, global_step)
            writer.add_scalar("Train/GradNorm", grad_norm, global_step)

            writer.add_scalar("Metrics/Margin", m["rewards_margins"], global_step)
            writer.add_scalar("Metrics/Accuracy", m["rewards_accuracies"], global_step)

            writer.add_scalar("Rewards/Chosen", m["rewards_chosen_mean"], global_step)
            writer.add_scalar("Rewards/Rejected", m["rewards_rejected_mean"], global_step)

            writer.add_scalar("LogPs/Chosen", m["logps_chosen_mean"], global_step)
            writer.add_scalar("LogPs/Rejected", m["logps_rejected_mean"], global_step)

            writer.add_scalar("Logits/Chosen", m["logits_chosen_mean"], global_step)
            writer.add_scalar("Logits/Rejected", m["logits_rejected_mean"], global_step)