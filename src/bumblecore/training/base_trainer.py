import os
import json
import inspect
from collections import deque
from typing import Callable
import shutil
import math

from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    get_scheduler,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoConfig,
    GenerationConfig,
)
from transformers.utils import is_flash_attn_2_available
from transformers.integrations import HfDeepSpeedConfig
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from peft import LoraConfig, get_peft_model, TaskType


from ..bumblebee import BumblebeeConfig, BumblebeeForCausalLM
from ..config import TrainConfig
from ..model_loss import ForCausalLMLoss
from ..utils import setup_trainer_logger, LossPlotter

AutoModelForCausalLM.register(BumblebeeConfig, BumblebeeForCausalLM)
AutoConfig.register("bumblebee", BumblebeeConfig)

class BaseTrainer:
    def __init__(
        self, 
        config: TrainConfig,
        train_dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        data_collator: Callable,
    ):
        self.config = config
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.data_collator = data_collator

        self._init_distributed()
        self.logger = setup_trainer_logger(
            output_dir=self.config.output_dir if self.config.save_train_log else None
        )
        self.deepspeed_config, self.dtype, self.total_train_batch_size = self._setup_deepspeed()
        self.model, self.vocab_size = self._build_model()
        self.train_sampler = self._build_train_sampler()
        self.trainloader = self._build_train_dataloader()

        self.num_update_steps_per_epoch, self.max_steps, self.warmup_steps = self._calculate_training_steps()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.model_engine = self._initialize_deepspeed_engine()

        self.ckpt_queue = self._setup_checkpointing()
        self.tb_writer = self._setup_tensorboard() if self.rank == 0 else None
        self.training_metrics = [] if self.rank == 0 else None
    
    def _init_distributed(self):
        # deepspeed.utils.logging.set_log_level_from_string("info")
        deepspeed.init_distributed()
        self.local_rank = deepspeed.comm.get_local_rank()
        torch.cuda.set_device(self.local_rank)
        self.world_size = deepspeed.comm.get_world_size()
        self.rank = deepspeed.comm.get_rank()

    def _load_deepspeed_config(self, config_path: str) -> dict:
        with open(config_path, "r") as f:
            return json.load(f)

    def _select_precision(self, ds_config: dict, precision: str):
        for key in ("fp16", "bf16"):
            if key in ds_config:
                ds_config[key]["enabled"] = False

        if precision not in {"fp16", "bf16"}:
            raise ValueError(f"Unsupported precision: {precision}")

        ds_config.setdefault(precision, {})["enabled"] = True

        if precision == "bf16":
            dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        return ds_config, dtype

    def _compute_batch_sizes(
        self,
        ds_config: dict,
        micro_batch: int,
        grad_acc: int,
    ):
        total_batch = micro_batch * grad_acc * self.world_size
        ds_config["train_micro_batch_size_per_gpu"] = micro_batch
        ds_config["gradient_accumulation_steps"] = grad_acc
        ds_config["train_batch_size"] = total_batch
        return ds_config, total_batch

    def _tune_deepspeed_communication(self, ds_config: dict, model_name_or_path: str):
        model_config = AutoConfig.from_pretrained(model_name_or_path)
        hidden_size = model_config.hidden_size
        zp = ds_config.setdefault("zero_optimization", {})
        if zp.get("stage") == 3:
            zp["reduce_bucket_size"] = hidden_size * hidden_size
            zp["stage3_prefetch_bucket_size"] = int(0.9 * hidden_size * hidden_size)
            zp["stage3_param_persistence_threshold"] = 10 * hidden_size
        return ds_config

    def _setup_tensorboard(self):
        if not getattr(self.config, "use_tensorboard", False):
            return None
            
        tb_log_dir = os.path.join(self.config.output_dir, "tensorboard_train_log")
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        self._print_log(f"TensorBoard logging enabled. Logs will be saved to: {tb_log_dir}")
        return tb_writer

    def _setup_deepspeed(self):
        ds_config = self._load_deepspeed_config(self.config.deepspeed_config_path)
        ds_config, dtype = self._select_precision(ds_config, self.config.train_model_precision)
        ds_config, total_train_batch_size = self._compute_batch_sizes(
            ds_config,
            self.config.train_micro_batch_size_per_gpu,
            self.config.gradient_accumulation_steps,
        )
        ds_config = self._tune_deepspeed_communication(ds_config, self.config.model_name_or_path)
        return ds_config, dtype, total_train_batch_size

    def _create_base_model(self):
        zero_stage = self.deepspeed_config.get("zero_optimization", {}).get("stage", 0)

        def _instantiate_model():
            common_kwargs = dict(
                trust_remote_code=self.config.trust_remote_code,
                attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "eager",
                dtype=self.dtype,
            )
            if self.config.training_stage == "pretrain":
                config = AutoConfig.from_pretrained(self.config.model_name_or_path)
                is_registered_model = config.model_type == "bumblebee"
                pretrain_kwargs = dict(
                    trust_remote_code=False if is_registered_model else self.config.trust_remote_code,
                    attn_implementation=common_kwargs["attn_implementation"],
                    dtype=common_kwargs["dtype"],
                )
                model = AutoModelForCausalLM.from_config(config, **pretrain_kwargs)
                self._print_log(f"[{self.config.training_stage}] Initialized from config: {self.config.model_name_or_path}")
            elif self.config.training_stage in ["continue_pretrain", "sft", "dpo"]:
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name_or_path, low_cpu_mem_usage=True, **common_kwargs,
                )
                self._print_log(f"[{self.config.training_stage}] Loaded model from: {self.config.model_name_or_path}")
            else:
                raise ValueError(f"Invalid training stage: {self.config.training_stage}")
                
            return model

        if zero_stage == 3 and self.config.training_stage != "pretrain":
            self.dschf = HfDeepSpeedConfig(self.deepspeed_config)
            # with deepspeed.zero.Init(config_dict_or_path=self.deepspeed_config):
            return _instantiate_model()
        else:
            return _instantiate_model()


    def _configure_model_for_training(self, model):
        vocab_size = model.config.vocab_size
        if self.config.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable()
        return model, vocab_size

    def _apply_lora(self, model):
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model
    

    def _load_generation_config_from_pretrained(self, model):
        gen_config_path = os.path.join(self.config.model_name_or_path, "generation_config.json")
        if os.path.exists(gen_config_path):
            try:
                generation_config = GenerationConfig.from_pretrained(self.config.model_name_or_path)
                model.generation_config = generation_config
                self._print_log(f"Loaded generation config from {gen_config_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load generation config: {e}")
        else:
            self._print_log("No generation_config.json found in model path; using default or existing config.")
        

    def _build_model(self):
        model = self._create_base_model()
        model, vocab_size = self._configure_model_for_training(model)
        
        if self.config.finetuning_type == "lora": 
            model = self._apply_lora(model)

        self._load_generation_config_from_pretrained(model)
        
        return model, vocab_size


    def _build_train_sampler(self):
        return DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            seed=42,
            shuffle=True,
            drop_last=True,
        )
    

    def _get_num_workers(self):
        nw = self.config.num_local_io_workers
        if nw is not None:
            if not isinstance(nw, int):
                raise TypeError("num_local_io_workers must be an integer or None. ")
            
            if nw < 1:
                raise ValueError(
                    "Setting `num_local_io_workers < 1` is not allowed in distributed training, "
                    "as it may severely degrade performance. Please use a positive integer (e.g., 2, 4, 8) "
                    "or leave it unset to use the default value."
                )
            return nw

        return 2 * get_accelerator().device_count()


    def _build_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_micro_batch_size_per_gpu,
            collate_fn=self.data_collator,
            sampler=self.train_sampler,
            num_workers=self._get_num_workers(),
            persistent_workers=True,
            pin_memory=True,
        )
    

    def _calculate_training_steps(self):
        len_dataloader = len(self.trainloader)

        num_update_steps_per_epoch = (
            len_dataloader // self.config.gradient_accumulation_steps
            + int(len_dataloader % self.config.gradient_accumulation_steps > 0)
        )
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        max_steps = math.ceil(num_update_steps_per_epoch * self.config.num_epochs)
        warmup_steps = math.ceil(self.config.warmup_ratio * max_steps)

        return num_update_steps_per_epoch, max_steps, warmup_steps


    def _build_optimizer(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
    
    # def _get_decay_parameter_names(self, model) -> set[str]:
    #     decay_params = set()
    #     for name, param in model.named_parameters():
    #         if not param.requires_grad:
    #             continue
    #         # Skip bias and normalization layers
    #         if any(nd in name.lower() for nd in ["bias", "layernorm", "rmsnorm"]):
    #             continue
    #         decay_params.add(name)
    #     return decay_params
    
    # def _build_optimizer(self):
    #     decay_param_names = self._get_decay_parameter_names(self.model)

    #     decay_params = []
    #     no_decay_params = []

    #     for name, param in self.model.named_parameters():
    #         if not param.requires_grad:
    #             continue
    #         if name in decay_param_names:
    #             decay_params.append(param)
    #         else:
    #             no_decay_params.append(param)

    #     optimizer_grouped_parameters = [
    #         {"params": decay_params, "weight_decay": self.config.weight_decay},
    #         {"params": no_decay_params, "weight_decay": 0.0},
    #     ]

    #     return torch.optim.AdamW(
    #         optimizer_grouped_parameters,
    #         lr=self.config.learning_rate,
    #     )


    def _build_scheduler(self):
        return get_scheduler(
            name= self.config.lr_scheduler_type,
            optimizer = self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.max_steps,
        )


    def _initialize_deepspeed_engine(self):
        model_engine, _, _, _ = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.scheduler,
            config_params=self.deepspeed_config,
        )
        return model_engine


    def _setup_checkpointing(self):
        if self.rank == 0:
            os.makedirs(self.config.output_dir, exist_ok=True)
        ckpt_queue = deque()
        return ckpt_queue


    def _rotate_checkpoints(self):
        while len(self.ckpt_queue) > self.config.save_total_limit:
            old_ckpt = self.ckpt_queue.popleft()
            if os.path.exists(old_ckpt):
                shutil.rmtree(old_ckpt)
    
    def _save_checkpoint(self, step, is_last=False):

        tag = str(step)
        ckpt_dir = os.path.join(self.config.output_dir, f"checkpoint-{tag}")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.model_engine.save_checkpoint(
            ckpt_dir,
            exclude_frozen_parameters=(self.config.finetuning_type == "lora")
        )

        if self.rank == 0:
            self._save_tokenizer(ckpt_dir)
            self._save_train_model(ckpt_dir)

            if not is_last:
                self.ckpt_queue.append(ckpt_dir)
                self._rotate_checkpoints()

        self._print_log(f"Saving model at step {step} to {ckpt_dir}")


    def _save_tokenizer(self, ckpt_dir):
        if isinstance(self.tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            self.tokenizer.save_pretrained(ckpt_dir)


    def _save_model_code_files(self, ckpt_dir):
        
        MODEL_CONFIG = {
            "bumblebee": (BumblebeeForCausalLM, ["modeling_bumblebee.py"]),
        }
        
        model_type = self.model_engine.module.config.model_type
        config = MODEL_CONFIG.get(model_type)
        
        if not config:
            return
        
        model_class, files_to_copy = config
        model_dir = os.path.dirname(inspect.getfile(model_class))
        
        for filename in files_to_copy:
            src_file = os.path.join(model_dir, filename)
            shutil.copy2(src_file, os.path.join(ckpt_dir, filename))
            self._print_log(f"Copied {filename} to {ckpt_dir}")


    def _save_train_model(self, ckpt_dir):
        zero_cfg = self.model_engine.config.get("zero_optimization", {})
        zero_stage = zero_cfg.get("stage", 0)

        if zero_stage == 3:
            state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_dir)
            state_dict = {k: v.to(self.dtype) for k, v in state_dict.items()}
        else:
            state_dict = self.model_engine.module_state_dict()

        self.model_engine.module.save_pretrained(
            ckpt_dir,
            state_dict=state_dict,
            safe_serialization=True,
            max_shard_size="5GB",
        )
        
        self._save_model_code_files(ckpt_dir)


    def train(self):
        self.tr_accumulated = {}

        progress_bar = tqdm(total=self.max_steps, desc="Training") if self.rank == 0 else None
        
        global_step = 0
        self.model_engine.train()
        for epoch in range(math.ceil(self.config.num_epochs)):
            global_step = self._train_epoch(epoch, global_step, progress_bar)
        
        if self.config.save_last:

            self._print_log(f"Training completed. Saving final checkpoint at step {global_step}...")
            self._save_checkpoint(global_step, is_last=True)

        if self.rank == 0:
            progress_bar.close()

        self._cleanup()


    def _train_epoch(self, epoch, global_step, progress_bar):
        self.train_sampler.set_epoch(epoch)

        for batch in self.trainloader:
            if global_step >= self.max_steps:
                break
            global_step = self._train_batch(batch, global_step, progress_bar)

        return global_step


    def _train_batch(self, batch, global_step, progress_bar):
        computation_result = self._train_step(batch)

        self._accumulate_training_results(computation_result)

        new_step = self.model_engine.global_steps
        if new_step <= global_step:
            return global_step
        
        global_step = new_step

        averaged_result = self._get_averaged_results()
        averaged_result = self.gather_scalar_for_log(averaged_result)

        if self.rank == 0:
            progress_bar.update(1)
            self._handle_logging_and_progress(global_step, averaged_result)
        
        if global_step % self.config.save_steps == 0:
            self._save_checkpoint(global_step)

        self._reset_accumulated_results()
        
        return global_step
    
    def _get_accumulation_keys(self):
        return {"loss"}
    
    def _accumulate_training_results(self, computation_result):
        accumulation_keys = self._get_accumulation_keys()
        self._accumulate_single_value("loss", computation_result["loss"])
        if "metrics" in computation_result:
            for key, value in computation_result["metrics"].items():
                self._accumulate_single_value(key, value, should_accumulate=(key in accumulation_keys))
    
    def _accumulate_single_value(self, key, value, should_accumulate=True):
        if should_accumulate:
            if key not in self.tr_accumulated:
                self.tr_accumulated[key] = torch.tensor(0.0, device=self.model_engine.device)
            self.tr_accumulated[key] += value
        else:
            self.tr_accumulated[key] = value
    
    def _get_averaged_results(self):
        accumulation_keys = self._get_accumulation_keys()
        result = {}
        
        if "loss" in self.tr_accumulated:
            result["loss"] = self.tr_accumulated["loss"] / self.config.gradient_accumulation_steps
        
        result["metrics"] = {}
        for key, accumulated_value in self.tr_accumulated.items():
            if key == "loss":
                continue
            
            if key in accumulation_keys:
                result["metrics"][key] = accumulated_value / self.config.gradient_accumulation_steps
            else:
                result["metrics"][key] = accumulated_value
        
        return result
    
    def _reset_accumulated_results(self):
        for key in self.tr_accumulated:
            self.tr_accumulated[key].zero_()

    def _handle_logging_and_progress(self, global_step, computation_result):
        if global_step % self.config.logging_steps == 0:
            lr = self.model_engine.get_lr()[0]
            grad_norm = self.model_engine.get_global_grad_norm()
            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.item()
            epochs_done = min(global_step / self.num_update_steps_per_epoch, math.ceil(self.config.num_epochs))
            self.log_metrics(lr, grad_norm, epochs_done, global_step, computation_result)
        

    def _train_step(self, batch):
        batch = {k: v.to(self.model_engine.device) if isinstance(
            v, torch.Tensor) else v for k, v in batch.items()}
        computation_result = self.compute_loss(self.model_engine, batch)
        loss = computation_result["loss"]
        self.model_engine.backward(loss)
        self.model_engine.step()
        computation_result["loss"] = loss.detach().clone()
        return computation_result
    

    def compute_loss(self, model_engine, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        output = model_engine(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        if self.config.average_tokens_across_devices:
            count = (labels != -100).sum().float()
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            loss = ForCausalLMLoss(
                logits=output.logits,
                labels=labels,
                vocab_size=self.vocab_size,
                num_items_in_batch=count
            ) * self.world_size
        else:
            loss = ForCausalLMLoss(
                logits=output.logits,
                labels=labels,
                vocab_size=self.vocab_size
            )
        return dict(loss = loss)
    

    def gather_scalar_for_log(self, computation_result):
        loss_tensor = computation_result["loss"]
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        computation_result["avg_loss"] = (loss_tensor / self.world_size).item()
        return computation_result


    def log_metrics(self, lr, grad_norm, epochs_done, global_step, computation_result):
        avg_loss = computation_result["avg_loss"]
        msg = {
            "step": global_step,
            "loss": avg_loss,
            "learning_rate": lr,
            "grad_norm": grad_norm,
            "epoch": epochs_done
        }
        self.training_metrics.append(msg)
        self._print_log(f"step: {global_step}/{self.max_steps} | loss: {avg_loss:.4f} | lr: {lr:.2e} | grad_norm: {grad_norm:.4f} | epoch: {epochs_done:.2f}")
        
        if self.tb_writer is not None:
            writer = self.tb_writer
            writer.add_scalar("Train/Loss", avg_loss, global_step)
            writer.add_scalar("Train/LearningRate", lr, global_step)
            writer.add_scalar("Train/GradNorm", grad_norm, global_step)

    def _save_training_metrics(self):
        if self.rank != 0:
            return

        metrics_path = os.path.join(self.config.output_dir, "training_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.training_metrics, f, indent=4, ensure_ascii=False)

        self._print_log(f"Training metrics saved to {metrics_path}")


    def plot_training_loss(self):
        if self.rank != 0:
            return
        
        steps = [item["step"] for item in self.training_metrics]
        losses = [item["loss"] for item in self.training_metrics]
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

        self._print_log(f"Training loss saved to {self.config.output_dir}")


    def _print_log(self, msg: str):
        if self.rank != 0:
            return

        tqdm.write(msg)
        self.logger.info(msg)
    

    def _cleanup(self):
        self.plot_training_loss()
        self._save_training_metrics()

        if self.rank == 0 and self.tb_writer is not None:
            self.tb_writer.close()
            self._print_log("TensorBoard writer closed.")

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


    def _print_train_parameters(self):
        self._print_log("=" * 80)
        self._print_log("           RUNNING TRAINING")
        self._print_log("=" * 80)
        self._print_log(f"  Num examples:          {len(self.train_dataset):>12,}")
        self._print_log(f"  Num epochs:            {self.config.num_epochs:>12,}")
        self._print_log(f"  Batch size per device: {self.config.train_micro_batch_size_per_gpu:>12,}")
        self._print_log(f"  Gradient accumulation: {self.config.gradient_accumulation_steps:>12}")
        self._print_log(f"  Total batch size:      {self.total_train_batch_size:>12,}")
        self._print_log(f"  Steps per epoch:       {self.num_update_steps_per_epoch:>12,}")
        self._print_log(f"  Total training steps:  {self.max_steps:>12,}")
        self._print_log(f"  Warmup steps:          {self.warmup_steps:>12,}")

        enabled_features = []

        enabled_features.append(f"Training stage: {self.config.training_stage}")
        enabled_features.append(f"Precision: {self.config.train_model_precision}")
        enabled_features.append(f"Learning rate scheduler: {self.config.lr_scheduler_type}")

        if self.config.finetuning_type == "lora":
            enabled_features.append(
                f"LoRA: enabled (r={self.config.lora_rank}, alpha={self.config.lora_alpha}, "
                f"dropout={self.config.lora_dropout}, modules={self.config.lora_target_modules})"
            )
        else:
            enabled_features.append("LoRA: disabled (full fine-tuning or other method)")

        if self.config.num_local_io_workers:
            enabled_features.append(f"Data loader workers: custom ({self.config.num_local_io_workers})")
        else:
            enabled_features.append("Data loader workers: auto (2 × GPU count)")

        if self.config.enable_gradient_checkpointing:
            enabled_features.append("Gradient checkpointing: enabled")
        else:
            enabled_features.append("Gradient checkpointing: disabled")

        if is_flash_attn_2_available():
            enabled_features.append("Attention backend: FlashAttention-2")
        else:
            enabled_features.append("Attention backend: Eager (FlashAttention-2 not available)")

        if getattr(self.config, "use_tensorboard", False):
            enabled_features.append("TensorBoard logging: enabled")
        else:
            enabled_features.append("TensorBoard logging: disabled")

        if self.config.average_tokens_across_devices:
            enabled_features.append("Loss normalization: average tokens across all devices")
        else:
            enabled_features.append("Loss normalization: per-device token count (no cross-device averaging)")

        if self.config.save_last:
            enabled_features.append("Final checkpoint: saved after training")
        else:
            enabled_features.append("Final checkpoint: NOT saved (save_last=False)")

        if self.config.save_train_log:
            enabled_features.append("Training log: saved to file")
        else:
            enabled_features.append("Training log: stdout only (not saved to file)")

        zero_stage = self.deepspeed_config.get("zero_optimization", {}).get("stage", 0)
        enabled_features.append(f"DeepSpeed Zero Stage: {zero_stage}")

        self._print_log("  Enabled features / key configurations:")
        for feat in enabled_features:
            self._print_log(f"    - {feat}")
        self._print_log("=" * 80)