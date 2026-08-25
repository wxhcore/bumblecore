from dataclasses import dataclass, field
from typing import Optional, List, Union


@dataclass
class TrainConfig:

    model_name_or_path: Optional[str] = field(default=None)
    dataset_path: Optional[str] = field(default=None)
    output_dir: str = field(default="./output")


    trust_remote_code: bool = field(default=False)
    tokenizer_use_fast: bool = field(default=False)
    cutoff_len: int = field(default=1024)

    learning_rate: float = field(default=5e-5)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.1)
    num_epochs: float = field(default=3.0)
    lr_scheduler_type: str = field(default="cosine")

    train_micro_batch_size_per_gpu: int = field(default=4)
    gradient_accumulation_steps: int = field(default=8)
    enable_gradient_checkpointing: bool = field(default=False)

    train_model_precision: str = field(default="bf16")

    num_local_io_workers: Optional[int] = field(default=None)

    save_steps: int = field(default=500)
    save_total_limit: Optional[int] = field(default=3)
    save_last: bool = field(default=False)
    logging_steps: int = field(default=1)
    save_train_log: bool = field(default=False)
    use_tensorboard: bool = field(default=False)

    average_tokens_across_devices: bool = field(default=False)
    deepspeed_config_path: Optional[str] = field(default=None)

    finetuning_type: str = field(default="full") 
    training_stage: str = field(default="sft") 

    lora_rank: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.1)
    lora_target_modules: Optional[Union[List[str], str]] = field(default=None)

    # Packing settings
    packing: bool = field(default=False)
    packing_num_proc: int = field(default=1)


    ld_alpha: float = field(default=1.0)       
    pref_beta: float = field(default=0.1)             
    dpo_label_smoothing: float = field(default=0.0)   
    sft_weight: float = field(default=0.0)    

    def __post_init__(self):
        if not self.model_name_or_path or (isinstance(self.model_name_or_path, str) and not self.model_name_or_path.strip()):
            raise ValueError("model_name_or_path must be specified and non-empty.")
        if not self.dataset_path or (isinstance(self.dataset_path, str) and not self.dataset_path.strip()):
            raise ValueError("dataset_path must be specified and non-empty.")
        if not self.output_dir or (isinstance(self.output_dir, str) and not self.output_dir.strip()):
            raise ValueError("output_dir must be specified and non-empty.")

        if isinstance(self.lora_target_modules, str):
            modules = [m.strip() for m in self.lora_target_modules.split(",") if m.strip()]
            self.lora_target_modules = modules if modules else None

        allowed_precisions = {"fp32", "fp16", "bf16"}
        if self.train_model_precision not in allowed_precisions:
            raise ValueError(
                f"train_model_precision must be one of {allowed_precisions}, got '{self.train_model_precision}'"
            )

        allowed_sft_types = {"full", "lora"}
        if self.finetuning_type not in allowed_sft_types:
            raise ValueError(
                f"finetuning_type must be one of {allowed_sft_types}, got '{self.finetuning_type}'"
            )