import argparse
import os
import yaml


def get_args():
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--yaml_config",
        type=str,
        default="",
        help="Path to the YAML config file",
    )
    cfg_args, _ = base_parser.parse_known_args()
    config_path = cfg_args.yaml_config
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    parser = argparse.ArgumentParser(
        description="Train with Trainer",
        parents=[base_parser],
    )

    # Model & Tokenizer
    parser.add_argument("--model_name_or_path", type=str, default=cfg.get("model_name_or_path"))
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=cfg.get("trust_remote_code", False)
    )
    parser.add_argument(
        "--tokenizer_use_fast", action="store_true", default=cfg.get("tokenizer_use_fast", False)
    )

    # Dataset
    parser.add_argument("--dataset_path", type=str, default=cfg.get("dataset_path"))
    parser.add_argument("--cutoff_len", type=int, default=cfg.get("cutoff_len", 1024))

    # Training
    parser.add_argument("--num_epochs", type=float, default=cfg.get("num_epochs", 3.0))
    parser.add_argument("--learning_rate", type=float, default=cfg.get("learning_rate", 5e-5))
    parser.add_argument("--weight_decay", type=float, default=cfg.get("weight_decay", 0.01))
    parser.add_argument("--lr_scheduler_type", type=str, default=cfg.get("lr_scheduler_type", "cosine"))
    parser.add_argument(
        "--enable_gradient_checkpointing",
        action="store_true",
        default=cfg.get("enable_gradient_checkpointing", False),
    )
    parser.add_argument("--packing", type=bool, default=cfg.get("packing", True))
    parser.add_argument("--packing_num_proc", type=int, default=cfg.get("packing_num_proc", 4))
    parser.add_argument("--warmup_ratio", type=float, default=cfg.get("warmup_ratio", 0.1))

    # Batch Size
    parser.add_argument(
        "--train_micro_batch_size_per_gpu",
        type=int,
        default=cfg.get("train_micro_batch_size_per_gpu", 4),
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=cfg.get("gradient_accumulation_steps", 8)
    )

    # Precision
    parser.add_argument(
        "--train_model_precision",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default=cfg.get("train_model_precision", "bf16"),
    )

    # DeepSpeed & IO
    parser.add_argument(
        "--deepspeed_config_path", type=str, default=cfg.get("deepspeed_config_path")
    )
    parser.add_argument("--num_local_io_workers", type=int, default=cfg.get("num_local_io_workers"))

    parser.add_argument("--local_rank", type=int, default=-1, help="用于分布式训练的本地GPU编号")

    parser.add_argument("--output_dir", type=str, default=cfg.get("output_dir", "./output"), help="模型和日志输出目录")

    # 保存 checkpoint
    parser.add_argument("--save_steps", type=int, default=cfg.get("save_steps", 500))
    parser.add_argument("--save_total_limit", type=int, default=cfg.get("save_total_limit", 3))
    parser.add_argument("--save_last", action="store_true", default=cfg.get("save_last", False))

    # 日志
    parser.add_argument("--logging_steps", type=int, default=cfg.get("logging_steps", 1))
    parser.add_argument("--save_train_log", action="store_true", default=cfg.get("save_train_log", False))
    parser.add_argument(
        "--use_tensorboard", action="store_true", default=cfg.get("use_tensorboard", False)
    )

    # 分布式/LoRA
    parser.add_argument(
        "--average_tokens_across_devices",
        action="store_true",
        default=cfg.get("average_tokens_across_devices", False),
    )
    parser.add_argument("--lora_rank", type=int, default=cfg.get("lora_rank", 64))
    parser.add_argument("--lora_alpha", type=int, default=cfg.get("lora_alpha", 128))
    parser.add_argument("--lora_dropout", type=float, default=cfg.get("lora_dropout", 0.1))
    parser.add_argument(
        "--lora_target_modules", type=str, nargs="+", default=cfg.get("lora_target_modules")
    )
    parser.add_argument(
        "--finetuning_type",
        type=str,
        default=cfg.get("finetuning_type", "full"),
        help="LoRA / full ",
    )
    parser.add_argument(
        "--training_stage",
        type=str,
        default=cfg.get("training_stage", "sft"),
        choices=["pretrain","continue_pretrain", "sft", "dpo"],
        help="Stage of training: pretrain, sft, or dpo",
    )

    parser.add_argument(
        "--ld_alpha",
        type=float,
        default=cfg.get("ld_alpha", 1.0),
        help="Alpha coefficient for LD loss (default: 1.0)",
    )
    parser.add_argument(
        "--pref_beta",
        type=float,
        default=cfg.get("pref_beta", 0.1),
        help="Beta coefficient for preference loss (default: 0.1)",
    )
    parser.add_argument(
        "--dpo_label_smoothing",
        type=float,
        default=cfg.get("dpo_label_smoothing", 0.0),
        help="Label smoothing factor (default: 0.0)",
    )
    parser.add_argument(
        "--sft_weight",
        type=float,
        default=cfg.get("sft_weight", 0.0),
        help="Weight for SFT loss when combined with DPO loss (default: 0.0)",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    print(args)
