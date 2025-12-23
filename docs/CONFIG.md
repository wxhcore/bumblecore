# BumbleCore Configuration Parameters

BumbleCore supports configuration through YAML configuration files or command-line arguments. This document details all available configuration parameters.

> 💡 **Configuration Priority**: Command-line arguments > YAML config file > Default values

---

## 📋 Table of Contents

- [Training Parameters](#training-parameters)
  - [Basic Configuration](#🎯-basic-configuration)
  - [Model and Tokenizer](#🤖-model-and-tokenizer)
  - [Dataset Configuration](#📊-dataset-configuration)
  - [Training Hyperparameters](#🎓-training-hyperparameters)
  - [Batch Size](#💾-batch-size)
  - [Precision Settings](#🔢-precision-settings)
  - [DeepSpeed and Distributed](#🚀-deepspeed-and-distributed)
  - [Save and Checkpoint](#💾-save-and-checkpoint-1)
  - [Logging Configuration](#📈-logging-configuration)
  - [LoRA Configuration](#🔧-lora-configuration)
  - [DPO Specific Parameters](#🎯-dpo-specific-parameters)
- [Inference Parameters](#inference-parameters)
  - [Model Loading Configuration](#🤖-model-loading-configuration-1)
  - [Conversation Basic Configuration](#💬-conversation-basic-configuration)
  - [Sampling Generation Parameters](#🎲-sampling-generation-parameters)
  - [Web Service Configuration](#🌐-web-service-configuration-bumblechat-only)

---

## Training Parameters

### 🎯 Basic Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `yaml_config` | str | "" | YAML configuration file path |
| `training_stage` | str | "sft" | Training stage: `pretrain` (pretraining), `continue_pretrain` (continual pretraining), `sft` (supervised fine-tuning), `dpo` (direct preference optimization) |
| `finetuning_type` | str | "full" | Fine-tuning type: `full` (full parameter fine-tuning), `lora` (LoRA fine-tuning) |
| `output_dir` | str | "./output" | Output directory for models and logs |

**Usage Notes:**
- `training_stage` determines data processing method and loss function calculation
- When `finetuning_type` is `lora`, [LoRA related parameters](#-lora-configuration) must be configured

---

### 🤖 Model and Tokenizer

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name_or_path` | str | None | Model path or HuggingFace model name (required) |
| `trust_remote_code` | bool | False | Whether to trust remote code (needed for custom models) |
| `tokenizer_use_fast` | bool | False | Whether to use fast tokenizer |

---

### 📊 Dataset Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_path` | str | None | Dataset file path, supports `.json` and `.jsonl` formats (required) |
| `cutoff_len` | int | 1024 | Maximum sequence length (tokens), will be truncated if exceeded |

---

### 🎓 Training Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_epochs` | float | 3.0 | Number of training epochs, supports decimals (e.g., 0.5 for half epoch) |
| `learning_rate` | float | 5e-5 | Learning rate |
| `weight_decay` | float | 0.01 | Weight decay coefficient for L2 regularization |
| `warmup_ratio` | float | 0.1 | Warmup step ratio, percentage of total training steps |
| `lr_scheduler_type` | str | "cosine" | Learning rate scheduler type: `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `constant`, `constant_with_warmup`, `inverse_sqrt`, `reduce_on_plateau`, `cosine_with_min_lr`, `warmup_stable_decay`, etc. |
| `enable_gradient_checkpointing` | bool | False | Whether to enable gradient checkpointing (reduces memory usage but increases training time) |

---

### 💾 Batch Size

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_micro_batch_size_per_gpu` | int | 4 | Micro batch size per GPU |
| `gradient_accumulation_steps` | int | 8 | Gradient accumulation steps<br>**Global batch size** = `micro_batch_size × num_GPUs × gradient_accumulation_steps` |

---

### 🔢 Precision Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_model_precision` | str | "bf16" | Training precision: `fp32` (32-bit float), `fp16` (16-bit float), `bf16` (Brain Float16) |

---

### 🚀 DeepSpeed and Distributed

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `deepspeed_config_path` | str | None | DeepSpeed configuration file path (recommended) |
| `num_local_io_workers` | int | None | Number of local I/O worker threads |
| `average_tokens_across_devices` | bool | False | Whether to average token count across devices (for load balancing) |
| `local_rank` | int | -1 | Local GPU rank for distributed training (automatically set by DeepSpeed) |

**DeepSpeed Configuration Options:**

| Config File | ZeRO Stage | Memory Optimization | Speed | Use Case |
|-------------|------------|---------------------|-------|----------|
| `ds_z0_config.json` | Stage 0 | None | Fastest | Small models, sufficient memory |
| `ds_z1_config.json` | Stage 1 | Optimizer state sharding | Fast | Medium models |
| `ds_z2_config.json` | Stage 2 | Optimizer + gradient sharding | Medium | Large models |
| `ds_z3_config.json` | Stage 3 | Full parameter sharding | Slower | Very large models, limited memory |

---

### 💾 Save and Checkpoint

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_steps` | int | 500 | Save checkpoint every N steps |
| `save_total_limit` | int | 3 | Maximum number of checkpoints to keep (old ones will be deleted) |
| `save_last` | bool | False | Whether to save the last checkpoint at the end of training |

---

### 📈 Logging Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `logging_steps` | int | 1 | Log every N steps |
| `save_train_log` | bool | False | Whether to save training logs to file |
| `use_tensorboard` | bool | False | Whether to enable TensorBoard logging |

---

### 🔧 LoRA Configuration

> Only required when `finetuning_type="lora"`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lora_rank` | int | 64 | LoRA rank, controls adapter size |
| `lora_alpha` | int | 128 | LoRA scaling factor, usually set to `rank × 2` |
| `lora_dropout` | float | 0.1 | Dropout rate for LoRA layers |
| `lora_target_modules` | list | None | Target modules to apply LoRA, e.g., `["q_proj", "v_proj"]` |

---

### 🎯 DPO Specific Parameters

> Only used when `training_stage="dpo"`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pref_beta` | float | 0.1 | Beta coefficient for DPO loss, controls preference strength |
| `dpo_label_smoothing` | float | 0.0 | Label smoothing coefficient |
| `sft_weight` | float | 0.0 | SFT loss weight (used when mixing with DPO loss) |
| `ld_alpha` | float | 1.0 | Alpha coefficient for LD (Length Difference) loss |

---

## Inference Parameters

### 🤖 Model Loading Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `yaml_config` | str | "" | YAML configuration file path |
| `model_path` | str | None | Model path (required) |
| `device_map` | str | "auto" | Device mapping: `auto` (automatic), `cpu`, `cuda:0`, etc. |
| `dtype` | str | "auto" | Model data type: `auto`, `torch.float16`, `torch.bfloat16`, etc. |
| `training_stage` | str | "sft" | Model training stage: `sft`, `dpo`, `pretrain` |

---

### 💬 Conversation Basic Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `system_prompt` | str | None | System prompt to set model role and behavior |
| `enable_history` | bool | False | Whether to enable multi-turn conversation history memory |
| `max_new_tokens` | int | None | Maximum number of tokens to generate (None uses model default) |

---

### 🎲 Sampling Generation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `do_sample` | bool | False | Whether to enable sampling (False means greedy decoding) |
| `temperature` | float | None | Sampling temperature, controls generation randomness<br>- Lower values (0.1-0.5): More deterministic<br>- Higher values (0.7-1.5): More creative |
| `top_k` | int | None | Top-K sampling, sample only from K most probable tokens |
| `top_p` | float | None | Top-P (Nucleus) sampling, stop when cumulative probability reaches P |
| `repetition_penalty` | float | None | Repetition penalty coefficient, value >1.0 reduces repetition<br>- 1.0: No penalty<br>- 1.2: Light penalty<br>- 1.5+: Heavy penalty |

---

### 🌐 Web Service Configuration (BumbleChat Only)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | str | "127.0.0.1" | Web server listening address<br>- `127.0.0.1`: Local access only<br>- `0.0.0.0`: Allow external access |
| `port` | int | 8000 | Web server port number |

