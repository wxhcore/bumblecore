# BumbleCore Three-Stage Training Tutorial

This tutorial will take you from scratch through the complete **Pretraining → Supervised Fine-Tuning → Preference Optimization** three-stage training process to train your own 1.5B parameter language model.

## 🎯 Training Pipeline Diagram

```
Pretraining
├─ Input: Large-scale text data (1B tokens)
├─ Output: Base Model
└─ Goal: Learn foundational language knowledge and expression capabilities
          ↓
Supervised Fine-Tuning (SFT)
├─ Input: Instruction-response pairs (42K samples) + Base Model
├─ Output: Instruct Model
└─ Goal: Learn to follow instructions and conversational abilities
          ↓
Preference Optimization (DPO)
├─ Input: Preference comparison data (10K samples) + Instruct Model
├─ Output: Aligned Model
└─ Goal: Align with human preferences, improve response quality
```
> 💡 **Tip**: For convenient debugging and quick pipeline validation, **minimal example datasets** required for all training and testing stages are provided in the project directory.  
> You can directly use samples in [`../datasets`](../datasets) for end-to-end testing.

---

## 🚀 Stage 1: Pretraining

**Experiment Goal**: Start from random initialization and teach the model foundational language knowledge.

### 1️⃣ Prepare Pretraining Data

We use the mini_pretrain_dataset from ModelScope, containing approximately **1B tokens** of Chinese pretraining data.

**Dataset URL**: [https://www.modelscope.cn/datasets/BazingaLyn/mini_pretrain_dataset/files](https://www.modelscope.cn/datasets/BazingaLyn/mini_pretrain_dataset/files)


**Input Data Format Example**:

```json
{"text": "Artificial intelligence is a branch of computer science that attempts to understand the essence of intelligence..."}
{"text": "Machine learning is a method of implementing AI, and deep learning is a subset of machine learning..."}
```

### 2️⃣ Configure Training Parameters

Edit the pretraining script `scripts/pretrain.sh`:

**Override YAML parameters via command line, modify dataset path**:

```bash
#!/bin/bash
deepspeed --include localhost:0,1 src/train.py \
    --yaml_config ./configs/pretrain/pretrain_full.yaml \
    --model_name_or_path ./models/bumblebee \
    --dataset_path <your dataset path> \
    --output_dir ./checkpoints/pretrain/bumblebee_1.5b_base
```

### 3️⃣ Start Training

```bash
bash scripts/pretrain.sh
```

### 4️⃣ Monitor Training Progress

During training, you can use TensorBoard to monitor training metrics in real-time:

```bash
# Start TensorBoard (run in a new terminal)
tensorboard --logdir=./checkpoints/pretrain/bumblebee_1.5b_base
```
> 💡 **Note**: The subsequent SFT and DPO stages can also use TensorBoard to monitor training. Just modify the `--logdir` parameter to the corresponding output directory.


**Final Training Loss Curve:**

![Pretraining Loss Curve](../assets/train_loss/pretrain_training_loss.png)

---

## 📝 Stage 2: Supervised Fine-Tuning (SFT)

**Experiment Goal**: Teach the base model to understand and follow human instructions.

### 1️⃣ Prepare SFT Data

We use the high-quality Chinese instruction dataset alpaca_gpt4_zh provided by LLaMA Factory, containing **42,677** instruction-response pairs.

**Dataset URL**: [https://huggingface.co/datasets/llamafactory/alpaca_gpt4_zh](https://huggingface.co/datasets/llamafactory/alpaca_gpt4_zh)

**Input Data Format Example** (Alpaca format):

```json
{
  "instruction": "Three tips for staying healthy.",
  "input": "",
  "output": "Here are three tips for staying healthy:\n1. Stay physically active. Do appropriate physical exercise daily...\n2. Eat a balanced diet. Consume fresh vegetables and fruits daily...\n3. Get sufficient sleep. Sleep is crucial for human health..."
}
```

### 2️⃣ Configure Training Parameters

Edit the SFT script `scripts/sft_full.sh`:

```bash
#!/bin/bash
deepspeed --include localhost:0,1 src/train.py \
    --yaml_config ./configs/sft/sft_full.yaml \
    --model_name_or_path ./checkpoints/pretrain/bumblebee_1.5b_base \
    --dataset_path <your dataset path> \
    --output_dir ./checkpoints/sft/bumblebee_1.5b_Instruct_full \
    --num_epochs 6.0
```

> 💡 **Important**: `--model_name_or_path` now points to Stage 1's output, enabling continuous model training

### 3️⃣ Start Training

```bash
bash scripts/sft_full.sh
```

**Final Training Loss Curve:**

![SFT Training Loss](../assets/train_loss/sft_training_loss.png)


---

## 🎯 Stage 3: Preference Optimization (DPO)

**Experiment Goal**: Use human preference data to make model responses higher quality and safer.

### 1️⃣ Prepare DPO Data

We use the **Chinese portion** of the bilingual preference dataset DPO-En-Zh-20k provided by LLaMA Factory, containing approximately **10,000** preference comparison data entries.

**Dataset URL**: [https://huggingface.co/datasets/llamafactory/DPO-En-Zh-20k](https://huggingface.co/datasets/llamafactory/DPO-En-Zh-20k)

**Data Format Example** (ShareGPT format with chosen/rejected):

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Tell me about Beijing"
    },
    {
      "from": "gpt",
      "value": "Beijing is the capital of the People's Republic of China, with over 3000 years of city history and over 860 years as a capital. As the nation's political center, cultural center, and international exchange center..."
    }
  ],
  "chosen": {
    "from": "gpt",
    "value": "Beijing is the capital of the People's Republic of China, with over 3000 years of city history and over 860 years as a capital. As the nation's political center, cultural center, and international exchange center..."
  },
  "rejected": {
    "from": "gpt",
    "value": "Beijing is a city."
  }
}
```

### 2️⃣ Configure Training Parameters

Edit the DPO script `scripts/dpo_lora.sh`:

```bash
#!/bin/bash
deepspeed --include localhost:0,1,2,3 src/train.py \
    --yaml_config ./configs/dpo/dpo_full.yaml \
    --model_name_or_path ./checkpoints/sft/bumblebee_1.5b_Instruct_full \
    --dataset_path <your dataset path> \
    --output_dir ./checkpoints/dpo/bumblebee_1.5b_dpo_lora
```

> 💡 **Important**: `--model_name_or_path` points to Stage 2's output, continuing model optimization

### 3️⃣ Start Training

```bash
bash scripts/dpo_lora.sh
```

**Final Training Loss Curve:**

![DPO Training Loss](../assets/train_loss/dpo_training_loss.png)

**Accuracy Curve:**

![DPO Training Rewards and Accuracies](../assets/train_loss/dpo_training_rewards_accuracies.png)


### 4️⃣ Merge LoRA Weights

Since the DPO stage used LoRA training, the training output is LoRA adapter weights that need to be merged back into the base model for use.

```bash
# Edit the merge script
vim tools/run_merge_lora.sh

# Configure the following parameters:
# --base_model_path ./checkpoints/sft/bumblebee_1.5b_Instruct_full
# --lora_model_path ./checkpoints/dpo/bumblebee_1.5b_dpo_lora
# --save_path ./checkpoints/dpo/bumblebee_1.5b_dpo_merged

# Execute the merge
bash tools/run_merge_lora.sh
```

> 💡 **Tip**: The merge process will load the base model and LoRA weights, requiring some GPU memory. After merging, the `save_path` directory will contain the complete model weights and tokenizer.

---

### 5️⃣ Test the Model

After training, you can test the model's conversational abilities:

```bash
# Edit scripts/chat.sh, set the model path
vim scripts/chat.sh
# Modify to: --model_path ./checkpoints/dpo/bumblebee_1.5b_dpo_merged

# Start conversation test
bash scripts/chat.sh
```

