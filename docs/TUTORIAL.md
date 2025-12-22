# BumbleCore 三阶段训练实验教程

本教程将带你从零开始，完整体验**预训练 → 监督微调 → 偏好优化**三阶段训练流程，训练一个属于你自己的 1.5B 参数语言模型。

## 🎯 训练流程图

```
预训练 (Pretraining)
├─ 输入：大规模文本数据 (1B tokens)
├─ 输出：基座模型 (Base Model)
└─ 目标：学习语言的基础知识和表达能力
          ↓
监督微调 (SFT)
├─ 输入：指令-回复对数据 (42K samples) + 基座模型
├─ 输出：指令模型 (Instruct Model)
└─ 目标：学习遵循指令和对话能力
          ↓
偏好优化 (DPO)
├─ 输入：偏好对比数据 (10K samples) + 指令模型
├─ 输出：对齐模型 (Aligned Model)
└─ 目标：对齐人类偏好，提升回复质量
```
> 💡 **提示**：为方便调试和快速验证流程，所有训练与测试阶段所需的**最小示例数据集**均已提供在项目目录中。  
> 你可以直接使用 [`../datasets`](../datasets)中的样本进行端到端测试。

---

## 🚀 阶段一：预训练（Pretraining）

**实验目的**：从随机初始化开始，让模型学习语言的基础知识。

### 1️⃣ 准备预训练数据

我们使用 ModelScope 上的 mini_pretrain_dataset 数据集，包含约 **1B tokens** 的中文预训练数据。

**数据集地址**：[https://www.modelscope.cn/datasets/BazingaLyn/mini_pretrain_dataset/files](https://www.modelscope.cn/datasets/BazingaLyn/mini_pretrain_dataset/files)


**输入数据格式示例**：

```json
{"text": "人工智能是计算机科学的一个分支，它企图了解智能的实质..."}
{"text": "机器学习是实现人工智能的一种方法，深度学习是机器学习的子集..."}
```

### 2️⃣ 配置训练参数

编辑预训练脚本 `scripts/pretrain.sh`：

**使用命令行覆盖 YAML 中的参数，修改数据集地址**：

```bash
#!/bin/bash
deepspeed --include localhost:0,1 src/train.py \
    --yaml_config ./configs/pretrain/pretrain_full.yaml \
    --model_name_or_path ./models/bumblebee \
    --dataset_path <your dataset path> \
    --output_dir ./checkpoints/pretrain/bumblebee_1.5b_base
```

### 3️⃣ 启动训练

```bash
bash scripts/pretrain.sh
```

### 4️⃣ 监控训练过程

训练过程中，你可以使用 TensorBoard 实时监控训练指标：

```bash
# 启动 TensorBoard（在新终端中执行）
tensorboard --logdir=./checkpoints/pretrain/bumblebee_1.5b_base
```
> 💡 **注意**：后续的 SFT 和 DPO 阶段也可以使用相同的方式启动 TensorBoard 监控训练过程，只需修改 `--logdir` 参数为对应的输出目录即可。


**最终训练损失曲线：**

![预训练损失曲线](../assets/train_loss/pretrain_training_loss.png)

---

## 📝 阶段二：监督微调（SFT）

**实验目的**：让基座模型学会理解和遵循人类指令。

### 1️⃣ 准备 SFT 数据

我们使用 LLaMA Factory 提供的高质量中文指令数据集 alpaca_gpt4_zh，包含 **42,677** 条指令-回复对。

**数据集地址**：[https://huggingface.co/datasets/llamafactory/alpaca_gpt4_zh](https://huggingface.co/datasets/llamafactory/alpaca_gpt4_zh)

**输入数据格式示例**（Alpaca 格式）：

```json
{
  "instruction": "保持健康的三个提示。",
  "input": "",
  "output": "以下是保持健康的三个提示：\n1. 保持身体活动。每天做适当的身体运动...\n2. 均衡饮食。每天食用新鲜的蔬菜、水果...\n3. 睡眠充足。睡眠对人体健康至关重要..."
}
```

### 2️⃣ 配置训练参数

编辑 SFT 脚本 `scripts/sft_full.sh`：

```bash
#!/bin/bash
deepspeed --include localhost:0,1 src/train.py \
    --yaml_config ./configs/sft/sft_full.yaml \
    --model_name_or_path ./checkpoints/pretrain/bumblebee_1.5b_base \
    --dataset_path <your dataset path> \
    --output_dir ./checkpoints/sft/bumblebee_1.5b_Instruct_full \
    --num_epochs 6.0
```

> 💡 **重要**：`--model_name_or_path` 现在指向阶段一的输出，实现模型的连续训练

### 3️⃣ 启动训练

```bash
bash scripts/sft_full.sh
```

**最终训练损失曲线：**

![SFT 训练损失](../assets/train_loss/sft_training_loss.png)


---

## 🎯 阶段三：偏好优化（DPO）

**实验目的**：通过人类偏好数据，让模型的回复更加优质和安全。

### 1️⃣ 准备 DPO 数据

我们使用 LLaMA Factory 提供的双语偏好数据集 DPO-En-Zh-20k 的**中文部分**，包含约 **10,000** 条偏好对比数据。

**数据集地址**：[https://huggingface.co/datasets/llamafactory/DPO-En-Zh-20k](https://huggingface.co/datasets/llamafactory/DPO-En-Zh-20k)

**数据格式示例**（ShareGPT 格式 with chosen/rejected）：

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "介绍一下北京"
    },
    {
      "from": "gpt",
      "value": "北京是中华人民共和国的首都，有着3000多年的建城史和860多年的建都史。作为全国的政治中心、文化中心和国际交往中心..."
    }
  ],
  "chosen": {
    "from": "gpt",
    "value": "北京是中华人民共和国的首都，有着3000多年的建城史和860多年的建都史。作为全国的政治中心、文化中心和国际交往中心..."
  },
  "rejected": {
    "from": "gpt",
    "value": "北京是个城市。"
  }
}
```

### 2️⃣ 配置训练参数

编辑 DPO 脚本 `scripts/dpo_lora.sh`：

```bash
#!/bin/bash
deepspeed --include localhost:0,1,2,3 src/train.py \
    --yaml_config ./configs/dpo/dpo_full.yaml \
    --model_name_or_path ./checkpoints/sft/bumblebee_1.5b_Instruct_full \
    --dataset_path <your dataset path> \
    --output_dir ./checkpoints/dpo/bumblebee_1.5b_dpo_lora
```

> 💡 **重要**：`--model_name_or_path` 指向阶段二的输出，继续优化模型

### 3️⃣ 启动训练

```bash
bash scripts/dpo_lora.sh
```

**最终训练损失曲线：**

![DPO 训练损失](../assets/train_loss/dpo_training_loss.png)

**准确率曲线：**

![DPO 训练奖励和准确率](../assets/train_loss/dpo_training_rewards_accuracies.png)


### 4️⃣ 合并 LoRA 权重

由于 DPO 阶段使用了 LoRA 训练，训练后得到的是 LoRA 适配器权重，需要将其合并到基座模型中才能使用。

```bash
# 编辑合并脚本
vim tools/run_merge_lora.sh

# 配置以下参数：
# --base_model_path ./checkpoints/sft/bumblebee_1.5b_Instruct_full
# --lora_model_path ./checkpoints/dpo/bumblebee_1.5b_dpo_lora
# --save_path ./checkpoints/dpo/bumblebee_1.5b_dpo_merged

# 执行合并
bash tools/run_merge_lora.sh
```

> 💡 **提示**：合并过程会加载基座模型和 LoRA 权重，需要一定的显存。合并完成后，`save_path` 目录将包含完整的模型权重和分词器。

---

### 5️⃣ 测试模型

训练完成后，可以测试模型的对话能力：

```bash
# 编辑 scripts/chat.sh，设置模型路径
vim scripts/chat.sh
# 修改为：--model_path ./checkpoints/dpo/bumblebee_1.5b_dpo_merged

# 启动对话测试
bash scripts/chat.sh
```