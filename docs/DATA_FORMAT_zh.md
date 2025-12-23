# BumbleCore 数据格式说明

本文档详细说明 BumbleCore 支持的各种数据格式，帮助你准备训练数据。

> 💡 **格式支持**：所有训练阶段均支持 **JSON** 和 **JSONL** 两种文件格式，框架会自动识别并处理。

---

## 📋 目录

- [预训练数据格式](#1️⃣-预训练数据格式)
- [SFT（监督微调）数据格式](#2️⃣-sft监督微调数据格式)
  - [Alpaca 格式](#格式一alpaca-格式)
  - [ShareGPT 格式](#格式二sharegpt-格式)
- [DPO（直接偏好优化）数据格式](#3️⃣-dpo直接偏好优化数据格式)
  - [Alpaca 格式](#格式一alpaca-格式-1)
  - [ShareGPT 格式](#格式二sharegpt-格式-1)
- [数据准备建议](#📝-数据准备建议)

---

## 1️⃣ 预训练数据格式

### 格式：

```json
{"text": "这是第一段预训练文本内容..."}
{"text": "这是第二段预训练文本内容..."}
{"text": "这是第三段预训练文本内容..."}
```

### 字段说明

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `text` | string | ✅ | 预训练的文本内容，可以是文章、代码、对话等任何文本 |

---

## 2️⃣ SFT（监督微调）数据格式

SFT 阶段支持两种主流数据格式：**Alpaca** 和 **ShareGPT**。

### 格式一：Alpaca 格式

Alpaca 格式是一种简洁的指令-输入-输出三元组格式。

#### 基础格式

```json
  {
    "instruction": "解释什么是机器学习",
    "input": "",
    "output": "机器学习是人工智能的一个分支..."
  },
  {
    "instruction": "将以下英文翻译成中文",
    "input": "Hello, how are you?",
    "output": "你好，你好吗？"
  }
```

#### 字段说明

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `instruction` | string | ✅ | 用户的指令或问题 |
| `input` | string | ❌ | 补充输入内容，如果为空可以省略或填 `""` |
| `output` | string | ✅ | 模型的回复内容 |
| `system` | string | ❌ | 自定义系统提示词，默认为 "You are a helpful assistant." |
| `tools` | string/list | ❌ | 工具定义，JSON 字符串或列表格式 |

#### 完整格式示例（包含 system 和 tools）

```json
    {
        "system": "你是一个专业的数学导师",
        "instruction": "解这个方程",
        "input": "x + 2 = 5",
        "output": "x = 3",
        "tools": "[{\"name\": \"calculator\", \"description\": \"数学计算器\"}]"
    }
```

### 格式二：ShareGPT 格式

ShareGPT 格式是一种对话式格式，支持多轮对话。

#### 基础格式

```json

  {
    "conversations": [
      {"from": "human", "value": "你好"},
      {"from": "gpt", "value": "你好！有什么可以帮助你的吗？"}
    ]
  },
  {
    "conversations": [
      {"from": "human", "value": "解释一下量子计算"},
      {"from": "gpt", "value": "量子计算是一种利用量子力学原理..."}
    ]
  }

```

#### 字段说明

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `conversations` | list | ✅ | 对话列表，包含多轮对话 |
| `conversations[].from` | string | ✅ | 角色标识：`"system"` / `"human"` / `"gpt"` |
| `conversations[].value` | string | ✅ | 对话内容 |
| `tools` | string/list | ❌ | 工具定义 |

#### 多轮对话示例

```json
[
  {
    "conversations": [
      {"from": "system", "value": "你是一个有帮助的AI助手"},
      {"from": "human", "value": "什么是深度学习？"},
      {"from": "gpt", "value": "深度学习是机器学习的一个子领域，它基于多层神经网络..."},
      {"from": "human", "value": "它有哪些应用？"},
      {"from": "gpt", "value": "深度学习在图像识别、自然语言处理、语音识别等领域都有广泛应用。"}
    ]
  }
]
```

#### 带工具的示例

```json
[
  {
    "conversations": [
      {"from": "system", "value": "你是一个专业的翻译助手"},
      {"from": "human", "value": "将'Hello'翻译成法语"},
      {"from": "gpt", "value": "Bonjour"}
    ],
    "tools": "[{\"name\": \"translator\", \"description\": \"多语言翻译工具\"}]"
  }
]
```

---

## 3️⃣ DPO（直接偏好优化）数据格式

DPO 数据包含 **chosen**（偏好回复）和 **rejected**（非偏好回复）两个回复，**ShareGPT** 格式。

### 格式一：Alpaca 格式

#### 基础格式

```json
[
  {
    "instruction": "写一首关于春天的诗",
    "input": "",
    "chosen": "春风拂面花满园，蝴蝶翩翩舞翻天。绿柳轻摇迎暖日，莺歌燕舞乐无边。",
    "rejected": "春天来了，花开了，很美。"
  }
]
```

#### 字段说明

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `instruction` | string | ✅ | 用户的指令或问题 |
| `input` | string | ❌ | 补充输入内容 |
| `chosen` | string | ✅ | 更好的回复（偏好回复） |
| `rejected` | string | ✅ | 较差的回复（非偏好回复） |
| `system` | string | ❌ | 自定义系统提示词 |
| `tools` | string/list | ❌ | 工具定义 |

#### 完整格式示例

```json
[
  {
    "system": "你是一个富有诗意的诗人",
    "instruction": "写一首关于海洋的诗",
    "input": "",
    "chosen": "浪花轻吟千古曲，潮汐诉说万年情。碧波荡漾映星月，深邃无垠藏乾坤。",
    "rejected": "大海很大，也很蓝。",
    "tools": "[{\"name\": \"rhyme_suggester\", \"description\": \"押韵建议工具\"}]"
  }
]
```

### 格式二：ShareGPT 格式

#### 基础格式

```json
[
  {
    "conversations": [
      {"from": "human", "value": "你好"},
      {"from": "gpt", "value": "你好！"},
      {"from": "human", "value": "你今天怎么样？"}
    ],
    "chosen": {"from": "gpt", "value": "我很好，谢谢关心！今天过得很充实。"},
    "rejected": {"from": "gpt", "value": "还行吧。"}
  }
]
```

#### 字段说明

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `conversations` | list | ✅ | 对话历史，包含前面的多轮对话 |
| `chosen` | object | ✅ | 偏好的最后一轮回复 |
| `chosen.from` | string | ✅ | 通常为 `"gpt"` |
| `chosen.value` | string | ✅ | 偏好回复内容 |
| `rejected` | object | ✅ | 非偏好的最后一轮回复 |
| `rejected.from` | string | ✅ | 通常为 `"gpt"` |
| `rejected.value` | string | ✅ | 非偏好回复内容 |
| `tools` | string/list | ❌ | 工具定义 |

#### 多轮对话示例

```json
[
  {
    "conversations": [
      {"from": "system", "value": "你是一个健康顾问"},
      {"from": "human", "value": "我最近总是感到疲劳"}
    ],
    "chosen": {
      "from": "gpt",
      "value": "感到疲劳可能有多种原因。建议你：1. 保证每天7-8小时睡眠；2. 适当运动，每天步行10-15分钟；3. 保持均衡饮食；4. 减少咖啡因摄入。如果持续疲劳，建议咨询医生。"
    },
    "rejected": {
      "from": "gpt",
      "value": "可能是你太懒了，多运动就好了。"
    },
    "tools": "[{\"name\": \"wellness_tips\", \"description\": \"基于证据的健康建议\"}]"
  }
]
```

