# BumbleCore 数据格式说明

本文档详细说明 BumbleCore 支持的各种数据格式，帮助你准备训练数据。

> 💡 **格式支持**：所有训练阶段均支持 **JSON** 和 **JSONL** 两种文件格式，框架会自动识别并处理。

---

## 📋 目录

- [预训练数据格式](#1️⃣-预训练数据格式)
- [SFT（监督微调）数据格式](#2️⃣-sft监督微调数据格式)
  - [Alpaca 格式](#格式一alpaca-格式)
  - [ShareGPT 格式](#格式二sharegpt-格式)
    - [工具调用支持](#工具调用支持sharegpt)
  - [Messages 格式（OpenAI 风格）](#格式三messages-格式openai-风格)
- [DPO（直接偏好优化）数据格式](#3️⃣-dpo直接偏好优化数据格式)
  - [Alpaca 格式](#格式一alpaca-格式-1)
  - [ShareGPT 格式](#格式二sharegpt-格式-1)
  - [Messages 格式（OpenAI 风格）](#格式三messages-格式openai-风格-1)
- [数据准备建议](#📝-数据准备建议)

> 🔍 **格式自动识别**：数据 formatter 通过检查首条样本的顶层字段自动识别格式，优先级为 `messages` → `conversations` → `instruction`。请确保单个文件内格式一致。

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

SFT 阶段支持三种格式：**Alpaca**、**ShareGPT**，以及 OpenAI 风格的 **Messages**。

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
| `system` | string | ❌ | 自定义系统提示词，默认为 "You are Bumblebee, a helpful AI assistant." |

> Alpaca 格式不支持工具调用。如需训练工具调用数据，请使用 ShareGPT 或 OpenAI 风格的 Messages 格式。

#### 完整格式示例（包含 system）

```json
    {
        "system": "你是一个专业的数学导师",
        "instruction": "解这个方程",
        "input": "x + 2 = 5",
        "output": "x = 3"
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
| `tools` | string/list | ❌ | 工具定义，通常与 `function_call` / `observation` 一起用于工具调用；单独提供 `tools` 不代表发生了工具调用 |

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

#### 工具调用支持（ShareGPT）

ShareGPT 通过两个额外的 `from` 取值支持工具调用：

| `from` | 说明 |
|--------|------|
| `function_call` | 模型调用工具，`value` 为包含 `name` 和 `arguments` 字段的 JSON 字符串。 |
| `observation`   | 工具的返回结果，**必须**紧跟在 `function_call` 或另一条 `observation` 之后。 |

这两种角色会在数据预处理阶段自动转换为 OpenAI / Qwen 的工具调用结构（`assistant.tool_calls` 与 `role: "tool"`），再交给 chat template。

```json
[
  {
    "conversations": [
      {"from": "human", "value": "今天北京的天气怎么样？"},
      {"from": "function_call", "value": "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"北京\"}}"},
      {"from": "observation", "value": "{\"city\": \"北京\", \"temperature\": 18, \"condition\": \"晴\"}"},
      {"from": "gpt", "value": "北京今天晴，约 18°C。"}
    ],
    "tools": "[{\"name\": \"get_weather\", \"description\": \"查询城市天气\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}]"
  }
]
```

> ⚠️ 结构不合法的样本（例如 `observation` 之前没有 `function_call`）会在加载时打印 warning 并被跳过。可参考真实样例：`datasets/glaive_toolcall_zh_demo.json`。

### 格式三：Messages 格式（OpenAI 风格）

如果你的数据已经是 OpenAI / Qwen 的 `messages` 形式（例如直接从对话 API 导出），可以直接使用该格式。**这是工具调用类数据的推荐格式**，无需做角色转换。

#### 基础格式

```json
[
  {
    "messages": [
      {"role": "user", "content": "用一句话解释什么是大语言模型。"},
      {"role": "assistant", "content": "大语言模型是一种基于海量文本训练、能够理解和生成自然语言的深度神经网络模型。"}
    ]
  }
]
```

#### 字段说明

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `messages` | list | ✅ | OpenAI 风格的消息列表 |
| `messages[].role` | string | ✅ | `"system"` / `"user"` / `"assistant"` / `"tool"` 之一 |
| `messages[].content` | string \| null | 视情况 | 大多数情况下必填。`assistant` 消息**仅当**带 `tool_calls` 时 `content` 才可以为 `null`；`content` 和 `tool_calls` 也可以**同时**出现（见下文"混合 assistant 消息"）。 |
| `messages[].tool_calls` | list | ❌ | assistant 触发工具调用时使用 |
| `tools` | string/list | ❌ | 工具定义，通常与 `messages` 中的 `tool_calls` 配套使用 |

如果首条消息不是 `system`，会自动在最前面注入默认 system prompt（`"You are Bumblebee, a helpful AI assistant."`）。

#### 混合 assistant 消息（content + tool_calls）

一条 `assistant` 消息可以**同时**携带文字 `content`（例如对用户的简短说明 / 推理）和 `tool_calls`。Qwen chat template 会在同一个 `<|im_start|>assistant ... <|im_end|>` 块内先输出 content，再输出 `<tool_call>...</tool_call>`：

```json
{
  "role": "assistant",
  "content": "好的，我先帮您查一下北京当前天气。",
  "tool_calls": [
    {
      "type": "function",
      "function": {"name": "get_weather", "arguments": {"city": "北京"}}
    }
  ]
}
```

如果你希望模型在调用工具前先用一句话告诉用户要做什么，**推荐**采用这种写法。

#### 工具调用示例

```json
[
  {
    "messages": [
      {"role": "user", "content": "今天北京的天气怎么样？"},
      {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "type": "function",
            "function": {"name": "get_weather", "arguments": {"city": "北京", "unit": "celsius"}}
          }
        ]
      },
      {"role": "tool", "content": "{\"city\": \"北京\", \"temperature\": 18, \"condition\": \"晴\"}"},
      {"role": "assistant", "content": "北京今天晴，约 18°C。"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "查询城市天气",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string"},
              "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
          }
        }
      }
    ]
  }
]
```

> 💡 完整可运行示例参见 `datasets/messages_zh_demo.json`。

---

## 3️⃣ DPO（直接偏好优化）数据格式

DPO 数据包含 **chosen**（偏好回复）和 **rejected**（非偏好回复）两个回复，支持三种格式：**Alpaca**、**ShareGPT**，以及 OpenAI 风格的 **Messages**。

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

> Alpaca 格式不支持工具调用。如需训练工具调用数据，请使用 ShareGPT 或 OpenAI 风格的 Messages 格式。

#### 完整格式示例

```json
[
  {
    "system": "你是一个富有诗意的诗人",
    "instruction": "写一首关于海洋的诗",
    "input": "",
    "chosen": "浪花轻吟千古曲，潮汐诉说万年情。碧波荡漾映星月，深邃无垠藏乾坤。",
    "rejected": "大海很大，也很蓝。"
  }
]
```

### 格式二：ShareGPT 格式

ShareGPT DPO 使用 `conversations` 表示候选回复之前的对话历史，`chosen` / `rejected` 表示同一上下文下的两条候选最终回复。

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
| `conversations[].from` | string | ✅ | `"system"` / `"human"` / `"gpt"`；工具调用历史还支持 `"function_call"` / `"observation"` |
| `conversations[].value` | string | ✅ | 对话内容；`function_call` 的值为包含 `name` 和 `arguments` 的 JSON 字符串 |
| `chosen` | object | ✅ | 偏好的最后一轮回复 |
| `chosen.from` | string | ✅ | 通常为 `"gpt"` |
| `chosen.value` | string | ✅ | 偏好回复内容 |
| `rejected` | object | ✅ | 非偏好的最后一轮回复 |
| `rejected.from` | string | ✅ | 通常为 `"gpt"` |
| `rejected.value` | string | ✅ | 非偏好回复内容 |
| `tools` | string/list | ❌ | 工具定义，只有在 `conversations` 中包含 `function_call` / `observation` 时才表示工具调用训练样本 |

#### 工具调用示例

```json
[
  {
    "conversations": [
      {"from": "human", "value": "今天北京的天气怎么样？"},
      {"from": "function_call", "value": "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"北京\"}}"},
      {"from": "observation", "value": "{\"city\": \"北京\", \"temperature\": 18, \"condition\": \"晴\"}"}
    ],
    "chosen": {
      "from": "gpt",
      "value": "北京今天晴，约 18°C，适合外出。"
    },
    "rejected": {
      "from": "gpt",
      "value": "我不知道北京天气。"
    },
    "tools": "[{\"name\": \"get_weather\", \"description\": \"查询城市天气\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}]"
  }
]
```

### 格式三：Messages 格式（OpenAI 风格）

DPO 数据也可以使用 OpenAI 风格的 `messages` 形式：`messages` 承载候选回复之前的完整对话历史，`chosen` / `rejected` 分别给出两条候选 assistant 回复。候选回复可以是纯文本，也可以包含 `tool_calls`。

#### 基础格式（`chosen` / `rejected` 为字符串）

```json
[
  {
    "messages": [
      {"role": "user", "content": "请帮我把下面这段中文翻译成英文：人工智能正在改变我们的生活方式。"}
    ],
    "chosen": "Artificial intelligence is changing the way we live.",
    "rejected": "AI change life."
  }
]
```

`chosen` / `rejected` 也可以写成 `{"role": "assistant", "content": "..."}` 对象形式；如果候选回复需要调用工具，也可以在对象中包含 `tool_calls`。

#### 工具调用示例

```json
[
  {
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "查询城市天气",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string"}
            },
            "required": ["city"]
          }
        }
      }
    ],
    "messages": [
      {"role": "user", "content": "今天北京的天气怎么样？"}
    ],
    "chosen": {
      "role": "assistant",
      "content": "我来查一下北京当前天气。",
      "tool_calls": [
        {
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": {"city": "北京"}
          }
        }
      ]
    },
    "rejected": {
      "role": "assistant",
      "content": "北京今天应该挺好，适合外出。"
    }
  }
]
```

#### 字段说明

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `messages` | list | ✅ | 候选回复之前的对话历史，可包含历史中的 `assistant.tool_calls` 和 `role: "tool"` |
| `chosen` | string \| object | ✅ | 偏好的 assistant 回复，字符串或 `{"role": "assistant", "content": "...", "tool_calls": [...]}` 形式均可 |
| `rejected` | string \| object | ✅ | 非偏好的最终 assistant 回复，形式同 `chosen` |
| `tools` | string/list | ❌ | 工具定义，通常与 `messages` 中的 `tool_calls` 配套使用 |

> 💡 完整可运行示例参见 `datasets/dpo_messages_zh_demo.json`。

## 📝 数据准备建议

1. **质量优于数量**：高质量、格式规范的数据比海量噪声数据更有价值。
2. **格式一致**：单个文件内的所有样本应保持同一种格式，避免混用。
3. **预校验**：训练前先验证 JSON / JSONL 文件结构，可以提前暴露格式错误。
4. **DPO 对比有效**：`chosen` 和 `rejected` 应在质量、风格或正确性上有明显差异。
5. **覆盖多样性**：包含不同任务、领域和边界场景的样本，提升泛化能力。
6. **工具调用数据**：建议直接使用 Messages 格式编写，结构与 chat template 完全对齐，最不容易出错。

