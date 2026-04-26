# BumbleCore Data Format Guide

This document details the various data formats supported by BumbleCore to help you prepare training data.

> 💡 **Format Support**: All training stages support both **JSON** and **JSONL** file formats, with automatic recognition and processing.

---

## 📋 Table of Contents

- [Pretraining Data Format](#1️⃣-pretraining-data-format)
- [SFT (Supervised Fine-Tuning) Data Format](#2️⃣-sft-supervised-fine-tuning-data-format)
  - [Alpaca Format](#format-1-alpaca-format)
  - [ShareGPT Format](#format-2-sharegpt-format)
    - [Tool Calling Support](#tool-calling-support-sharegpt)
  - [Messages Format (OpenAI-style)](#format-3-messages-format-openai-style)
- [DPO (Direct Preference Optimization) Data Format](#3️⃣-dpo-direct-preference-optimization-data-format)
  - [Alpaca Format](#format-1-alpaca-format-1)
  - [ShareGPT Format](#format-2-sharegpt-format-1)
  - [Messages Format (OpenAI-style)](#format-3-messages-format-openai-style-1)
- [Data Preparation Tips](#📝-data-preparation-tips)

> 🔍 **Format auto-detection**: The data formatter detects the input format by inspecting the first sample's top-level keys, in priority order: `messages` → `conversations` → `instruction`. Make sure each file uses a single consistent format.

---

## 1️⃣ Pretraining Data Format

### Format:

```json
{"text": "This is the first pretraining text content..."}
{"text": "This is the second pretraining text content..."}
{"text": "This is the third pretraining text content..."}
```

### Field Description

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | ✅ | Pretraining text content, can be articles, code, conversations, or any text |

---

## 2️⃣ SFT (Supervised Fine-Tuning) Data Format

SFT stage supports three formats: **Alpaca**, **ShareGPT**, and OpenAI-style **Messages**.

### Format 1: Alpaca Format

Alpaca format is a concise instruction-input-output triplet format.

#### Basic Format

```json
  {
    "instruction": "Explain what machine learning is",
    "input": "",
    "output": "Machine learning is a branch of artificial intelligence..."
  },
  {
    "instruction": "Translate the following English to Chinese",
    "input": "Hello, how are you?",
    "output": "你好，你好吗？"
  }
```

#### Field Description

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `instruction` | string | ✅ | User's instruction or question |
| `input` | string | ❌ | Supplementary input content, can be omitted or filled with `""` if empty |
| `output` | string | ✅ | Model's response content |
| `system` | string | ❌ | Custom system prompt, defaults to "You are Bumblebee, a helpful AI assistant." |

> Alpaca format does not support tool calling. Use ShareGPT or OpenAI-style Messages format for tool-calling data.

#### Complete Format Example (with system)

```json
    {
        "system": "You are a professional math tutor",
        "instruction": "Solve this equation",
        "input": "x + 2 = 5",
        "output": "x = 3"
    }
```

### Format 2: ShareGPT Format

ShareGPT format is a conversational format supporting multi-turn dialogues.

#### Basic Format

```json

  {
    "conversations": [
      {"from": "human", "value": "Hello"},
      {"from": "gpt", "value": "Hello! How can I help you?"}
    ]
  },
  {
    "conversations": [
      {"from": "human", "value": "Explain quantum computing"},
      {"from": "gpt", "value": "Quantum computing is a type of computing that uses quantum mechanics principles..."}
    ]
  }

```

#### Field Description

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `conversations` | list | ✅ | Conversation list containing multi-turn dialogues |
| `conversations[].from` | string | ✅ | Role identifier: `"system"` / `"human"` / `"gpt"` |
| `conversations[].value` | string | ✅ | Conversation content |
| `tools` | string/list | ❌ | Tool definitions, usually used with `function_call` / `observation` for tool calling. Providing `tools` alone does not mean a tool call happened. |

#### Multi-turn Conversation Example

```json
[
  {
    "conversations": [
      {"from": "system", "value": "You are a helpful AI assistant"},
      {"from": "human", "value": "What is deep learning?"},
      {"from": "gpt", "value": "Deep learning is a subfield of machine learning based on multi-layer neural networks..."},
      {"from": "human", "value": "What are its applications?"},
      {"from": "gpt", "value": "Deep learning has wide applications in image recognition, natural language processing, speech recognition, and more."}
    ]
  }
]
```

#### Tool Calling Support (ShareGPT)

ShareGPT format supports tool calling via two extra `from` values:

| `from` | Description |
|--------|-------------|
| `function_call` | The assistant invokes a tool. `value` is a JSON string with `name` and `arguments`. |
| `observation`   | Tool's return value. Must immediately follow a `function_call` or another `observation`. |

These two roles are converted internally to the OpenAI / Qwen tool-calling structure (`assistant.tool_calls` and `role: "tool"`) before being passed to the chat template.

```json
[
  {
    "conversations": [
      {"from": "human", "value": "What's the weather in Beijing?"},
      {"from": "function_call", "value": "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Beijing\"}}"},
      {"from": "observation", "value": "{\"city\": \"Beijing\", \"temperature\": 18, \"condition\": \"sunny\"}"},
      {"from": "gpt", "value": "It's sunny in Beijing today, around 18°C."}
    ],
    "tools": "[{\"name\": \"get_weather\", \"description\": \"Look up weather for a city\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}]"
  }
]
```

> ⚠️ Samples with malformed tool-calling structure (e.g. an `observation` not preceded by a `function_call`) are skipped with a warning at load time. See `datasets/glaive_toolcall_zh_demo.json` for a real example.

### Format 3: Messages Format (OpenAI-style)

If your data is already in OpenAI / Qwen `messages` form (e.g. exported from a chat API), you can use it directly. This is the recommended format for tool-calling data because it requires no role translation.

#### Basic Format

```json
[
  {
    "messages": [
      {"role": "user", "content": "Explain large language models in one sentence."},
      {"role": "assistant", "content": "An LLM is a deep neural network trained on massive text corpora to understand and generate natural language."}
    ]
  }
]
```

#### Field Description

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `messages` | list | ✅ | OpenAI-style message list |
| `messages[].role` | string | ✅ | One of `"system"` / `"user"` / `"assistant"` / `"tool"` |
| `messages[].content` | string \| null | depends | Required for most roles. For an `assistant` message, `content` may be `null` **only if** `tool_calls` is set. `content` and `tool_calls` may also coexist - see "mixed assistant message" below. |
| `messages[].tool_calls` | list | ❌ | Used when an assistant message invokes tools |
| `tools` | string/list | ❌ | Tool definitions, usually used with `tool_calls` in `messages` |

If no `system` message exists at the start, the default system prompt (`"You are Bumblebee, a helpful AI assistant."`) is automatically prepended.

#### Mixed assistant message (text + tool_calls)

An `assistant` message may carry **both** a textual `content` (e.g. a brief reasoning or status update for the user) **and** `tool_calls` in the same message. The Qwen chat template renders the text first, followed by `<tool_call>...</tool_call>` blocks within the same `<|im_start|>assistant ... <|im_end|>` turn:

```json
{
  "role": "assistant",
  "content": "Sure, let me check the weather in Beijing for you.",
  "tool_calls": [
    {
      "type": "function",
      "function": {"name": "get_weather", "arguments": {"city": "Beijing"}}
    }
  ]
}
```

This is the recommended pattern when you want the model to learn to verbalize its intent before invoking a tool.

#### Tool Calling Example

```json
[
  {
    "messages": [
      {"role": "user", "content": "What's the weather in Beijing?"},
      {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "type": "function",
            "function": {"name": "get_weather", "arguments": {"city": "Beijing", "unit": "celsius"}}
          }
        ]
      },
      {"role": "tool", "content": "{\"city\": \"Beijing\", \"temperature\": 18, \"condition\": \"sunny\"}"},
      {"role": "assistant", "content": "It's sunny in Beijing today, about 18°C."}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Look up weather for a city",
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

> 💡 See `datasets/messages_zh_demo.json` for runnable end-to-end examples.

---

## 3️⃣ DPO (Direct Preference Optimization) Data Format

DPO data contains **chosen** (preferred response) and **rejected** (non-preferred response) replies, supporting three formats: **Alpaca**, **ShareGPT**, and OpenAI-style **Messages**.

### Format 1: Alpaca Format

#### Basic Format

```json
[
  {
    "instruction": "Write a poem about spring",
    "input": "",
    "chosen": "Spring breeze caresses the blooming garden fair, Butterflies dance gracefully through the air. Green willows sway to greet the warming sun, Birds sing and swallows soar, their joy has just begun.",
    "rejected": "Spring has come, flowers bloom, it's beautiful."
  }
]
```

#### Field Description

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `instruction` | string | ✅ | User's instruction or question |
| `input` | string | ❌ | Supplementary input content |
| `chosen` | string | ✅ | Better response (preferred response) |
| `rejected` | string | ✅ | Worse response (non-preferred response) |
| `system` | string | ❌ | Custom system prompt |

> Alpaca format does not support tool calling. Use ShareGPT or OpenAI-style Messages format for tool-calling data.

#### Complete Format Example

```json
[
  {
    "system": "You are a poetic poet",
    "instruction": "Write a poem about the ocean",
    "input": "",
    "chosen": "Waves gently hum their ancient tune, Tides tell tales of a thousand moons. Blue waters dance with stars aglow, Deep and vast, where secrets flow.",
    "rejected": "The ocean is big and also blue."
  }
]
```

### Format 2: ShareGPT Format

ShareGPT DPO uses `conversations` for the prompt history before the candidate response, and `chosen` / `rejected` for two competing final assistant responses under the same context.

#### Basic Format

```json
[
  {
    "conversations": [
      {"from": "human", "value": "Hello"},
      {"from": "gpt", "value": "Hello!"},
      {"from": "human", "value": "How are you today?"}
    ],
    "chosen": {"from": "gpt", "value": "I'm doing great, thanks for asking! I've had a very productive day."},
    "rejected": {"from": "gpt", "value": "Okay I guess."}
  }
]
```

#### Field Description

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `conversations` | list | ✅ | Conversation history containing previous multi-turn dialogues |
| `conversations[].from` | string | ✅ | `"system"` / `"human"` / `"gpt"`; tool-calling history also supports `"function_call"` / `"observation"` |
| `conversations[].value` | string | ✅ | Message content. For `function_call`, this is a JSON string with `name` and `arguments`. |
| `chosen` | object | ✅ | Preferred final response |
| `chosen.from` | string | ✅ | Usually `"gpt"` |
| `chosen.value` | string | ✅ | Preferred response content |
| `rejected` | object | ✅ | Non-preferred final response |
| `rejected.from` | string | ✅ | Usually `"gpt"` |
| `rejected.value` | string | ✅ | Non-preferred response content |
| `tools` | string/list | ❌ | Tool definitions. This is a tool-calling training sample only when `conversations` contains `function_call` / `observation`. |

#### Tool Calling Example

```json
[
  {
    "conversations": [
      {"from": "human", "value": "What's the weather in Beijing today?"},
      {"from": "function_call", "value": "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Beijing\"}}"},
      {"from": "observation", "value": "{\"city\": \"Beijing\", \"temperature\": 18, \"condition\": \"sunny\"}"}
    ],
    "chosen": {
      "from": "gpt",
      "value": "It's sunny in Beijing today, around 18°C, so it should be comfortable to go outside."
    },
    "rejected": {
      "from": "gpt",
      "value": "I don't know the weather in Beijing."
    },
    "tools": "[{\"name\": \"get_weather\", \"description\": \"Look up weather for a city\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}]"
  }
]
```

### Format 3: Messages Format (OpenAI-style)

You can also feed DPO data in the OpenAI-style `messages` form. The `messages` array carries the full prompt history before the candidate response, and `chosen` / `rejected` carry the two competing assistant responses. Candidate responses may be plain text or include `tool_calls`.

#### Basic Format (string `chosen` / `rejected`)

```json
[
  {
    "messages": [
      {"role": "user", "content": "Translate to English: 人工智能正在改变我们的生活方式。"}
    ],
    "chosen": "Artificial intelligence is changing the way we live.",
    "rejected": "AI change life."
  }
]
```

`chosen` / `rejected` may also use the object form `{"role": "assistant", "content": "..."}`. If a candidate response should invoke a tool, include `tool_calls` in that object.

#### Tool Calling Example

```json
[
  {
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Look up weather for a city",
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
      {"role": "user", "content": "What's the weather in Beijing today?"}
    ],
    "chosen": {
      "role": "assistant",
      "content": "Let me check the current weather in Beijing.",
      "tool_calls": [
        {
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": {"city": "Beijing"}
          }
        }
      ]
    },
    "rejected": {
      "role": "assistant",
      "content": "The weather in Beijing should be nice today."
    }
  }
]
```

#### Field Description

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `messages` | list | ✅ | Conversation history before the candidate response. May include historical `assistant.tool_calls` and `role: "tool"`. |
| `chosen` | string \| object | ✅ | Preferred assistant response. String, or `{"role": "assistant", "content": "...", "tool_calls": [...]}` |
| `rejected` | string \| object | ✅ | Non-preferred final assistant response, same form as `chosen` |
| `tools` | string/list | ❌ | Tool definitions, usually used with `tool_calls` in `messages` |

> 💡 See `datasets/dpo_messages_zh_demo.json` for runnable end-to-end examples.

## 📝 Data Preparation Tips

1. **Quality over Quantity**: High-quality, well-formatted data is more valuable than large amounts of noisy data.
2. **Consistent Formatting**: All samples within a single file should use the same format - do not mix formats.
3. **Validation**: Validate your JSON/JSONL files before training to catch formatting errors.
4. **Balance**: For DPO, ensure chosen and rejected responses are meaningfully different.
5. **Diversity**: Include diverse examples covering different use cases and edge cases.
6. **Tool-calling data**: Prefer the Messages format for tool-calling samples - it maps 1:1 to the chat template and is the least error-prone.

