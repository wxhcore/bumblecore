# BumbleCore Data Format Guide

This document details the various data formats supported by BumbleCore to help you prepare training data.

> 💡 **Format Support**: All training stages support both **JSON** and **JSONL** file formats, with automatic recognition and processing.

---

## 📋 Table of Contents

- [Pretraining Data Format](#1️⃣-pretraining-data-format)
- [SFT (Supervised Fine-Tuning) Data Format](#2️⃣-sft-supervised-fine-tuning-data-format)
  - [Alpaca Format](#format-1-alpaca-format)
  - [ShareGPT Format](#format-2-sharegpt-format)
- [DPO (Direct Preference Optimization) Data Format](#3️⃣-dpo-direct-preference-optimization-data-format)
  - [Alpaca Format](#format-1-alpaca-format-1)
  - [ShareGPT Format](#format-2-sharegpt-format-1)
- [Data Preparation Tips](#📝-data-preparation-tips)

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

SFT stage supports two mainstream data formats: **Alpaca** and **ShareGPT**.

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
| `system` | string | ❌ | Custom system prompt, defaults to "You are a helpful assistant." |
| `tools` | string/list | ❌ | Tool definitions, JSON string or list format |

#### Complete Format Example (with system and tools)

```json
    {
        "system": "You are a professional math tutor",
        "instruction": "Solve this equation",
        "input": "x + 2 = 5",
        "output": "x = 3",
        "tools": "[{\"name\": \"calculator\", \"description\": \"Math calculator\"}]"
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
| `tools` | string/list | ❌ | Tool definitions |

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

#### Example with Tools

```json
[
  {
    "conversations": [
      {"from": "system", "value": "You are a professional translation assistant"},
      {"from": "human", "value": "Translate 'Hello' to French"},
      {"from": "gpt", "value": "Bonjour"}
    ],
    "tools": "[{\"name\": \"translator\", \"description\": \"Multi-language translation tool\"}]"
  }
]
```

---

## 3️⃣ DPO (Direct Preference Optimization) Data Format

DPO data contains **chosen** (preferred response) and **rejected** (non-preferred response) replies, supporting both **Alpaca** and **ShareGPT** formats.

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
| `tools` | string/list | ❌ | Tool definitions |

#### Complete Format Example

```json
[
  {
    "system": "You are a poetic poet",
    "instruction": "Write a poem about the ocean",
    "input": "",
    "chosen": "Waves gently hum their ancient tune, Tides tell tales of a thousand moons. Blue waters dance with stars aglow, Deep and vast, where secrets flow.",
    "rejected": "The ocean is big and also blue.",
    "tools": "[{\"name\": \"rhyme_suggester\", \"description\": \"Rhyme suggestion tool\"}]"
  }
]
```

### Format 2: ShareGPT Format

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
| `chosen` | object | ✅ | Preferred final response |
| `chosen.from` | string | ✅ | Usually `"gpt"` |
| `chosen.value` | string | ✅ | Preferred response content |
| `rejected` | object | ✅ | Non-preferred final response |
| `rejected.from` | string | ✅ | Usually `"gpt"` |
| `rejected.value` | string | ✅ | Non-preferred response content |
| `tools` | string/list | ❌ | Tool definitions |

#### Multi-turn Conversation Example

```json
[
  {
    "conversations": [
      {"from": "system", "value": "You are a health advisor"},
      {"from": "human", "value": "I've been feeling tired recently"}
    ],
    "chosen": {
      "from": "gpt",
      "value": "Fatigue can have multiple causes. I recommend: 1. Ensure 7-8 hours of sleep per night; 2. Exercise moderately, walk for 10-15 minutes daily; 3. Maintain a balanced diet; 4. Reduce caffeine intake. If fatigue persists, please consult a doctor."
    },
    "rejected": {
      "from": "gpt",
      "value": "You're probably just lazy, exercise more."
    },
    "tools": "[{\"name\": \"wellness_tips\", \"description\": \"Evidence-based health advice\"}]"
  }
]
```

## 📝 Data Preparation Tips

1. **Quality over Quantity**: High-quality, well-formatted data is more valuable than large amounts of noisy data.
2. **Consistent Formatting**: Ensure all data entries follow the same format structure.
3. **Validation**: Validate your JSON/JSONL files before training to catch formatting errors.
4. **Balance**: For DPO, ensure chosen and rejected responses are meaningfully different.
5. **Diversity**: Include diverse examples covering different use cases and edge cases.

