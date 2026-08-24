import torch
import pytest
from transformers import AutoTokenizer
from bumblecore.data_processing import (
    PretrainDataset,
    SFTDataset,
    DPODataset,
    DataCollator,
    DPOCollator,
)

tokenizer = AutoTokenizer.from_pretrained("./models/bumblebee")


# ==============================
# PretrainDataset 测试
# ==============================

def test_pretrain_dataset():
    """测试 PretrainDataset 的端到端功能"""
    train_dataset = [
        {"text": "Hello"},
        {"text": "World"},
    ]
    
    max_length = 128
    
    dataset = PretrainDataset(train_dataset, tokenizer, max_length)
    result = dataset[0]
    
    expected_input_ids = torch.tensor([9707, 151645], dtype=torch.long)
    expected_attention_mask = torch.ones_like(expected_input_ids)
    expected_labels = expected_input_ids.clone()
    
    expected = {
        "input_ids": expected_input_ids,
        "attention_mask": expected_attention_mask,
        "labels": expected_labels,
    }
    
    assert torch.equal(result["input_ids"], expected["input_ids"])
    assert torch.equal(result["attention_mask"], expected["attention_mask"])
    assert torch.equal(result["labels"], expected["labels"])


def test_pretrain_dataset_with_eos_token():
    """测试 PretrainDataset 使用 eos_token"""
    train_dataset = [{"text": "Test"}]
    
    dataset = PretrainDataset(train_dataset, tokenizer, max_length=128)
    result = dataset[0]
    
    test_tokens = tokenizer("Test", return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
    expected_input_ids = torch.cat([test_tokens, torch.tensor([tokenizer.eos_token_id])])
    expected_attention_mask = torch.ones_like(expected_input_ids)
    expected_labels = expected_input_ids.clone()
    
    expected = {
        "input_ids": expected_input_ids,
        "attention_mask": expected_attention_mask,
        "labels": expected_labels,
    }
    
    assert torch.equal(result["input_ids"], expected["input_ids"])
    assert torch.equal(result["attention_mask"], expected["attention_mask"])
    assert torch.equal(result["labels"], expected["labels"])


def test_pretrain_dataset_with_bos_token():
    """测试 PretrainDataset 使用 bos_token（当没有 eos_token 时）"""
    train_dataset = [{"text": "Test"}]
    
    original_eos_token_id = tokenizer.eos_token_id
    original_eos_token = tokenizer.eos_token
    tokenizer.eos_token_id = None
    tokenizer.eos_token = None
    tokenizer.bos_token_id = 1
    tokenizer.bos_token = "<|bos|>"
    
    try:
        dataset = PretrainDataset(train_dataset, tokenizer, max_length=128)
        result = dataset[0]
        
        # 期望输出：文本 tokenized + <|im_end|>（当 eos_token_id 是 None 时，会添加 <|im_end|>）
        test_tokens = tokenizer("Test", return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
        im_end_tokens = tokenizer("<|im_end|>", return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
        expected_input_ids = torch.cat([test_tokens, im_end_tokens])
        expected_attention_mask = torch.ones_like(expected_input_ids)
        expected_labels = expected_input_ids.clone()
        
        expected = {
            "input_ids": expected_input_ids,
            "attention_mask": expected_attention_mask,
            "labels": expected_labels,
        }
        
        assert torch.equal(result["input_ids"], expected["input_ids"])
        assert torch.equal(result["attention_mask"], expected["attention_mask"])
        assert torch.equal(result["labels"], expected["labels"])
    finally:
        tokenizer.eos_token_id = original_eos_token_id
        tokenizer.eos_token = original_eos_token


# ==============================
# SFTDataset 测试
# ==============================

def test_sft_dataset():
    """测试 SFTDataset 的端到端功能"""
    train_dataset = [
        {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
            "tools": None,
        }
    ]
    
    max_length = 256
    
    dataset = SFTDataset(train_dataset, tokenizer, max_length)
    result = dataset[0]
    
    expected_input_ids = torch.tensor([
        151644, 8948, 198, 2610, 525, 10950, 13, 151645, 198, 151644, 872, 198, 
        13048, 151645, 198, 151644, 77091, 198, 9707, 151645, 198
    ], dtype=torch.long)
    expected_attention_mask = torch.ones_like(expected_input_ids)
    
    expected_labels = torch.tensor([
        -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        -100, -100, -100, -100, -100, -100, -100, -100, 9707, 151645, 198
    ], dtype=torch.long)
    
    expected = {
        "input_ids": expected_input_ids,
        "attention_mask": expected_attention_mask,
        "labels": expected_labels,
    }
    
    assert torch.equal(result["input_ids"], expected["input_ids"])
    assert torch.equal(result["attention_mask"], expected["attention_mask"])
    assert torch.equal(result["labels"], expected["labels"])


def test_sft_dataset_with_tools():
    """测试 SFTDataset 带 tools 的情况"""
    train_dataset = [
        {
            "messages": [
                {"role": "system", "content": "You are a math tutor."},
                {"role": "user", "content": "Solve 2+2"},
                {"role": "assistant", "content": "4"},
            ],
            "tools": [{"name": "calculator"}],
        }
    ]
    
    dataset = SFTDataset(train_dataset, tokenizer, max_length=256)
    result = dataset[0]
    
    assert len(result["input_ids"]) == 111
    assert len(result["labels"]) == 111
    
    non_negative_100 = (result["labels"] != -100).nonzero(as_tuple=True)[0]
    assert len(non_negative_100) == 3
    assert torch.equal(result["labels"][non_negative_100], torch.tensor([19, 151645, 198], dtype=torch.long))
    
    assert torch.equal(result["input_ids"][non_negative_100], result["labels"][non_negative_100])


def test_sft_dataset_with_tool_calls():
    """SFT 工具调用：assistant.tool_calls + role:tool 应通过 chat template，且仅 assistant 段被监督"""
    train_dataset = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's the weather in Beijing?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "Beijing"},
                            },
                        }
                    ],
                },
                {"role": "tool", "content": '{"temperature": 18, "condition": "sunny"}'},
                {"role": "assistant", "content": "It's sunny in Beijing, around 18°C."},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get city weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
        }
    ]

    dataset = SFTDataset(train_dataset, tokenizer, max_length=512)
    result = dataset[0]

    assert len(result["input_ids"]) == len(result["labels"]) == len(result["attention_mask"])
    assert (result["attention_mask"] == 1).all()

    decoded_full = tokenizer.decode(result["input_ids"])
    assert "<tool_call>" in decoded_full and "</tool_call>" in decoded_full
    assert "<tool_response>" in decoded_full and "</tool_response>" in decoded_full
    assert "get_weather" in decoded_full

    supervised_idx = (result["labels"] != -100).nonzero(as_tuple=True)[0]
    assert len(supervised_idx) > 0, "labels 全 -100，assistant 段未被监督"

    supervised_text = tokenizer.decode(result["labels"][supervised_idx])
    assert "get_weather" in supervised_text, "assistant tool_call 段应进入 labels"
    assert "Beijing" in supervised_text, "assistant 最终回答应进入 labels"

    assert "What's the weather in Beijing?" not in supervised_text
    assert "<tool_response>" not in supervised_text


def test_sft_dataset_assistant_text_with_tool_calls():
    """assistant content + tool_calls 共存：两部分都应被监督"""
    train_dataset = [
        {
            "messages": [
                {"role": "user", "content": "今天北京天气怎么样？"},
                {
                    "role": "assistant",
                    "content": "好的，我帮您查一下北京当前天气。",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "北京"},
                            },
                        }
                    ],
                },
                {"role": "tool", "content": '{"temperature": 18}'},
                {"role": "assistant", "content": "北京目前 18°C。"},
            ],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        }
    ]

    dataset = SFTDataset(train_dataset, tokenizer, max_length=512)
    result = dataset[0]

    decoded_full = tokenizer.decode(result["input_ids"])
    assert "好的，我帮您查一下北京当前天气。" in decoded_full
    assert "<tool_call>" in decoded_full

    supervised_idx = (result["labels"] != -100).nonzero(as_tuple=True)[0]
    supervised_text = tokenizer.decode(result["labels"][supervised_idx])

    assert "好的，我帮您查一下北京当前天气。" in supervised_text, \
        "assistant 文字 content 应进入 labels"
    assert "get_weather" in supervised_text, \
        "assistant tool_call 应进入 labels"
    assert "北京目前 18°C" in supervised_text, \
        "最终 assistant 回答应进入 labels"

    assert "今天北京天气怎么样？" not in supervised_text


def test_sft_dataset_multiple_tool_calls():
    """多轮工具调用：每个 assistant 段都应进入 labels，每个 tool 响应都应被 mask"""
    train_dataset = [
        {
            "messages": [
                {"role": "user", "content": "查一下北京和上海的天气。"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": {"city": "北京"}},
                        }
                    ],
                },
                {"role": "tool", "content": '{"temperature": 18}'},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": {"city": "上海"}},
                        }
                    ],
                },
                {"role": "tool", "content": '{"temperature": 22}'},
                {"role": "assistant", "content": "北京 18°C，上海 22°C。"},
            ],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        }
    ]

    dataset = SFTDataset(train_dataset, tokenizer, max_length=512)
    result = dataset[0]

    supervised_idx = (result["labels"] != -100).nonzero(as_tuple=True)[0]
    supervised_text = tokenizer.decode(result["labels"][supervised_idx])

    assert supervised_text.count("get_weather") == 2, \
        f"两个 tool_call 都应进入 labels，实际: {supervised_text!r}"
    assert "北京 18°C，上海 22°C。" in supervised_text
    assert "<tool_response>" not in supervised_text
    assert '"temperature": 18' not in supervised_text
    assert '"temperature": 22' not in supervised_text


def test_sft_dataset_max_length_truncation():
    """测试 SFTDataset 的 max_length 截断逻辑"""
    train_dataset = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. " * 10},  # 长 system 消息
                {"role": "user", "content": "Please explain machine learning in detail. " * 20},  # 长 user 消息
                {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence. " * 15},  # 长 assistant 回复
            ],
            "tools": None,
        }
    ]
    
    max_length = 50
    
    dataset = SFTDataset(train_dataset, tokenizer, max_length)
    result = dataset[0]
    
    assert len(result["input_ids"]) <= max_length, f"序列长度 {len(result['input_ids'])} 超过了 max_length {max_length}"
    assert len(result["attention_mask"]) <= max_length
    assert len(result["labels"]) <= max_length
    
    assert len(result["input_ids"]) == len(result["attention_mask"])
    assert len(result["input_ids"]) == len(result["labels"])
    
    dataset_no_truncation = SFTDataset(train_dataset, tokenizer, max_length=10000)
    result_no_truncation = dataset_no_truncation[0]
    original_length = len(result_no_truncation["input_ids"])
    
    if original_length > max_length:
        assert len(result["input_ids"]) == max_length, \
            f"当原始长度 {original_length} > max_length {max_length} 时，截断后长度应该等于 max_length"
    
    non_negative_100 = (result["labels"] != -100).nonzero(as_tuple=True)[0]
    if len(non_negative_100) > 0:

        assert torch.equal(
            result["input_ids"][non_negative_100], 
            result["labels"][non_negative_100]
        ), "input_ids 和 labels 在非 -100 位置的值应该相同"
        
        assert non_negative_100[-1] == len(result["labels"]) - 1 or \
               non_negative_100[-1] >= len(result["labels"]) - 2, \
               "labels 中非 -100 的部分应该延伸到序列末尾附近"
    
    valid_positions = result["attention_mask"] == 1
    assert valid_positions.sum().item() == len(result["input_ids"]), \
        "attention_mask 的有效位置数量应该等于序列长度"


# ==============================
# DPODataset 测试
# ==============================

def test_dpo_dataset():
    """测试 DPODataset 的端到端功能"""
    train_dataset = [
        {
            "chosen_messages": {
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Write a poem"},
                    {"role": "assistant", "content": "Roses are red"},
                ],
                "tools": None,
            },
            "rejected_messages": {
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Write a poem"},
                    {"role": "assistant", "content": "I don't know"},
                ],
                "tools": None,
            },
        }
    ]
    
    max_length = 256
    
    dataset = DPODataset(train_dataset, tokenizer, max_length)
    result = dataset[0]
    
    assert len(result["chosen_input_ids"]) == 26
    assert len(result["rejected_input_ids"]) == 26
    
    assert "chosen_input_ids" in result
    assert "chosen_attention_mask" in result
    assert "chosen_labels" in result
    assert isinstance(result["chosen_input_ids"], torch.Tensor)
    assert isinstance(result["chosen_attention_mask"], torch.Tensor)
    assert isinstance(result["chosen_labels"], torch.Tensor)
    
    assert "rejected_input_ids" in result
    assert "rejected_attention_mask" in result
    assert "rejected_labels" in result
    assert isinstance(result["rejected_input_ids"], torch.Tensor)
    assert isinstance(result["rejected_attention_mask"], torch.Tensor)
    assert isinstance(result["rejected_labels"], torch.Tensor)
    
    assert not torch.equal(result["chosen_input_ids"], result["rejected_input_ids"])


def test_dpo_dataset_from_messages_format():
    """DPODataset 端到端：使用 DataFormatter 从 messages 格式产出的结构"""
    from bumblecore.data_processing import DataFormatter

    raw = [
        {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Translate to English: 你好。"},
            ],
            "chosen": "Hello.",
            "rejected": "Hi.",
        }
    ]
    formatted = DataFormatter("dpo")(raw)

    dataset = DPODataset(formatted, tokenizer, max_length=256)
    result = dataset[0]

    chosen_supervised = (result["chosen_labels"] != -100).nonzero(as_tuple=True)[0]
    rejected_supervised = (result["rejected_labels"] != -100).nonzero(as_tuple=True)[0]
    assert len(chosen_supervised) > 0 and len(rejected_supervised) > 0

    chosen_text = tokenizer.decode(result["chosen_labels"][chosen_supervised])
    rejected_text = tokenizer.decode(result["rejected_labels"][rejected_supervised])
    assert "Hello." in chosen_text
    assert "Hi." in rejected_text

    assert "Translate to English" not in chosen_text
    assert "Translate to English" not in rejected_text



# ==============================
# DataCollator 测试
# ==============================

def test_data_collator():
    """测试 DataCollator 的端到端功能"""
    collator = DataCollator(tokenizer)
    
    batch = [
        {
            "input_ids": torch.tensor([1, 10, 20, 30, 2]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
            "labels": torch.tensor([1, 10, 20, 30, 2]),
        },
        {
            "input_ids": torch.tensor([1, 15, 25, 2]),
            "attention_mask": torch.tensor([1, 1, 1, 1]),
            "labels": torch.tensor([1, 15, 25, 2]),
        },
        {
            "input_ids": torch.tensor([1, 12, 22, 32, 42, 52, 2]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1, 1]),
            "labels": torch.tensor([1, 12, 22, 32, 42, 52, 2]),
        },
    ]
    
    result = collator(batch)
    
    pad_token_id = tokenizer.pad_token_id
    expected = {
        "input_ids": torch.tensor([
            [1, 10, 20, 30, 2, pad_token_id, pad_token_id],
            [1, 15, 25, 2, pad_token_id, pad_token_id, pad_token_id], 
            [1, 12, 22, 32, 42, 52, 2],
        ], dtype=torch.long),
        "attention_mask": torch.tensor([
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
        ], dtype=torch.long),
        "labels": torch.tensor([
            [1, 10, 20, 30, 2, -100, -100], 
            [1, 15, 25, 2, -100, -100, -100],
            [1, 12, 22, 32, 42, 52, 2],
        ], dtype=torch.long),
    }
    
    assert torch.equal(result["input_ids"], expected["input_ids"])
    assert torch.equal(result["attention_mask"], expected["attention_mask"])
    assert torch.equal(result["labels"], expected["labels"])


def test_data_collator_with_labels_negative_100():
    """测试 DataCollator 处理 labels 中的 -100（用于 SFT）"""
    collator = DataCollator(tokenizer)
    
    # 输入：labels 中包含 -100
    batch = [
        {
            "input_ids": torch.tensor([1, 10, 20, 30, 2]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
            "labels": torch.tensor([-100, -100, 20, 30, 2]),
        },
        {
            "input_ids": torch.tensor([1, 15, 25, 2]),
            "attention_mask": torch.tensor([1, 1, 1, 1]),
            "labels": torch.tensor([-100, -100, 25, 2]),
        },
    ]
    
    result = collator(batch)
    
    pad_token_id = tokenizer.pad_token_id
    expected = {
        "input_ids": torch.tensor([
            [1, 10, 20, 30, 2],
            [1, 15, 25, 2, pad_token_id],
        ], dtype=torch.long),
        "attention_mask": torch.tensor([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0],
        ], dtype=torch.long),
        "labels": torch.tensor([
            [-100, -100, 20, 30, 2],
            [-100, -100, 25, 2, -100],
        ], dtype=torch.long),
    }
    
    assert torch.equal(result["input_ids"], expected["input_ids"])
    assert torch.equal(result["attention_mask"], expected["attention_mask"])
    assert torch.equal(result["labels"], expected["labels"])


# ==============================
# DPOCollator 测试
# ==============================

def test_dpo_collator():
    """测试 DPOCollator 的端到端功能"""
    collator = DPOCollator(tokenizer)
    
    batch = [
        {
            "chosen_input_ids": torch.tensor([1, 10, 2]),  
            "chosen_attention_mask": torch.tensor([1, 1, 1]),
            "chosen_labels": torch.tensor([-100, 10, 2]),
            "rejected_input_ids": torch.tensor([1, 10, 20, 30, 40, 2]), 
            "rejected_attention_mask": torch.tensor([1, 1, 1, 1, 1, 1]),
            "rejected_labels": torch.tensor([-100, -100, 20, 30, 40, 2]),
        },
        {
            "chosen_input_ids": torch.tensor([1, 15, 25, 35, 2]),  
            "chosen_attention_mask": torch.tensor([1, 1, 1, 1, 1]),
            "chosen_labels": torch.tensor([-100, -100, 25, 35, 2]),
            "rejected_input_ids": torch.tensor([1, 15, 45, 2]),  
            "rejected_attention_mask": torch.tensor([1, 1, 1, 1]),
            "rejected_labels": torch.tensor([-100, -100, 45, 2]),
        },
    ]
    
    result = collator(batch)
    
    pad_token_id = tokenizer.pad_token_id
    expected = {
        "chosen_input_ids": torch.tensor([
            [1, 10, 2, pad_token_id, pad_token_id, pad_token_id],  
            [1, 15, 25, 35, 2, pad_token_id], 
        ], dtype=torch.long),
        "chosen_attention_mask": torch.tensor([
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0],
        ], dtype=torch.long),
        "chosen_labels": torch.tensor([
            [-100, 10, 2, -100, -100, -100],
            [-100, -100, 25, 35, 2, -100],
        ], dtype=torch.long),
        "rejected_input_ids": torch.tensor([
            [1, 10, 20, 30, 40, 2],  
            [1, 15, 45, 2, pad_token_id, pad_token_id],  
        ], dtype=torch.long),
        "rejected_attention_mask": torch.tensor([
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0],
        ], dtype=torch.long),
        "rejected_labels": torch.tensor([
            [-100, -100, 20, 30, 40, 2],
            [-100, -100, 45, 2, -100, -100],
        ], dtype=torch.long),
    }
    
    assert torch.equal(result["chosen_input_ids"], expected["chosen_input_ids"])
    assert torch.equal(result["chosen_attention_mask"], expected["chosen_attention_mask"])
    assert torch.equal(result["chosen_labels"], expected["chosen_labels"])
    assert torch.equal(result["rejected_input_ids"], expected["rejected_input_ids"])
    assert torch.equal(result["rejected_attention_mask"], expected["rejected_attention_mask"])
    assert torch.equal(result["rejected_labels"], expected["rejected_labels"])


# ==============================
# 端到端契约测试: DataFormatter -> Dataset
# 验证 formatter 输出格式恰好是 Dataset 期望的输入
# ==============================

def _supervised_text(result, key="labels"):
    """从 dataset 输出里取出被监督部分（labels != -100）的文本"""
    idx = (result[key] != -100).nonzero(as_tuple=True)[0]
    return tokenizer.decode(result[key][idx])


def test_chain_sft_alpaca_formatter_to_dataset():
    """SFT Alpaca: raw -> DataFormatter -> SFTDataset 端到端"""
    from bumblecore.data_processing import DataFormatter

    raw = [
        {
            "instruction": "Translate to French",
            "input": "Hello",
            "output": "Bonjour",
        }
    ]
    formatted = DataFormatter("sft")(raw)
    assert "messages" in formatted[0] and "tools" in formatted[0]

    dataset = SFTDataset(formatted, tokenizer, max_length=256)
    result = dataset[0]

    supervised = _supervised_text(result)
    assert "Bonjour" in supervised
    assert "Translate to French" not in supervised
    assert "Hello" not in supervised


def test_chain_sft_sharegpt_formatter_to_dataset():
    """SFT ShareGPT: raw -> DataFormatter -> SFTDataset 端到端"""
    from bumblecore.data_processing import DataFormatter

    raw = [
        {
            "conversations": [
                {"from": "system", "value": "You are helpful."},
                {"from": "human", "value": "What is 2+2?"},
                {"from": "gpt", "value": "4"},
                {"from": "human", "value": "And 3+3?"},
                {"from": "gpt", "value": "6"},
            ]
        }
    ]
    formatted = DataFormatter("sft")(raw)
    assert formatted[0]["messages"][0] == {"role": "system", "content": "You are helpful."}

    dataset = SFTDataset(formatted, tokenizer, max_length=256)
    result = dataset[0]

    supervised = _supervised_text(result)
    assert "4" in supervised and "6" in supervised
    assert "What is 2+2?" not in supervised
    assert "And 3+3?" not in supervised


def test_chain_sft_sharegpt_toolcall_formatter_to_dataset():
    """关键链路: ShareGPT 的 function_call/observation 经 formatter 转成
    assistant.tool_calls + role:tool 后, SFTDataset 应正确处理掩码"""
    from bumblecore.data_processing import DataFormatter

    raw = [
        {
            "conversations": [
                {"from": "human", "value": "查一下北京天气。"},
                {
                    "from": "function_call",
                    "value": '{"name": "get_weather", "arguments": {"city": "北京"}}',
                },
                {"from": "observation", "value": '{"temperature": 18, "condition": "sunny"}'},
                {"from": "gpt", "value": "北京 18°C，天气晴。"},
            ],
            "tools": '[{"type": "function", "function": {"name": "get_weather"}}]',
        }
    ]
    formatted = DataFormatter("sft")(raw)
    msgs = formatted[0]["messages"]
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in msgs)
    assert any(m.get("role") == "tool" for m in msgs)
    assert isinstance(formatted[0]["tools"], list)

    dataset = SFTDataset(formatted, tokenizer, max_length=512)
    result = dataset[0]

    decoded_full = tokenizer.decode(result["input_ids"])
    assert "<tool_call>" in decoded_full and "<tool_response>" in decoded_full
    assert "get_weather" in decoded_full

    supervised = _supervised_text(result)
    assert "get_weather" in supervised, "tool_call 应进入 labels"
    assert "北京 18°C，天气晴。" in supervised, "最终回答应进入 labels"
    assert "查一下北京天气。" not in supervised
    assert "<tool_response>" not in supervised
    assert '"temperature": 18' not in supervised


def test_chain_dpo_alpaca_formatter_to_dataset():
    """DPO Alpaca: raw -> DataFormatter -> DPODataset 端到端"""
    from bumblecore.data_processing import DataFormatter

    raw = [
        {
            "instruction": "Write a poem about spring",
            "input": "",
            "chosen": "Spring blooms in vibrant hues.",
            "rejected": "Flowers.",
        }
    ]
    formatted = DataFormatter("dpo")(raw)
    assert "chosen_messages" in formatted[0] and "rejected_messages" in formatted[0]
    assert "messages" in formatted[0]["chosen_messages"]
    assert "tools" in formatted[0]["chosen_messages"]

    dataset = DPODataset(formatted, tokenizer, max_length=256)
    result = dataset[0]

    chosen = _supervised_text(result, "chosen_labels")
    rejected = _supervised_text(result, "rejected_labels")
    assert "Spring blooms in vibrant hues." in chosen
    assert "Flowers." in rejected
    assert "Write a poem about spring" not in chosen
    assert "Write a poem about spring" not in rejected


def test_chain_dpo_sharegpt_formatter_to_dataset():
    """DPO ShareGPT: raw -> DataFormatter -> DPODataset 端到端"""
    from bumblecore.data_processing import DataFormatter

    raw = [
        {
            "conversations": [
                {"from": "system", "value": "You are helpful."},
                {"from": "human", "value": "Greet me."},
            ],
            "chosen": {"from": "gpt", "value": "Hello there, friend!"},
            "rejected": {"from": "gpt", "value": "hi"},
        }
    ]
    formatted = DataFormatter("dpo")(raw)
    assert formatted[0]["chosen_messages"]["messages"][-1]["content"] == "Hello there, friend!"
    assert formatted[0]["rejected_messages"]["messages"][-1]["content"] == "hi"

    dataset = DPODataset(formatted, tokenizer, max_length=256)
    result = dataset[0]

    chosen = _supervised_text(result, "chosen_labels")
    rejected = _supervised_text(result, "rejected_labels")
    assert "Hello there, friend!" in chosen
    assert "hi" in rejected
    assert "Greet me." not in chosen
    assert "Greet me." not in rejected


def test_chain_dpo_messages_tool_trajectory_formatter_to_dataset():
    """DPO messages 双工具轨迹: 两边只监督各自的 assistant 段"""
    from bumblecore.data_processing import DataFormatter

    raw = [
        {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "查询城市天气",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_air_quality",
                        "description": "查询城市空气质量",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                },
            ],
            "messages": [
                {"role": "user", "content": "今天北京的天气怎么样？"},
            ],
            "chosen": [
                {
                    "role": "assistant",
                    "content": "我来查一下北京当前天气。",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "北京"},
                            },
                        }
                    ],
                },
                {"role": "tool", "content": '{"temperature": 18, "condition": "晴"}'},
                {"role": "assistant", "content": "北京今天晴，18°C，适合外出。"},
            ],
            "rejected": [
                {
                    "role": "assistant",
                    "content": "我来查询一下相关信息。",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_air_quality",
                                "arguments": {"city": "北京"},
                            },
                        }
                    ],
                },
                {"role": "tool", "content": '{"aqi": 42, "level": "优"}'},
                {"role": "assistant", "content": "北京今天空气质量优，适合外出。"},
            ],
        }
    ]

    formatted = DataFormatter("dpo")(raw)
    assert formatted[0]["chosen_messages"]["completion_start_idx"] == 2

    dataset = DPODataset(formatted, tokenizer, max_length=512)
    result = dataset[0]

    chosen = _supervised_text(result, "chosen_labels")
    rejected = _supervised_text(result, "rejected_labels")
    assert "我来查一下北京当前天气。" in chosen
    assert "<tool_call>" in chosen
    assert "get_weather" in chosen
    assert "北京今天晴，18°C，适合外出。" in chosen
    assert "<tool_response>" not in chosen
    assert "temperature" not in chosen
    assert "今天北京的天气怎么样？" not in chosen
    assert "我来查询一下相关信息。" in rejected
    assert "<tool_call>" in rejected
    assert "get_air_quality" in rejected
    assert "北京今天空气质量优，适合外出。" in rejected
    assert "<tool_response>" not in rejected
    assert '"aqi": 42' not in rejected


def test_chain_dpo_messages_plain_multi_turn_trajectory():
    """普通多轮 DPO: assistant 段进入 labels，中间 user 段只作为上下文"""
    from bumblecore.data_processing import DataFormatter

    raw = [
        {
            "messages": [
                {"role": "user", "content": "帮我推荐一台电脑。"},
            ],
            "chosen": [
                {"role": "assistant", "content": "你的预算和主要用途是什么？"},
                {"role": "user", "content": "八千元，主要编程。"},
                {"role": "assistant", "content": "建议选择 32GB 内存、1TB SSD 的机型。"},
            ],
            "rejected": [
                {"role": "assistant", "content": "直接买最贵的游戏本。"},
            ],
        }
    ]

    formatted = DataFormatter("dpo")(raw)
    assert formatted[0]["chosen_messages"]["completion_start_idx"] == 2
    assert formatted[0]["rejected_messages"]["completion_start_idx"] == 2

    dataset = DPODataset(formatted, tokenizer, max_length=512)
    result = dataset[0]

    chosen = _supervised_text(result, "chosen_labels")
    rejected = _supervised_text(result, "rejected_labels")
    assert "你的预算和主要用途是什么？" in chosen
    assert "建议选择 32GB 内存、1TB SSD 的机型。" in chosen
    assert "八千元，主要编程。" not in chosen
    assert "帮我推荐一台电脑。" not in chosen
    assert "直接买最贵的游戏本。" in rejected
