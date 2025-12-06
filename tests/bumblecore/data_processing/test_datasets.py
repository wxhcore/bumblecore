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
    # 输入：使用 test_data_formatter.py 的输出格式
    train_dataset = [
        {"text": "Hello"},
        {"text": "World"},
    ]
    
    max_length = 128
    
    dataset = PretrainDataset(train_dataset, tokenizer, max_length)
    result = dataset[0]
    
    # 期望输出：根据真实 tokenizer 的实际输出
    # "Hello" -> [9707, 151645] (Hello + eos token)
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
    
    # 期望输出：根据真实 tokenizer 的实际输出
    # "Test" -> tokenized + eos token
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
    
    # 临时修改 tokenizer 属性
    original_eos_token_id = tokenizer.eos_token_id
    original_eos_token = tokenizer.eos_token
    tokenizer.eos_token_id = None
    tokenizer.eos_token = None
    tokenizer.bos_token_id = 1
    tokenizer.bos_token = "<|bos|>"
    
    try:
        dataset = PretrainDataset(train_dataset, tokenizer, max_length=128)
        result = dataset[0]
        
        # 期望输出：文本 tokenized（没有 eos，因为 eos_token_id 是 None）
        test_tokens = tokenizer("Test", return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
        expected_input_ids = test_tokens
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
        # 恢复原始属性
        tokenizer.eos_token_id = original_eos_token_id
        tokenizer.eos_token = original_eos_token


# ==============================
# SFTDataset 测试
# ==============================

def test_sft_dataset():
    """测试 SFTDataset 的端到端功能"""
    # 输入：使用 test_data_formatter.py 的输出格式
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
    
    # 期望输出：根据真实 tokenizer 的实际输出
    expected_input_ids = torch.tensor([
        151644, 8948, 198, 2610, 525, 10950, 13, 151645, 198, 151644, 872, 198, 
        13048, 151645, 198, 151644, 77091, 198, 9707, 151645, 198
    ], dtype=torch.long)
    expected_attention_mask = torch.ones_like(expected_input_ids)
    
    # labels: 只有最后 3 个位置（assistant 部分）不是 -100
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
    
    # 期望输出：根据真实 tokenizer 的实际输出
    # input_ids 长度是 111，labels 中最后 3 个位置不是 -100: [19, 151645, 198]
    assert len(result["input_ids"]) == 111
    assert len(result["labels"]) == 111
    
    # 验证 labels 中只有最后 3 个位置不是 -100
    non_negative_100 = (result["labels"] != -100).nonzero(as_tuple=True)[0]
    assert len(non_negative_100) == 3
    assert torch.equal(result["labels"][non_negative_100], torch.tensor([19, 151645, 198], dtype=torch.long))
    
    # 验证 input_ids 和 labels 在非 -100 位置的值相同
    assert torch.equal(result["input_ids"][non_negative_100], result["labels"][non_negative_100])


# ==============================
# DPODataset 测试
# ==============================

def test_dpo_dataset():
    """测试 DPODataset 的端到端功能"""
    # 输入：使用 test_data_formatter.py 的输出格式
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
    
    # 期望输出：根据真实 tokenizer 的实际输出
    # chosen 和 rejected 都是 26 个 tokens
    assert len(result["chosen_input_ids"]) == 26
    assert len(result["rejected_input_ids"]) == 26
    
    # 验证 chosen 字段的格式（扁平化格式）
    assert "chosen_input_ids" in result
    assert "chosen_attention_mask" in result
    assert "chosen_labels" in result
    assert isinstance(result["chosen_input_ids"], torch.Tensor)
    assert isinstance(result["chosen_attention_mask"], torch.Tensor)
    assert isinstance(result["chosen_labels"], torch.Tensor)
    
    # 验证 rejected 字段的格式（扁平化格式）
    assert "rejected_input_ids" in result
    assert "rejected_attention_mask" in result
    assert "rejected_labels" in result
    assert isinstance(result["rejected_input_ids"], torch.Tensor)
    assert isinstance(result["rejected_attention_mask"], torch.Tensor)
    assert isinstance(result["rejected_labels"], torch.Tensor)
    
    # 验证 chosen 和 rejected 的 input_ids 不同（因为 assistant 回复不同）
    assert not torch.equal(result["chosen_input_ids"], result["rejected_input_ids"])





# ==============================
# DataCollator 测试
# ==============================

def test_data_collator():
    """测试 DataCollator 的端到端功能"""
    collator = DataCollator(tokenizer)
    
    # 输入：一批不同长度的样本
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
    
    # 期望输出：所有序列填充到相同长度（最长的是 7）
    # 使用真实 tokenizer 的 pad_token_id
    pad_token_id = tokenizer.pad_token_id
    expected = {
        "input_ids": torch.tensor([
            [1, 10, 20, 30, 2, pad_token_id, pad_token_id],  # 填充 pad_token_id
            [1, 15, 25, 2, pad_token_id, pad_token_id, pad_token_id],   # 填充 pad_token_id
            [1, 12, 22, 32, 42, 52, 2], # 最长，不填充
        ], dtype=torch.long),
        "attention_mask": torch.tensor([
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
        ], dtype=torch.long),
        "labels": torch.tensor([
            [1, 10, 20, 30, 2, -100, -100],  # 填充 -100
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
    
    # 期望输出：填充到最长序列的长度（5），填充的 labels 应该是 -100
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
    
    # 输入：DPO 样本（扁平化格式，长度参差不齐）
    # 设计不同长度：第一个样本 chosen=3, rejected=6；第二个样本 chosen=5, rejected=4
    # 最大长度是 6，所有字段都需要填充到 6
    batch = [
        {
            "chosen_input_ids": torch.tensor([1, 10, 2]),  # 长度 3
            "chosen_attention_mask": torch.tensor([1, 1, 1]),
            "chosen_labels": torch.tensor([-100, 10, 2]),
            "rejected_input_ids": torch.tensor([1, 10, 20, 30, 40, 2]),  # 长度 6（最大）
            "rejected_attention_mask": torch.tensor([1, 1, 1, 1, 1, 1]),
            "rejected_labels": torch.tensor([-100, -100, 20, 30, 40, 2]),
        },
        {
            "chosen_input_ids": torch.tensor([1, 15, 25, 35, 2]),  # 长度 5
            "chosen_attention_mask": torch.tensor([1, 1, 1, 1, 1]),
            "chosen_labels": torch.tensor([-100, -100, 25, 35, 2]),
            "rejected_input_ids": torch.tensor([1, 15, 45, 2]),  # 长度 4
            "rejected_attention_mask": torch.tensor([1, 1, 1, 1]),
            "rejected_labels": torch.tensor([-100, -100, 45, 2]),
        },
    ]
    
    result = collator(batch)
    
    # 期望输出：所有字段都填充到最大长度 6
    # 注意：DPOCollator 会找出 chosen 和 rejected 中的所有序列的最大长度，然后统一填充
    pad_token_id = tokenizer.pad_token_id
    expected = {
        "chosen_input_ids": torch.tensor([
            [1, 10, 2, pad_token_id, pad_token_id, pad_token_id],  # 长度 3 -> 6
            [1, 15, 25, 35, 2, pad_token_id],  # 长度 5 -> 6
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
            [1, 10, 20, 30, 40, 2],  # 长度 6，无需填充
            [1, 15, 45, 2, pad_token_id, pad_token_id],  # 长度 4 -> 6
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



