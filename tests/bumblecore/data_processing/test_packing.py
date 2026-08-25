import torch
import pytest
from transformers import AutoTokenizer
from bumblecore.data_processing import (
    SFTDataset,
    DataCollator,
    PackingDataCollator,
)

tokenizer = AutoTokenizer.from_pretrained("./models/bumblebee")


# ==============================
# PackingDataCollator 测试
# ==============================

def test_packing_data_collator_basic():
    """测试 PackingDataCollator 的基本功能"""
    collator = PackingDataCollator(tokenizer)
    
    # 创建包含 position_ids 的批次数据
    batch = [
        {
            "input_ids": torch.tensor([1, 10, 20, 30, 2]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
            "labels": torch.tensor([1, 10, 20, 30, 2]),
            "position_ids": torch.tensor([0, 1, 2, 3, 4]),
        },
        {
            "input_ids": torch.tensor([1, 15, 25, 2]),
            "attention_mask": torch.tensor([1, 1, 1, 1]),
            "labels": torch.tensor([1, 15, 25, 2]),
            "position_ids": torch.tensor([0, 1, 2, 3]),
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
            [1, 10, 20, 30, 2],
            [1, 15, 25, 2, -100],
        ], dtype=torch.long),
        "position_ids": torch.tensor([
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 0],
        ], dtype=torch.long),
    }
    
    assert torch.equal(result["input_ids"], expected["input_ids"]), f"input_ids 不匹配, 结果: {result['input_ids']}, 期望: {expected['input_ids']}"
    assert torch.equal(result["attention_mask"], expected["attention_mask"]), "attention_mask 不匹配"
    assert torch.equal(result["labels"], expected["labels"]), "labels 不匹配"
    assert torch.equal(result["position_ids"], expected["position_ids"]), "position_ids 不匹配"


def test_packing_data_collator_with_varying_lengths():
    """测试 PackingDataCollator 处理不同长度序列"""
    collator = PackingDataCollator(tokenizer)
    
    batch = [
        {
            "input_ids": torch.tensor([1, 10, 2]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([-100, 10, 2]),
            "position_ids": torch.tensor([0, 1, 2]),
        },
        {
            "input_ids": torch.tensor([1, 15, 25, 35, 45, 2]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1]),
            "labels": torch.tensor([-100, -100, 25, 35, 45, 2]),
            "position_ids": torch.tensor([0, 1, 2, 3, 4, 5]),
        },
        {
            "input_ids": torch.tensor([1, 12, 2]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([-100, 12, 2]),
            "position_ids": torch.tensor([0, 1, 2]),
        },
    ]
    
    result = collator(batch)
    
    pad_token_id = tokenizer.pad_token_id
    expected = {
        "input_ids": torch.tensor([
            [1, 10, 2, pad_token_id, pad_token_id, pad_token_id],
            [1, 15, 25, 35, 45, 2],
            [1, 12, 2, pad_token_id, pad_token_id, pad_token_id],
        ], dtype=torch.long),
        "attention_mask": torch.tensor([
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
        ], dtype=torch.long),
        "labels": torch.tensor([
            [-100, 10, 2, -100, -100, -100],
            [-100, -100, 25, 35, 45, 2],
            [-100, 12, 2, -100, -100, -100],
        ], dtype=torch.long),
        "position_ids": torch.tensor([
            [0, 1, 2, 0, 0, 0],
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 0, 0, 0],
        ], dtype=torch.long),
    }
    
    assert torch.equal(result["input_ids"], expected["input_ids"])
    assert torch.equal(result["attention_mask"], expected["attention_mask"])
    assert torch.equal(result["labels"], expected["labels"])
    assert torch.equal(result["position_ids"], expected["position_ids"])


def test_packing_data_collator_empty_batch():
    """测试 PackingDataCollator 处理空批次"""
    collator = PackingDataCollator(tokenizer)
    
    batch = []
    
    result = collator(batch)
    
    assert result["input_ids"].shape == (0, 0)
    assert result["attention_mask"].shape == (0, 0)
    assert result["labels"].shape == (0, 0)
    assert result["position_ids"].shape == (0, 0)


# ==============================
# SFTDataset Packing 功能测试
# ==============================

def test_sft_dataset_packing_disabled():
    """测试 SFTDataset 禁用 packing 时的行为"""
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
    
    # 禁用 packing
    dataset = SFTDataset(train_dataset, tokenizer, max_length, packing=False)
    
    # 检查长度
    assert len(dataset) == len(train_dataset)
    
    # 获取样本
    result = dataset[0]
    
    # 检查返回的字段
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result
    assert "position_ids" not in result  # 禁用 packing 时不返回 position_ids
    
    # 检查数据类型
    assert isinstance(result["input_ids"], torch.Tensor)
    assert isinstance(result["attention_mask"], torch.Tensor)
    assert isinstance(result["labels"], torch.Tensor)


def test_sft_dataset_packing_enabled():
    """测试 SFTDataset 启用 packing 时的基本行为"""
    train_dataset = [
        {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
            "tools": None,
        },
        {
            "messages": [
                {"role": "system", "content": "You are a math tutor."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ],
            "tools": None,
        }
    ]
    
    max_length = 256
    
    # 启用 packing
    dataset = SFTDataset(train_dataset, tokenizer, max_length, packing=True)
    
    # 检查长度（packing 会改变数据集长度）
    assert len(dataset) >= 1  # 至少有一个pack
    
    # 获取样本
    result = dataset[0]
    
    # 检查返回的字段
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result
    assert "position_ids" in result  # 启用 packing 时返回 position_ids
    
    # 检查数据类型
    assert isinstance(result["input_ids"], torch.Tensor)
    assert isinstance(result["attention_mask"], torch.Tensor)
    assert isinstance(result["labels"], torch.Tensor)
    assert isinstance(result["position_ids"], torch.Tensor)
    
    # 检查序列长度一致性
    input_ids_len = len(result["input_ids"])
    assert len(result["attention_mask"]) == input_ids_len
    assert len(result["labels"]) == input_ids_len
    assert len(result["position_ids"]) == input_ids_len


def test_sft_dataset_packing_with_single_sample():
    """测试 SFTDataset packing 处理单个样本"""
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
    
    # 启用 packing
    dataset = SFTDataset(train_dataset, tokenizer, max_length, packing=True)
    
    # 检查长度
    assert len(dataset) == 1  # 单个样本应该只有一个pack
    
    # 获取样本
    result = dataset[0]
    
    # 检查返回的字段
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result
    assert "position_ids" in result
    
    # 检查 position_ids 是否正确重置
    position_ids = result["position_ids"]
    assert torch.equal(position_ids, torch.arange(len(position_ids), dtype=torch.long))


def test_sft_dataset_packing_position_ids_reset():
    """测试 SFTDataset packing 中 position_ids 的重置逻辑"""
    train_dataset = [
        {
            "messages": [
                {"role": "system", "content": "Short message."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
            "tools": None,
        },
        {
            "messages": [
                {"role": "system", "content": "Another short message."},
                {"role": "user", "content": "Hey"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "tools": None,
        }
    ]
    
    max_length = 100  # 设置较小的max_length以触发packing
    
    # 启用 packing
    dataset = SFTDataset(train_dataset, tokenizer, max_length, packing=True)
    
    # 获取样本
    result = dataset[0]
    
    # 检查 position_ids 是否正确重置
    position_ids = result["position_ids"]
    
    # 获取 packed_idx 中的实际顺序
    sequence_indices = dataset.packed_idx[0]
    
    # 根据实际顺序验证 position_ids
    offset = 0
    for seq_idx in sequence_indices:
        sample = dataset._process_single_sample(seq_idx)
        seq_len = len(sample["input_ids"])
        
        # 验证当前序列的 position_ids 正确重置为 [0, 1, ..., seq_len-1]
        expected_position_ids = torch.arange(seq_len, dtype=torch.long)
        actual_position_ids = position_ids[offset:offset + seq_len]
        assert torch.equal(actual_position_ids, expected_position_ids), \
            f"Sequence {seq_idx}: expected {expected_position_ids.tolist()}, got {actual_position_ids.tolist()}"
        
        offset += seq_len
    
    # 验证总长度正确
    assert offset == len(position_ids), f"Total length mismatch: {offset} vs {len(position_ids)}"


def test_sft_dataset_packing_with_tools():
    """测试 SFTDataset packing 带 tools 的情况"""
    train_dataset = [
        {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Calculate 2+2"},
                {"role": "assistant", "content": "4"},
            ],
            "tools": [{"name": "calculator"}],
        },
        {
            "messages": [
                {"role": "system", "content": "You are a math tutor."},
                {"role": "user", "content": "What is 3*3?"},
                {"role": "assistant", "content": "9"},
            ],
            "tools": [{"name": "calculator"}],
        }
    ]
    
    max_length = 256
    
    # 启用 packing
    dataset = SFTDataset(train_dataset, tokenizer, max_length, packing=True)
    
    # 检查长度
    assert len(dataset) >= 1
    
    # 获取样本
    result = dataset[0]
    
    # 检查返回的字段
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result
    assert "position_ids" in result
    
    # 检查数据类型
    assert isinstance(result["input_ids"], torch.Tensor)
    assert isinstance(result["attention_mask"], torch.Tensor)
    assert isinstance(result["labels"], torch.Tensor)
    assert isinstance(result["position_ids"], torch.Tensor)


def test_sft_dataset_packing_labels_consistency():
    """测试 SFTDataset packing 中 labels 的一致性"""
    train_dataset = [
        {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
            "tools": None,
        },
        {
            "messages": [
                {"role": "system", "content": "You are a math tutor."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ],
            "tools": None,
        }
    ]
    
    max_length = 256
    
    # 启用 packing
    dataset = SFTDataset(train_dataset, tokenizer, max_length, packing=True)
    
    # 获取样本
    result = dataset[0]
    
    input_ids = result["input_ids"]
    labels = result["labels"]
    attention_mask = result["attention_mask"]
    
    # 检查 labels 中非 -100 的位置与 input_ids 一致
    non_negative_100_mask = labels != -100
    assert torch.equal(input_ids[non_negative_100_mask], labels[non_negative_100_mask])
    
    # 检查 attention_mask 的有效性
    assert torch.all(attention_mask == 1)  # packing 时 attention_mask 应该全为1


def test_sft_dataset_packing_edge_cases():
    """测试 SFTDataset packing 的边缘情况"""
    
    # 测试空数据集
    empty_dataset = []
    with pytest.raises(ValueError, match="train_dataset cannot be empty"):
        dataset = SFTDataset(empty_dataset, tokenizer, max_length=256, packing=True)
    
    # 测试单个长样本
    long_message_dataset = [
        {
            "messages": [
                {"role": "system", "content": "A" * 1000},  # 很长的消息
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
            "tools": None,
        }
    ]
    
    max_length = 50  # 设置较小的max_length
    
    dataset = SFTDataset(long_message_dataset, tokenizer, max_length, packing=True)
    result = dataset[0]
    
    # 检查序列长度不超过max_length
    assert len(result["input_ids"]) <= max_length


# ==============================
# 集成测试：PackingDataCollator + SFTDataset
# ==============================

def test_packing_integration():
    """测试 PackingDataCollator 和 SFTDataset 的集成"""
    train_dataset = [
        {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
            "tools": None,
        },
        {
            "messages": [
                {"role": "system", "content": "You are a math tutor."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ],
            "tools": None,
        }
    ]
    
    max_length = 256
    
    # 创建带 packing 的 dataset
    dataset = SFTDataset(train_dataset, tokenizer, max_length, packing=True)
    
    # 创建 PackingDataCollator
    collator = PackingDataCollator(tokenizer)
    
    # 创建批次
    batch = [dataset[i] for i in range(min(2, len(dataset)))]
    
    # 使用 collator 处理批次
    result = collator(batch)
    
    # 检查结果
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result
    assert "position_ids" in result
    
    # 检查批次维度
    batch_size = len(batch)
    assert result["input_ids"].shape[0] == batch_size
    assert result["attention_mask"].shape[0] == batch_size
    assert result["labels"].shape[0] == batch_size
    assert result["position_ids"].shape[0] == batch_size


def test_packing_vs_standard_collator():
    """比较 PackingDataCollator 和标准 DataCollator 的区别"""
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
    
    # 创建带 packing 的 dataset
    dataset_packing = SFTDataset(train_dataset, tokenizer, max_length, packing=True)
    dataset_standard = SFTDataset(train_dataset, tokenizer, max_length, packing=False)
    
    # 创建两种 collator
    packing_collator = PackingDataCollator(tokenizer)
    standard_collator = DataCollator(tokenizer)
    
    # 获取样本
    sample_packing = dataset_packing[0]
    sample_standard = dataset_standard[0]
    
    # 检查 packing 样本包含 position_ids
    assert "position_ids" in sample_packing
    assert "position_ids" not in sample_standard
    
    # 使用 collator 处理
    batch_packing = packing_collator([sample_packing])
    batch_standard = standard_collator([sample_standard])
    
    # 检查 packing collator 返回 position_ids
    assert "position_ids" in batch_packing
    assert "position_ids" not in batch_standard


if __name__ == "__main__":
    # 运行测试
    test_packing_data_collator_basic()
    test_packing_data_collator_with_varying_lengths()
    test_packing_data_collator_empty_batch()
    test_sft_dataset_packing_disabled()
    test_sft_dataset_packing_enabled()
    test_sft_dataset_packing_with_single_sample()
    test_sft_dataset_packing_position_ids_reset()
    test_sft_dataset_packing_with_tools()
    test_sft_dataset_packing_labels_consistency()
    test_sft_dataset_packing_edge_cases()
    test_packing_integration()
    test_packing_vs_standard_collator()
    
    print("所有 packing 功能测试通过！")