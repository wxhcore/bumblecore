import pytest

from bumblecore.data_processing import DataFormatter


PRETRAIN_DATA = [
    {"text": "This is pretrain sample 1."},
    {"text": "This is pretrain sample 2."},
]


def test_pretrain_valid():
    formatter = DataFormatter("pretrain")
    result = formatter(PRETRAIN_DATA)
    assert result == PRETRAIN_DATA


def test_pretrain_invalid():
    formatter = DataFormatter("pretrain")
    invalid_data = [{"no_text": "missing"}]
    with pytest.raises(ValueError, match="Pretrain data format is invalid"):
        formatter(invalid_data)


# ==============================
# SFT - Alpaca 格式测试
# ==============================

def test_sft_alpaca():
    formatter = DataFormatter("sft")
    data = [
        {
            "instruction": "Translate English to French",
            "input": "Hello",
            "output": "Bonjour",
        }
    ]
    result = formatter(data)
    expected = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Translate English to French\nHello"},
                {"role": "assistant", "content": "Bonjour"},
            ],
            "tools": None,
        }
    ]
    assert result == expected


def test_sft_alpaca_no_input():
    """测试 Alpaca 格式中没有 'input' 字段的情况"""
    formatter = DataFormatter("sft")
    data = [
        {
            "instruction": "Say hello",
            "output": "Hi",
        }
    ]
    result = formatter(data)
    expected = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello"},
                {"role": "assistant", "content": "Hi"},
            ],
            "tools": None,
        }
    ]
    assert result == expected


def test_sft_alpaca_with_system_and_tools():
    formatter = DataFormatter("sft")
    data = [
        {
            "system": "You are a helpful math tutor.",
            "instruction": "Solve the equation",
            "input": "x + 2 = 5",
            "output": "x = 3",
            "tools": '[{"name": "calculator", "description": "Evaluates math expressions"}]'
        }
    ]
    result = formatter(data)
    expected = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful math tutor."},
                {"role": "user", "content": "Solve the equation\nx + 2 = 5"},
                {"role": "assistant", "content": "x = 3"},
            ],
            "tools": [{"name": "calculator", "description": "Evaluates math expressions"}],
        }
    ]
    assert result == expected


@pytest.mark.parametrize("tools_input, expected_tools", [
    (None, None),
    ([], None),
    ("[]", None),
    ("", None),
])
def test_sft_alpaca_tools_empty_cases(tools_input, expected_tools):
    formatter = DataFormatter("sft")
    
    base_item = {
        "instruction": "Hello",
        "output": "Hi"
    }
    if tools_input is not None:
        base_item["tools"] = tools_input
    
    data = [base_item]
    result = formatter(data)
    assert result[0]["tools"] == expected_tools


def test_sft_alpaca_tools_missing():
    """测试完全不提供 'tools' 字段"""
    formatter = DataFormatter("sft")
    data = [{"instruction": "Hi", "output": "Hello"}]
    result = formatter(data)
    assert result[0]["tools"] is None


# ==============================
# SFT - ShareGPT 格式测试
# ==============================

@pytest.mark.parametrize("tools_input, expected_tools", [
    (None, None),
    ([], None),
    ("[]", None),
    ("", None),
])
def test_sft_sharegpt_tools_empty_cases(tools_input, expected_tools):
    formatter = DataFormatter("sft")
    
    base_item = {
        "conversations": [
            {"from": "human", "value": "Hi"},
            {"from": "gpt", "value": "Hello!"},
        ]
    }
    if tools_input is not None:
        base_item["tools"] = tools_input
    
    data = [base_item]
    result = formatter(data)
    assert result[0]["tools"] == expected_tools


def test_sft_sharegpt_no_system():
    """ShareGPT 数据未提供 system，应使用默认"""
    formatter = DataFormatter("sft")
    data = [
        {
            "conversations": [
                {"from": "human", "value": "My app crashes"},
                {"from": "gpt", "value": "Sorry to hear that."},
            ]
        }
    ]
    result = formatter(data)
    expected = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "My app crashes"},
                {"role": "assistant", "content": "Sorry to hear that."},
            ],
            "tools": None,
        }
    ]
    assert result == expected


def test_sft_sharegpt_with_system_and_tools():
    formatter = DataFormatter("sft")
    data = [
        {
            "conversations": [
                {"from": "system","value": "You are a translator."},
                {"from": "human", "value": "Translate: Hello"},
                {"from": "gpt", "value": "Bonjour"},
            ],
            "tools": '[{"name": "translator"}]',
        }
    ]
    result = formatter(data)
    expected = [
        {
            "messages": [
                {"role": "system", "content": "You are a translator."},
                {"role": "user", "content": "Translate: Hello"},
                {"role": "assistant", "content": "Bonjour"},
            ],
            "tools": [{"name": "translator"}],
        }
    ]
    assert result == expected


# ==============================
# DPO - Alpaca 格式测试
# ==============================


@pytest.mark.parametrize("tools_input, expected_tools", [
    (None, None),
    ([], None),
    ("[]", None),
    ("", None),
])
def test_dpo_alpaca_tools_empty_cases(tools_input, expected_tools):
    formatter = DataFormatter("dpo")
    
    base_item = {
        "instruction": "Write a poem",
        "chosen": "Roses are red...",
        "rejected": "I don't know.",
    }
    if tools_input is not None:
        base_item["tools"] = tools_input
    
    data = [base_item]
    result = formatter(data)
    # DPO 返回的是 chosen_messages 和 rejected_messages，各自包含 tools
    assert result[0]["chosen_messages"]["tools"] == expected_tools
    assert result[0]["rejected_messages"]["tools"] == expected_tools

def test_dpo_alpaca():
    formatter = DataFormatter("dpo")
    data = [
        {
            "instruction": "Write a poem",
            "chosen": "Roses are red...",
            "rejected": "I don't know.",
        }
    ]
    result = formatter(data)
    expected = [
        {
            "chosen_messages": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Write a poem"},
                    {"role": "assistant", "content": "Roses are red..."},
                ],
                "tools": None,
            },
            "rejected_messages": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Write a poem"},
                    {"role": "assistant", "content": "I don't know."},
                ],
                "tools": None,
            },
        }
    ]
    assert result == expected



def test_dpo_alpaca_with_system_and_tools():
    tools = [{"name": "rhyme_suggester", "description": "Suggests rhyming words"}]
    data = [
        {
            "system": "You are a lyrical poet.",
            "instruction": "Write a poem about the ocean",
            "chosen": "The waves whisper secrets...",
            "rejected": "It's big and blue.",
            "tools": str(tools).replace("'", '"'),  # simulate JSON string
        }
    ]
    formatter = DataFormatter("dpo")
    result = formatter(data)
    expected = [
        {
            "chosen_messages": {
                "messages": [
                    {"role": "system", "content": "You are a lyrical poet."},
                    {"role": "user", "content": "Write a poem about the ocean"},
                    {"role": "assistant", "content": "The waves whisper secrets..."},
                ],
                "tools": tools,
            },
            "rejected_messages": {
                "messages": [
                    {"role": "system", "content": "You are a lyrical poet."},
                    {"role": "user", "content": "Write a poem about the ocean"},
                    {"role": "assistant", "content": "It's big and blue."},
                ],
                "tools": tools,
            },
        }
    ]
    assert result == expected


# ==============================
# DPO - ShareGPT 格式测试
# ==============================


@pytest.mark.parametrize("tools_input, expected_tools", [
    (None, None),
    ([], None),
    ("[]", None),
    ("", None),
])
def test_dpo_sharegpt_tools_empty_cases(tools_input, expected_tools):
    formatter = DataFormatter("dpo")
    
    base_item = {
        "conversations": [
            {"from": "human", "value": "Hi"},
            {"from": "gpt", "value": "Hello!"},
            {"from": "human", "value": "good day!"},
        ],
        "chosen": {"value": "Great!"},
        "rejected": {"value": "Bad."},
    }
    if tools_input is not None:
        base_item["tools"] = tools_input
    
    data = [base_item]
    result = formatter(data)
    assert result[0]["chosen_messages"]["tools"] == expected_tools
    assert result[0]["rejected_messages"]["tools"] == expected_tools

def test_dpo_sharegpt():
    formatter = DataFormatter("dpo")
    data = [
        {
            "conversations": [
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello!"},
                {"from": "human", "value": "How are you?"},
            ],
            "chosen": {"value": "I'm great!"},
            "rejected": {"value": "Not good."},
        }
    ]
    result = formatter(data)
    expected = [
        {
            "chosen_messages": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                    {"role": "user", "content": "How are you?"},
                    {"role": "assistant", "content": "I'm great!"},
                ],
                "tools": None,
            },
            "rejected_messages": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                    {"role": "user", "content": "How are you?"},
                    {"role": "assistant", "content": "Not good."},
                ],
                "tools": None,
            },
        }
    ]
    assert result == expected


def test_dpo_sharegpt_with_system_and_tools():
    formatter = DataFormatter("dpo")
    data = [
        {
            "conversations": [
                {"from": "system","value": "You are a health coach."},
                {"from": "human", "value": "I feel tired."},
            ],
            "chosen": {"value": "Try walking 10 minutes daily."},
            "rejected": {"value": "Maybe you’re lazy."},
            "tools": '[{"name": "wellness_tips", "description": "Evidence-based suggestions"}]'
        }
    ]
    result = formatter(data)
    expected = [
        {
            "chosen_messages": {
                "messages": [
                    {"role": "system", "content": "You are a health coach."},
                    {"role": "user", "content": "I feel tired."},
                    {"role": "assistant", "content": "Try walking 10 minutes daily."},
                ],
                "tools": [{"name": "wellness_tips", "description": "Evidence-based suggestions"}],
            },
            "rejected_messages": {
                "messages": [
                    {"role": "system", "content": "You are a health coach."},
                    {"role": "user", "content": "I feel tired."},
                    {"role": "assistant", "content": "Maybe you’re lazy."},
                ],
                "tools": [{"name": "wellness_tips", "description": "Evidence-based suggestions"}],
            },
        }
    ]
    assert result == expected


# ==============================
# 其他通用测试
# ==============================

def test_unsupported_stage():
    formatter = DataFormatter("rlhf")
    with pytest.raises(ValueError, match="Unsupported training stage: rlhf"):
        formatter([{"dummy": "data"}])