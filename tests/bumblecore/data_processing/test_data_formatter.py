import pytest

from bumblecore.data_processing import DataFormatter


DEFAULT_SYSTEM = "You are Bumblebee, a helpful AI assistant."


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
                {"role": "system", "content": DEFAULT_SYSTEM},
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
                {"role": "system", "content": DEFAULT_SYSTEM},
                {"role": "user", "content": "Say hello"},
                {"role": "assistant", "content": "Hi"},
            ],
            "tools": None,
        }
    ]
    assert result == expected


def test_sft_alpaca_with_system_ignores_tools():
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
            "tools": None,
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
                {"role": "system", "content": DEFAULT_SYSTEM},
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
                    {"role": "system", "content": DEFAULT_SYSTEM},
                    {"role": "user", "content": "Write a poem"},
                    {"role": "assistant", "content": "Roses are red..."},
                ],
                "tools": None,
            },
            "rejected_messages": {
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM},
                    {"role": "user", "content": "Write a poem"},
                    {"role": "assistant", "content": "I don't know."},
                ],
                "tools": None,
            },
        }
    ]
    assert result == expected



def test_dpo_alpaca_with_system_ignores_tools():
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
                "tools": None,
            },
            "rejected_messages": {
                "messages": [
                    {"role": "system", "content": "You are a lyrical poet."},
                    {"role": "user", "content": "Write a poem about the ocean"},
                    {"role": "assistant", "content": "It's big and blue."},
                ],
                "tools": None,
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
                    {"role": "system", "content": DEFAULT_SYSTEM},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                    {"role": "user", "content": "How are you?"},
                    {"role": "assistant", "content": "I'm great!"},
                ],
                "tools": None,
            },
            "rejected_messages": {
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM},
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
# SFT - ShareGPT 工具调用 / 脏样本测试
# ==============================

def test_sft_sharegpt_toolcall_roundtrip():
    """function_call -> assistant.tool_calls; observation -> role=tool"""
    formatter = DataFormatter("sft")
    data = [
        {
            "conversations": [
                {"from": "human", "value": "给John开张发票"},
                {
                    "from": "function_call",
                    "value": '{"name": "gen_invoice", "arguments": {"name": "John"}}',
                },
                {
                    "from": "observation",
                    "value": '{"invoice_id": "INV1"}',
                },
                {"from": "gpt", "value": "已生成 INV1"},
            ],
            "tools": '[{"name": "gen_invoice"}]',
        }
    ]
    msgs = formatter(data)[0]["messages"]
    assert msgs[0] == {"role": "system", "content": DEFAULT_SYSTEM}
    assert msgs[1] == {"role": "user", "content": "给John开张发票"}
    assert msgs[2] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "gen_invoice", "arguments": {"name": "John"}},
            }
        ],
    }
    assert msgs[3] == {"role": "tool", "content": '{"invoice_id": "INV1"}'}
    assert msgs[4] == {"role": "assistant", "content": "已生成 INV1"}


def test_sft_sharegpt_toolcall_arguments_string():
    """function_call.arguments 即使被序列化成字符串也要还原成 dict"""
    formatter = DataFormatter("sft")
    data = [
        {
            "conversations": [
                {"from": "human", "value": "x"},
                {
                    "from": "function_call",
                    "value": '{"name": "f", "arguments": "{\\"k\\": 1}"}',
                },
                {"from": "observation", "value": "ok"},
                {"from": "gpt", "value": "done"},
            ],
        }
    ]
    msgs = formatter(data)[0]["messages"]
    assert msgs[2]["tool_calls"][0]["function"]["arguments"] == {"k": 1}


def test_sft_sharegpt_orphan_observation_skipped():
    """孤儿 observation -> warn + skip 整条"""
    formatter = DataFormatter("sft")
    data = [
        {
            "conversations": [
                {"from": "human", "value": "a"},
                {"from": "observation", "value": "stray"},
                {"from": "gpt", "value": "b"},
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "ok"},
                {"from": "gpt", "value": "fine"},
            ]
        },
    ]
    with pytest.warns(UserWarning, match="observation"):
        result = formatter(data)
    assert len(result) == 1
    assert result[0]["messages"][1]["content"] == "ok"


def test_sft_sharegpt_unknown_role_skipped():
    """未知 from 角色 -> warn + skip 整条"""
    formatter = DataFormatter("sft")
    data = [
        {
            "conversations": [
                {"from": "human", "value": "a"},
                {"from": "alien", "value": "??"},
                {"from": "gpt", "value": "b"},
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "ok"},
                {"from": "gpt", "value": "fine"},
            ]
        },
    ]
    with pytest.warns(UserWarning, match="Unsupported conversation role"):
        result = formatter(data)
    assert len(result) == 1


def test_ensure_system_message_does_not_mutate():
    """重复跑同一份数据集不应该污染原始 conversations"""
    formatter = DataFormatter("sft")
    original = [
        {
            "conversations": [
                {"from": "human", "value": "a"},
                {"from": "gpt", "value": "b"},
            ]
        }
    ]
    snapshot = [t.copy() for t in original[0]["conversations"]]
    formatter(original)
    formatter(original)
    assert original[0]["conversations"] == snapshot


# ==============================
# SFT / DPO - Messages (OpenAI) 格式测试
# ==============================

def test_sft_messages_passthrough_with_default_system():
    formatter = DataFormatter("sft")
    data = [
        {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        }
    ]
    result = formatter(data)
    assert result[0]["messages"] == [
        {"role": "system", "content": DEFAULT_SYSTEM},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    assert result[0]["tools"] is None


def test_sft_messages_with_tool_calls():
    """messages 输入直接带 tool_calls 应原样透出"""
    formatter = DataFormatter("sft")
    data = [
        {
            "messages": [
                {"role": "system", "content": "Custom"},
                {"role": "user", "content": "invoice for John"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "gen_invoice",
                                "arguments": {"name": "John"},
                            },
                        }
                    ],
                },
                {"role": "tool", "content": '{"id": 1}'},
                {"role": "assistant", "content": "done"},
            ],
            "tools": [{"name": "gen_invoice"}],
        }
    ]
    result = formatter(data)
    assert result[0]["messages"][0] == {"role": "system", "content": "Custom"}
    assert result[0]["messages"][2]["tool_calls"][0]["function"]["name"] == "gen_invoice"
    assert result[0]["messages"][3] == {"role": "tool", "content": '{"id": 1}'}
    assert result[0]["tools"] == [{"name": "gen_invoice"}]


def test_sft_messages_assistant_text_with_tool_calls():
    """assistant 同时带文字 content 和 tool_calls 应原样保留（OpenAI / Qwen 标准）"""
    formatter = DataFormatter("sft")
    data = [
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
            ]
        }
    ]
    result = formatter(data)
    asst = result[0]["messages"][2]
    assert asst["role"] == "assistant"
    assert asst["content"] == "好的，我帮您查一下北京当前天气。"
    assert asst["tool_calls"][0]["function"]["name"] == "get_weather"


def test_sft_messages_invalid_role_skipped():
    formatter = DataFormatter("sft")
    data = [
        {
            "messages": [
                {"role": "user", "content": "x"},
                {"role": "alien", "content": "y"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "ok"},
                {"role": "assistant", "content": "fine"},
            ]
        },
    ]
    with pytest.warns(UserWarning, match="Unsupported message role"):
        result = formatter(data)
    assert len(result) == 1


def test_dpo_messages_basic():
    formatter = DataFormatter("dpo")
    data = [
        {
            "messages": [
                {"role": "user", "content": "Write a poem"},
            ],
            "chosen": "Roses are red...",
            "rejected": "I don't know.",
        }
    ]
    result = formatter(data)
    assert result == [
        {
            "chosen_messages": {
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM},
                    {"role": "user", "content": "Write a poem"},
                    {"role": "assistant", "content": "Roses are red..."},
                ],
                "tools": None,
            },
            "rejected_messages": {
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM},
                    {"role": "user", "content": "Write a poem"},
                    {"role": "assistant", "content": "I don't know."},
                ],
                "tools": None,
            },
        }
    ]


def test_dpo_messages_chosen_dict_form():
    """DPO messages 也接受 {role, content} 形式的 chosen/rejected"""
    formatter = DataFormatter("dpo")
    data = [
        {
            "messages": [
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "hi"},
            ],
            "chosen": {"role": "assistant", "content": "hi back"},
            "rejected": {"role": "assistant", "content": "..."},
        }
    ]
    result = formatter(data)
    chosen = result[0]["chosen_messages"]["messages"]
    rejected = result[0]["rejected_messages"]["messages"]
    assert chosen[0] == {"role": "system", "content": "be concise"}
    assert chosen[-1] == {"role": "assistant", "content": "hi back"}
    assert rejected[-1] == {"role": "assistant", "content": "..."}


def test_dpo_messages_chosen_with_tool_calls():
    """DPO messages 支持偏好回复为 content + tool_calls，非偏好回复为纯文本"""
    formatter = DataFormatter("dpo")
    data = [
        {
            "messages": [
                {"role": "user", "content": "今天北京天气怎么样？"},
            ],
            "chosen": {
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
            "rejected": {"role": "assistant", "content": "北京今天应该挺好。"},
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        }
    ]

    result = formatter(data)
    chosen = result[0]["chosen_messages"]["messages"]
    rejected = result[0]["rejected_messages"]["messages"]
    assert chosen[-1]["content"] == "我来查一下北京当前天气。"
    assert chosen[-1]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert "tool_calls" not in rejected[-1]
    assert rejected[-1] == {"role": "assistant", "content": "北京今天应该挺好。"}
    assert result[0]["chosen_messages"]["tools"] == [
        {"type": "function", "function": {"name": "get_weather"}}
    ]


def test_dpo_messages_chosen_list_tool_trajectory():
    """DPO messages 支持 chosen/rejected 调用不同工具的完整消息轨迹"""
    formatter = DataFormatter("dpo")
    data = [
        {
            "messages": [
                {"role": "user", "content": "今天北京天气怎么样？"},
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
                {"role": "assistant", "content": "北京今天空气质量优。"},
            ],
            "tools": [
                {"type": "function", "function": {"name": "get_weather"}},
                {"type": "function", "function": {"name": "get_air_quality"}},
            ],
        }
    ]

    result = formatter(data)
    chosen = result[0]["chosen_messages"]
    rejected = result[0]["rejected_messages"]
    assert chosen["completion_start_idx"] == 2
    assert rejected["completion_start_idx"] == 2
    assert chosen["messages"][2]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert chosen["messages"][3] == {
        "role": "tool",
        "content": '{"temperature": 18, "condition": "晴"}',
    }
    assert chosen["messages"][-1] == {
        "role": "assistant",
        "content": "北京今天晴，18°C，适合外出。",
    }
    assert rejected["messages"][2]["tool_calls"][0]["function"]["name"] == "get_air_quality"
    assert rejected["messages"][3] == {
        "role": "tool",
        "content": '{"aqi": 42, "level": "优"}',
    }
    assert rejected["messages"][-1] == {
        "role": "assistant",
        "content": "北京今天空气质量优。",
    }


# ==============================
# 其他通用测试
# ==============================

def test_unsupported_stage():
    formatter = DataFormatter("rlhf")
    with pytest.raises(ValueError, match="Unsupported training stage: rlhf"):
        formatter([{"dummy": "data"}])
