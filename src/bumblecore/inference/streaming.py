import argparse
import os

import yaml

from .inference import BumblebeeChat, HFStreamChat

def start_chat_session(
    model_path,
    device_map,
    dtype,
    max_new_tokens,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    system_prompt,
    do_sample,
    enable_history,
    training_stage
):
    print(f"正在加载模型: {model_path}，请稍候...")
    bot = BumblebeeChat(
        model_path=model_path,
        device_map=device_map,
        dtype=dtype
    )
    print("模型加载完成！输入 'quit' 或 'exit' 退出聊天。\n")

    messages = []

    while True:
        user_input = input("👤 用户: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("👋 再见！")
            break

        if training_stage == "pretrain":
            print("🤖 助手: ", end="", flush=True)
            response_chunks = []
            for text in bot.stream_chat(
                messages=user_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample
            ):
                print(text, end="", flush=True)
                response_chunks.append(text)
            print()
            continue

        if enable_history:
            messages.append({"role": "user", "content": user_input})
            current_messages = messages
        else:
            current_messages = [{"role": "user", "content": user_input}]

        print("🤖 助手: ", end="", flush=True)

        response_chunks = []
        for text in bot.stream_chat(
            messages=current_messages,
            max_new_tokens=max_new_tokens,
            system_prompt=system_prompt,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample
        ):
            print(text, end="", flush=True)
            response_chunks.append(text)
        print()

        full_response = "".join(response_chunks)

        if enable_history:
            messages.append({"role": "assistant", "content": full_response})

        print()


def bumblebee_streaming_chat():
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--yaml_config",
        type=str,
        default="",
        help="YAML 配置文件路径",
    )
    cfg_args, _ = base_parser.parse_known_args()
    config_path = cfg_args.yaml_config
    
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    
    parser = argparse.ArgumentParser(
        description="启动 Bumblebee 聊天会话",
        parents=[base_parser],
    )
    parser.add_argument("--model_path", type=str, 
                        default=cfg.get("model_path"),
                        help="模型路径")
                        
    parser.add_argument("--device_map", type=str, 
                        default=cfg.get("device_map", "auto"),
                        help="设备映射（如 'auto', 'cpu', 'cuda:0' 等）")

    parser.add_argument("--dtype", type=str, 
                        default=cfg.get("dtype", "auto"),
                        help="模型数据类型（如 'torch.float16', 'torch.bfloat16', 'auto'）")

    parser.add_argument("--max_new_tokens", type=int, 
                        default=cfg.get("max_new_tokens", None),
                        help="最大生成 token 数")

    parser.add_argument("--system_prompt", type=str, 
                        default=cfg.get("system_prompt", None),
                        help="设置 system prompt（不传使用默认）")

    parser.add_argument("--temperature", type=float, 
                        default=cfg.get("temperature", None),
                        help="采样温度（不传使用模型默认）")

    parser.add_argument("--top_k", type=int, 
                        default=cfg.get("top_k", None),
                        help="Top_k 采样（不传使用模型默认）")

    parser.add_argument("--top_p", type=float, 
                        default=cfg.get("top_p", None),
                        help="Top_p (nucleus) 采样（不传使用模型默认）")

    parser.add_argument("--repetition_penalty", type=float, 
                        default=cfg.get("repetition_penalty", None),
                        help="重复惩罚系数（不传使用模型默认）")

    parser.add_argument("--do_sample", action="store_true",
                        default=cfg.get("do_sample", False),
                        help="启用采样（否则使用贪婪解码）")

    parser.add_argument("--enable_history", action="store_true",
                        default=cfg.get("enable_history", False),
                        help="启用多轮对话历史")
    
    parser.add_argument("--training_stage", type=str,
                        default=cfg.get("training_stage","sft"),
                        choices=["sft", "dpo", "pretrain"],
                        help="模型训练阶段：sft（指令微调）、dpo（偏好优化）、pretrain（预训练）。")

    args = parser.parse_args()
    args_dict = vars(args)
    args_dict.pop("yaml_config", None)

    start_chat_session(**args_dict)