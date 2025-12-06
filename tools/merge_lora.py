import os
import argparse
import sys

current_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

import src.bumblecore.bumblebee


def parse_dtype(dtype_str):
    if dtype_str == "auto":
        return "auto"
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Invalid dtype: {dtype_str}. Choose from 'auto', 'float16', 'bfloat16', 'float32'.")
    return dtype_map[dtype_str]

def merge_lora_into_base_model(
    base_model_path: str,
    lora_model_path: str,
    save_path: str,
    device_map: str,
    dtype: str,
    trust_remote_code: bool,
    low_cpu_mem_usage: bool
):
    os.makedirs(save_path, exist_ok=True)
    print(f"Loading base model from {base_model_path}...")

    dtype = parse_dtype(dtype)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=low_cpu_mem_usage,
        trust_remote_code=trust_remote_code
    )

    print(f"Loading and merging LoRA weights from {lora_model_path}...")
    model = PeftModel.from_pretrained(model, lora_model_path, device_map=device_map)
    model = model.merge_and_unload()

    print(f"Loading tokenizer from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=trust_remote_code)

    print(f"Saving merged model and tokenizer to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base model")
    parser.add_argument("--lora_model_path", type=str, required=True, help="Path to the LoRA adapter")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the merged model")
    parser.add_argument("--device_map", type=str, default="auto", help='Device map (default: "auto","cpu")')
    parser.add_argument("--dtype", type=str, default="auto", help='Torch dtype (e.g., "float16", "bfloat16", "float32", "auto")')
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code")
    parser.add_argument("--low_cpu_mem_usage", action="store_true", help="Use low CPU memory")
    args = parser.parse_args()
    merge_lora_into_base_model(
        base_model_path=args.base_model_path,
        lora_model_path=args.lora_model_path,
        save_path=args.save_path,
        device_map=args.device_map,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=args.low_cpu_mem_usage
    )

