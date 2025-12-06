export CUDA_VISIBLE_DEVICES="6,7"
python tools/merge_lora.py \
  --base_model_path <your base model path> \
  --lora_model_path <your lora model path> \
  --save_path <your save path> \
  --dtype auto \
  --device_map auto \
  --trust_remote_code \
  --low_cpu_mem_usage

