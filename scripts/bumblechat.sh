#!/bin/bash
CUDA_VISIBLE_DEVICES="0,1" python src/api.py \
    --yaml_config ./configs/inference/bumblechat.yaml \
    --model_path <your model path> \
    --device_map cpu