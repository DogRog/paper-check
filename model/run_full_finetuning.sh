#!/bin/bash
# Run full finetuning for Qwen3
python finetune_qwen.py \
    --mode full \
    --disable_thinking \
    --max_seq_length 32768 \
    --packing
