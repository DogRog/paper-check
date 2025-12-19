#!/bin/bash
# Export the finetuned Qwen3 model to Hugging Face Hub
python export_model.py \
  --model_path "outputs/qwen3_14b_full_run" \
  --hub_repo "LeeundEr/Qwen3-14B-Finetune" \
  --mode merged_16bit \
  --max_seq_length 32768