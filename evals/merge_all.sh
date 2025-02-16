#!/bin/bash
set -e

base_model_path="../paper_models/kud-llama2-7b_lora_privacy"

memory_dir="../memory"

for adapter_dir in "$memory_dir"/*/; do
  adapter_name=$(basename "$adapter_dir")
  
  if [[ "$adapter_name" == llama2* ]]  && [[ "$adapter_name" != *-full ]]; then
    for checkpoint_dir in "$adapter_dir"*/; do
      if [[ "$checkpoint_dir" == *checkpoint* ]]; then
        checkpoint_name=$(basename "$checkpoint_dir")
        if [[ $checkpoint_name == *full ]]; then
          echo "${checkpoint_name} merged"
          continue
        fi

        save_checkpoint_dir="$adapter_dir/${checkpoint_name}-full"
        
        if [ -d "$save_checkpoint_dir" ]; then
          echo "Skipping $checkpoint_dir because $save_checkpoint_dir already exists."
          continue
        fi
        
        CUDA_VISIBLE_DEVICES=0 python merge_model.py \
          --base_model_path "$base_model_path" \
          --adapter_path "$checkpoint_dir" \
          --save_path "$save_checkpoint_dir"
      fi
    done
  fi
done
