#!/bin/bash
set -e

memory_dir="../memory"
pretrained__model_path="../paper_models/llama2-7b_lora_kud_privacy"

forget_val_data_path="../dataset/KnowUnDo/privacy/unlearn_val.json"
retain_val_data_path="../dataset/KnowUnDo/privacy/retention_val.json"

output_file_dir="../kud-llama-inf"

mkdir -p "$output_file_dir"

for adapter_dir in "$memory_dir"/*; do
    if [ -d "$adapter_dir" ]; then
        adapter_name=$(basename "$adapter_dir")

        if [[ "$adapter_name" == llama2* && "$adapter_name" != *-full ]] ; then
            for checkpoint_dir in "$adapter_dir"/*; do
                if [ -d "$checkpoint_dir" ]; then
                    checkpoint_name=$(basename "$checkpoint_dir")

                    if [[ "$checkpoint_name" == *-full ]]; then
                        method="${adapter_name}__${checkpoint_name}"

                        output_file_forget="$output_file_dir/${method}__forget.json"
                        output_file_retain="$output_file_dir/${method}__retain.json"

                        if [ -f "$output_file_forget" ] && [ -f "$output_file_retain" ]; then
                            echo "Output files for $method already exist. Skipping..."
                            continue
                        fi

                        CUDA_VISIBLE_DEVICES=0 python generate.py \
                            --model_path "$checkpoint_dir" \
                            --forget_val_data_path "$forget_val_data_path" \
                            --retain_val_data_path "$retain_val_data_path" \
                            --output_file_forget "$output_file_forget" \
                            --output_file_retain "$output_file_retain" \
                            --use_vllm 
                    fi
                fi
            done
        fi
    fi
done



method="pretrained__model"

output_file_forget="$output_file_dir/${method}__forget.json"
output_file_retain="$output_file_dir/${method}__retain.json"

if [ -f "$output_file_forget" ] && [ -f "$output_file_retain" ]; then
    echo "Output files for $method already exist. Skipping..."
else
    CUDA_VISIBLE_DEVICES=0 python generate.py \
        --model_path "$pretrained_model_path" \
        --forget_val_data_path "$forget_val_data_path" \
        --retain_val_data_path "$retain_val_data_path" \
        --output_file_forget "$output_file_forget" \
        --output_file_retain "$output_file_retain" \
        --use_vllm
fi