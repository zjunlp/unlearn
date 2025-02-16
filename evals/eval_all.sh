#!/bin/bash
set -e

language_model_path="../../Llama-2-7b-chat-hf/" # Path to the HF model before pretraining
embedding_model_path="../../all-MiniLM-L12-v2" # Path to the MiniLM model 
entailment_model_path="../../deberta-v3-base-tasksource-nli" # Path to the nli model

memory_dir="../memory"

output_dir="../kud-llama-eval"

results_dir="../kud-llama-inf"

if [ ! -d "$output_dir" ]; then
  mkdir -p "$output_dir"
fi

for result_file in "$results_dir"/*_forget.json; do
    base_name=$(basename "$result_file" "__forget.json")
    
    forget_path="$results_dir/${base_name}__forget.json"
    retain_path="$results_dir/${base_name}__retain.json"
    
    if [ -f "$forget_path" ] && [ -f "$retain_path" ]; then
        test_model_name="$base_name"
        
        result_path="$output_dir/${test_model_name}.json"
        
        if [ -f "$result_path" ]; then
            echo "Result file for $test_model_name already exists. Skipping..."
            continue
        fi
        
        python evaluate.py \
            --language_model_path "$language_model_path" \
            --embedding_model_path "$embedding_model_path" \
            --entailment_model_path "$entailment_model_path" \
            --test_model_name "$test_model_name" \
            --forget_path "$forget_path" \
            --retain_path "$retain_path" \
            --output_path "$result_path"
    else
        echo "Warning: Missing files for $base_name. Skipping..."
    fi
done

pretrained_forget_path="$results_dir/pretrained__model__forget.json"
pretrained_retain_path="$results_dir/pretrained__model__retain.json"

pretrained_model_name="pretrained__model"

pretrained_result_path="$output_dir/${pretrained_model_name}.json"

if [ -f "$pretrained_forget_path" ] && [ -f "$pretrained_retain_path" ]; then
    if [ -f "$pretrained_result_path" ]; then
        echo "Result file for $pretrained_model_name already exists. Skipping..."
    else
        python evaluate.py \
            --language_model_path "$language_model_path" \
            --embedding_model_path "$embedding_model_path" \
            --entailment_model_path "$entailment_model_path" \
            --test_model_name "$pretrained_model_name" \
            --forget_path "$pretrained_forget_path" \
            --retain_path "$pretrained_retain_path" \
            --output_path "$pretrained_result_path"
    fi
else
    echo "Warning: Missing pretrained model files for evaluation. Skipping..."
fi