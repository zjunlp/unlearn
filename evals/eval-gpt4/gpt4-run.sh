set -e
# conda activate unlearn
forget_data_path="../tofu-llama2-inf/llama2-7b_tofu_forget_candidates.json"
retain_data_path="../tofu-llama2-inf/llama2-7b_tofu_retain_candidates.json"
forget_save_path="../tofu-llama2-gpt/llama2-7b_tofu_forget_candidates_evaluated.json"
retain_save_path="../tofu-llama2-gpt/llama2-7b_tofu_retain_candidates_evaluated.json"

python relvev_fluen_gpt4o.py \
    --data_path $forget_data_path \
    --save_path $forget_save_path

python relvev_fluen_gpt4o.py \
    --data_path $retain_data_path \
    --save_path $retain_save_path