set -e
# conda activate unlearn
forget_results="../tofu-llama2-gpt/llama2-7b_tofu_forget_candidates_evaluated.json"
retain_results="../tofu-llama2-gpt/llama2-7b_tofu_retain_candidates_evaluated.json"
output_file="../tofu-llama2-gpt/llama2-7b_tofu_results.json"

model_name="llama2-7b_tofu"
python compute_relev_fluen.py \
    --forget_results $forget_results \
    --retain_results $retain_results \
    --output $output_file \
    --model_name $model_name