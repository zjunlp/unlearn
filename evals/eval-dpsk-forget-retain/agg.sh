set -e
# conda activate unlearn
forget_results="../kud-gemma-gpt/gemma-2-2b-it_kud_forget_candidates_evaluated.json"
retain_results="../kud-gemma-gpt/gemma-2-2b-it_kud_retain_candidates_evaluated.json"
output_file="../kud-gemma-gpt/gemma-2-2b-it_kud_results.json"

model_name="gemma-2-2b-it_kud"
python compute_forget_retain.py \
    --forget_results $forget_results \
    --retain_results $retain_results \
    --output $output_file \
    --model_name $model_name