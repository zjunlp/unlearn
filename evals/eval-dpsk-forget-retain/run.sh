set -e
# conda activate unlearn
forget_data_path="../kud-gemma-inf/gemma-2-2b-it_kud_forget_candidates.json"
retain_data_path="../kud-gemma-inf/gemma-2-2b-it_kud_retain_candidates.json"

mkdir -p "../kud-gemma-gpt"
forget_save_path="../kud-gemma-gpt/gemma-2-2b-it_kud_forget_candidates_evaluated.json"
retain_save_path="../kud-gemma-gpt/gemma-2-2b-it_kud_retain_candidates_evaluated.json"

python forget_retain_dpsk.py \
    --data_path $forget_data_path \
    --save_path $forget_save_path

python forget_retain_dpsk.py \
    --data_path $retain_data_path \
    --save_path $retain_save_path