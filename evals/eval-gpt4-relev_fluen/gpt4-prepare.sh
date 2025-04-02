set -e
# conda activate unlearn
data_dir="../tofu-llama2-inf"
output_prefix="llama2-7b_tofu"
python relev_fluen_datapre.py \
    --data_dir $data_dir \
    --output_prefix $output_prefix