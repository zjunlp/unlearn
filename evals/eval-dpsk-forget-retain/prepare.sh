set -e
# conda activate unlearn
data_dir="../kud-gemma-inf"
output_prefix="gemma-2-2b-it_kud"
python forget_retain_datapre.py \
    --data_dir $data_dir \
    --output_prefix $output_prefix