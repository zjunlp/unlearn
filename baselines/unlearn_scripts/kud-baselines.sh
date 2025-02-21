#!/bin/bash
master_port=28132
set -e

data_subset="privacy"

forget_data_path="../../dataset/augument_data/knowundo_${data_subset}.json"
retain_data_path="../../dataset/KnowUnDo/${data_subset}/retention_train.json"

idonknow_file_path="../../dataset/idontknow.txt"

model_family=kud-llama2-7b
model_path="../../paper_models/llama2-7b_lora_kud_privacy/"
lr=1e-5
num_epochs=5
ds_config="../config/ds_z0_config.json"
max_length=512
loss_types=( "ga_gdr" "ga_klr" "ga_gdr_sure" "ga_klr_sure" "npo_gdr" "npo_klr" "npo_gdr_sure" "npo_klr_sure" )

for loss_type in "${loss_types[@]}"; do
    echo $loss_type
    save_dir="../../memory/${model_family}_${loss_type}_${data_subset}_${max_length}_${lr}"
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$master_port ../unlearn.py --config-name=forget_lora.yaml batch_size=1 gradient_accumulation_steps=8 model_family=${model_family} lr=${lr} model_path=${model_path} forget_data_path=${forget_data_path} retain_data_path=${retain_data_path} idonknow_file_path=${idonknow_file_path} loss_type=${loss_type} ds_config=${ds_config} max_length=${max_length} save_dir=${save_dir} num_epochs=${num_epochs}
done
