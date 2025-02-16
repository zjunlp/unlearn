#!/bin/bash

data_path="../dataset/TOFU/forget10.jsonl"
model="zhipu"
save_path="../dataset/augument_data/tofu.jsonl"

python proc.py --data_path $data_path --model $model 

python gather_proc_data.py --data_path $data_path --save_path $save_path