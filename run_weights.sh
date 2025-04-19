#!/bin/bash
id=0

for id in 0 1
do
    WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 python finetune_weight.py \
        --base_model '/path/to/Llama2-7b' \
        --cutoff_len=1024 \
        --group_by_length \
        --task_id=2 \
        --data_id=${id}  \
        --prompt_template_name 'llama2'
done

for id in 0 1
do
    WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0,1 python finetune_weight.py \
        --base_model '/path/to/Qwen1.5-14B-Chat' \
        --cutoff_len=1024 \
        --group_by_length \
        --task_id=2 \
        --data_id=${id}  \
        --prompt_template_name 'qwen'
done