#!/bin/bash
id=0

for id in 0 1
do
    WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0,1 python finetune_twoTask.py \
        --base_model '/path/to/Llama2-7b' \
        --learning_rate=1e-4 \
        --num_epochs=0.1 \
        --cutoff_len=1024 \
        --group_by_length \
        --lora_target_modules='[q_proj,v_proj,k_proj,o_proj]' \
        --micro_batch_size=1 \
        --batch_size=1 \
        --task_id=2 \
        --data_id=${id}  \
        --prompt_template_name 'llama2'
done

for id in 0
do
    WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0,1 python finetune_twoTask.py \
        --base_model '/path/to/Qwen1.5-14B-Chat' \
        --learning_rate=1e-4 \
        --num_epochs=0.07 \
        --cutoff_len=1024 \
        --group_by_length \
        --lora_target_modules='[q_proj,v_proj,k_proj,o_proj]' \
        --micro_batch_size=1 \
        --batch_size=1 \
        --task_id=2 \
        --data_id=${id}  \
        --prompt_template_name 'qwen'
done