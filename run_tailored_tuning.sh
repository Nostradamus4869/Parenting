#!/bin/bash
begin_id=0



# python ipt_analyse.py \
#     --model_name 'llama2' \
#     --dataset_id=2 \
#     --service_begin_id=0  \
#     --adherence_checkpoint=6000 \
#     --robustness_checkpoint=4000 \
#     --all_checkpoint=4000

python ipt_analyse.py \
    --model_name 'qwen' \
    --dataset_id=2 \
    --service_begin_id=0  \
    --adherence_checkpoint=3000 \
    --robustness_checkpoint=3000 \
    --all_checkpoint=4000
