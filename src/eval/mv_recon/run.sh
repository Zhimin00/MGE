#!/bin/bash

set -e
workdir='..'
model_name='StreamVGGT'
ckpt_name='checkpoints' #checkpoint-3' #
model_weights="${workdir}/ckpt/${ckpt_name}.pth"
# model_weights="${workdir}/checkpoints/StreamVGGT_alpha0.1_lr1e-5_nog_more3epochs_old/${ckpt_name}.pth"

output_dir="${workdir}/eval_results/mv_recon_random10_mask_proj/${model_name}_${ckpt_name}"
echo "$output_dir"
accelerate launch --num_processes 1 --main_process_port 29602 ./eval/mv_recon/launch.py \
    --weights "$model_weights" \
    --output_dir "$output_dir" \
    --model_name "$model_name" \
    --use_proj
     