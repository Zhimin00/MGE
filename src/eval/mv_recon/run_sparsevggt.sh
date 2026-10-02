#!/bin/bash

set -e
workdir='..'
model_name='SparseVGGT'
ckpt_name='checkpoint-final'
model_weights="${workdir}/checkpoints/SparseVGGT_alpha0.1_lr1e-5_nog/${ckpt_name}.pth"

output_dir="${workdir}/eval_results/mv_recon_dense/${model_name}_${ckpt_name}_nomask"
echo "$output_dir"
accelerate launch --num_processes 1 --main_process_port 29602 ./eval/mv_recon/launch.py \
    --weights "$model_weights" \
    --output_dir "$output_dir" \
    --model_name "$model_name" \
     