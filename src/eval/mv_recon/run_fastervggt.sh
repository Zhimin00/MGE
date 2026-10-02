#!/bin/bash

set -e
workdir='..'
model_name='FasterVGGT'

output_dir="${workdir}/eval_results/mv_recon/${model_name}_0.5new"
echo "$output_dir"
accelerate launch --num_processes 1 --main_process_port 29602 ./eval/mv_recon/launch.py \
    --weights "$model_weights" \
    --output_dir "$output_dir" \
    --model_name "$model_name" \
     