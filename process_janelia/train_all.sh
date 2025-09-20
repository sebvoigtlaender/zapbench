#!/bin/bash

set -e

GPUS=(5 6 7)
SUBJECT_IDS=("02" "06" "12" "14" "17")
MODELS=("linear" "tide" "timemix" "tsmixer")
BASE_WORKDIR="/home/sebastian/logs/zapbench"

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            IFS=',' read -ra GPUS <<< "$2"
            shift 2
            ;;
    esac
done

gpu_idx=0
for subject_id in "${SUBJECT_IDS[@]}"; do
    for model in "${MODELS[@]}"; do
        gpu=${GPUS[$gpu_idx]}
        echo "Training $model on subject $subject_id on GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu python zapbench/ts_forecasting/main_train.py --config zapbench/ts_forecasting/configs/$model.py:dataset_name="subject_$subject_id" --workdir $BASE_WORKDIR/$model/subject_$subject_id &
        gpu_idx=$(( (gpu_idx + 1) % ${#GPUS[@]} ))
    done
done
wait
