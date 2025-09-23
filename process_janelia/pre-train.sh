#!/bin/bash

MODELS=("linear" "timemix")
TIMESTEPS_INPUT=(4 32)

for MODEL_NAME in "${MODELS[@]}"; do
    for T in "${TIMESTEPS_INPUT[@]}"; do
        echo "pre-training ${MODEL_NAME}..."
        TRAIN_WORKDIR="/mnt/storage/misc/zapbench/training/n_steps_${T}/${MODEL_NAME}/janelia_pretrain"
        if ! CUDA_VISIBLE_DEVICES=7 python /home/sebastian/git/zapbench/zapbench/ts_forecasting/main_train.py \
            --config /home/sebastian/git/zapbench/zapbench/ts_forecasting/configs/${MODEL_NAME}.py:dataset_name=janelia_pretrain,runlocal=False,timesteps_input=$T \
            --workdir "$TRAIN_WORKDIR"; then
            echo "ERROR: pre-training failed for ${MODEL_NAME}; continuing"
            continue
        fi
    done
done

echo "pre-training complete!"
