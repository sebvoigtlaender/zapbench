#!/bin/bash

MODELS=("mean" "linear" "timemix" "tsmixer" "tide")
TIMESTEPS_INPUT=(4 32)

for MODEL_NAME in "${MODELS[@]}"; do
    for T in "${TIMESTEPS_INPUT[@]}"; do
        echo "training ${MODEL_NAME}..."
        TRAIN_WORKDIR="/mnt/storage/misc/zapbench/training/n_steps_${T}/${MODEL_NAME}/zapbench"
        if ! CUDA_VISIBLE_DEVICES=7 python /home/sebastian/git/zapbench/zapbench/ts_forecasting/main_train.py \
            --config /home/sebastian/git/zapbench/zapbench/ts_forecasting/configs/${MODEL_NAME}.py:dataset_name=240930_traces,runlocal=False,timesteps_input=$T \
            --workdir "$TRAIN_WORKDIR"; then
            echo "ERROR: training failed for ${MODEL_NAME}; continuing"
            continue
        fi
    done
done

echo "zapbench training complete!"
