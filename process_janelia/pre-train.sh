#!/bin/bash

MODELS=("linear" "timemix")
TIMESTEPS_INPUT=(4 32)
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the parent directory for training outputs}"
: "${ROOT_PATH:?Set ROOT_PATH to the directory whose ts_files/ contains the registered stores}"

for MODEL_NAME in "${MODELS[@]}"; do
    for T in "${TIMESTEPS_INPUT[@]}"; do
        echo "pre-training ${MODEL_NAME}..."
        TRAIN_WORKDIR="${OUTPUT_ROOT}/training/n_steps_${T}/${MODEL_NAME}/janelia_pretrain"
        if ! python zapbench/ts_forecasting/main_train.py \
            --config "zapbench/ts_forecasting/configs/${MODEL_NAME}.py:dataset_name=janelia_pretrain,runlocal=False,timesteps_input=${T}" \
            --workdir "$TRAIN_WORKDIR"; then
            echo "ERROR: pre-training failed for ${MODEL_NAME}; continuing"
            continue
        fi
    done
done

echo "pre-training complete!"
