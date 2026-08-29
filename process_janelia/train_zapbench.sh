#!/bin/bash

MODELS=("mean" "linear" "timemix" "tsmixer" "tide")
TIMESTEPS_INPUT=(4 32)
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the parent directory for training outputs}"

for MODEL_NAME in "${MODELS[@]}"; do
    for T in "${TIMESTEPS_INPUT[@]}"; do
        echo "training ${MODEL_NAME}..."
        TRAIN_WORKDIR="${OUTPUT_ROOT}/training/n_steps_${T}/${MODEL_NAME}/zapbench"
        if ! python zapbench/ts_forecasting/main_train.py \
            --config "zapbench/ts_forecasting/configs/${MODEL_NAME}.py:dataset_name=240930_traces,runlocal=False,timesteps_input=${T}" \
            --workdir "$TRAIN_WORKDIR"; then
            echo "ERROR: training failed for ${MODEL_NAME}; continuing"
            continue
        fi
    done
done

echo "zapbench training complete!"
