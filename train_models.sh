#!/bin/bash

# Training script for multiple models and subjects
# Runs training for each model-subject combination

set -e  # Exit on error

# MODELS=("mean" "linear" "tsmixer" "timemix")
# SUBJECTS=(5 6 14 16 17)
MODELS=("tsmixer" "timemix")
SUBJECTS=(5)

for MODEL_NAME in "${MODELS[@]}"; do
    for SUBJECT_ID in "${SUBJECTS[@]}"; do
        SUBJECT_ID_PADDED=$(printf "%02d" $SUBJECT_ID)
        echo "Training ${MODEL_NAME} for subject ${SUBJECT_ID_PADDED}..."
        python zapbench/ts_forecasting/main_train.py \
            --config "zapbench/ts_forecasting/configs/${MODEL_NAME}.py:dataset_name=subject_${SUBJECT_ID_PADDED}" \
            --workdir "/Users/s/vault/zapbench/${MODEL_NAME}/${SUBJECT_ID_PADDED}"
    done
done

echo "jobs completed"
