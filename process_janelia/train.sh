#!/bin/bash

# MODELS=("timemix")
# SUBJECTS=(1 2 3 4 5 6 7 12 13 14 15 16 17)
MODELS=("tsmixer", "tide")
SUBJECTS=(1 6 15 17)

for MODEL_NAME in "${MODELS[@]}"; do
    for SUBJECT_ID in "${SUBJECTS[@]}"; do
        SUBJECT_ID_PADDED=$(printf "%02d" $SUBJECT_ID)
        TRAIN_WORKDIR="/mnt/storage/misc/exp/${MODEL_NAME}/subject_${SUBJECT_ID_PADDED}"
        if [ -d "$TRAIN_WORKDIR" ]; then
            echo "Skipping training ${MODEL_NAME}; subject ${SUBJECT_ID_PADDED} - directory already exists: $TRAIN_WORKDIR"
        else
            echo "Training ${MODEL_NAME}; subject ${SUBJECT_ID_PADDED}..."
            if ! CUDA_VISIBLE_DEVICES=7 python /home/sebastian/git/zapbench/zapbench/ts_forecasting/main_train.py \
                --config /home/sebastian/git/zapbench/zapbench/ts_forecasting/configs/${MODEL_NAME}.py:dataset_name=subject_${SUBJECT_ID_PADDED},runlocal=False \
                --workdir "$TRAIN_WORKDIR"; then
                echo "ERROR: Training failed for ${MODEL_NAME}; subject ${SUBJECT_ID_PADDED}. Continuing to next..."
                continue
            fi
        fi
    done
done

echo "All processing complete!"
