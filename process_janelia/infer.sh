#!/bin/bash

MODELS=("timemix" "tsmixer")
SUBJECTS=(1 2 3 4 5 6 7 12 13 14 15 16 17)

for MODEL_NAME in "${MODELS[@]}"; do
    for SUBJECT_ID in "${SUBJECTS[@]}"; do
        SUBJECT_ID_PADDED=$(printf "%02d" $SUBJECT_ID)
        TRAIN_WORKDIR="/mnt/storage/misc/exp/${MODEL_NAME}/subject_${SUBJECT_ID_PADDED}"
        INFER_WORKDIR="/mnt/storage/misc/zapbench/inference/n_steps_32/${MODEL_NAME}/subject_${SUBJECT_ID_PADDED}"
        if [ -d "$INFER_WORKDIR" ]; then
            echo "skipping inference ${MODEL_NAME}; subject ${SUBJECT_ID_PADDED} - directory already exists: $INFER_WORKDIR"
        else
            echo "running inference for ${MODEL_NAME}; subject ${SUBJECT_ID_PADDED}..."
            if ! CUDA_VISIBLE_DEVICES=7 python /home/sebastian/git/zapbench/zapbench/ts_forecasting/main_infer.py \
                --config /home/sebastian/git/zapbench/zapbench/ts_forecasting/configs/infer.py:exp_workdir=$TRAIN_WORKDIR \
                --workdir $INFER_WORKDIR; then
                echo "ERROR: inference failed for ${MODEL_NAME}; subject ${SUBJECT_ID_PADDED}; continuing to next..."
                continue
            fi
        fi
    done
done

echo "Inference complete!"
