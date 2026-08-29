#!/bin/bash

# MODELS=("tsmixer" "tide")
# SUBJECTS=(1 6 15 17)
MODELS=("mean")
SUBJECTS=(1 2 3 4 5 6 7 12 13 14 15 16 17)
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the parent directory for training outputs}"
: "${ROOT_PATH:?Set ROOT_PATH to the directory whose ts_files/ contains the registered stores}"

for MODEL_NAME in "${MODELS[@]}"; do
    for SUBJECT_ID in "${SUBJECTS[@]}"; do
        SUBJECT_ID_PADDED=$(printf "%02d" $SUBJECT_ID)
        TRAIN_WORKDIR="${OUTPUT_ROOT}/training/n_steps_32/${MODEL_NAME}/subject_${SUBJECT_ID_PADDED}"
        if [ -d "$TRAIN_WORKDIR" ]; then
            echo "Skipping training ${MODEL_NAME}; subject ${SUBJECT_ID_PADDED} - directory already exists: $TRAIN_WORKDIR"
        else
            echo "Training ${MODEL_NAME}; subject ${SUBJECT_ID_PADDED}..."
            if ! python zapbench/ts_forecasting/main_train.py \
                --config "zapbench/ts_forecasting/configs/${MODEL_NAME}.py:dataset_name=subject_${SUBJECT_ID_PADDED},runlocal=False,timesteps_input=32" \
                --workdir "$TRAIN_WORKDIR"; then
                echo "ERROR: Training failed for ${MODEL_NAME}; subject ${SUBJECT_ID_PADDED}. Continuing to next..."
                continue
            fi
        fi
    done
done

echo "All processing complete!"
