#!/bin/bash

# MODELS=("tsmixer" "tide")
# SUBJECTS=("subject_01" "subject_05" "subject_15" "subject_17" "zapbench")
MODELS=("mean" "linear" "timemix")
SUBJECTS=("janelia_pretrain" "zapbench")
N_STEPS=(4 32)
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the parent directory for training and inference outputs}"
: "${ROOT_PATH:?Set ROOT_PATH to the directory whose ts_files/ contains the registered stores}"

for MODEL_NAME in "${MODELS[@]}"; do
    for SUBJECT_ID in "${SUBJECTS[@]}"; do
        for STEP in "${N_STEPS[@]}"; do
            TRAIN_WORKDIR="${OUTPUT_ROOT}/training/n_steps_${STEP}/${MODEL_NAME}/${SUBJECT_ID}"
            INFER_WORKDIR="${OUTPUT_ROOT}/inference/n_steps_${STEP}/${MODEL_NAME}/${SUBJECT_ID}"
            if [ -d "$INFER_WORKDIR" ]; then
                echo "skipping inference ${MODEL_NAME}; subject ${SUBJECT_ID} - directory already exists: $INFER_WORKDIR"
            else
                echo "running inference for ${MODEL_NAME}; ${SUBJECT_ID}..."
                echo "infer workdir ${INFER_WORKDIR}"
                echo "train workdir ${TRAIN_WORKDIR}"
                if ! python zapbench/ts_forecasting/main_infer.py \
                    --config "zapbench/ts_forecasting/configs/infer.py:exp_workdir=${TRAIN_WORKDIR}" \
                    --workdir "$INFER_WORKDIR"; then
                    echo "ERROR: inference failed for ${MODEL_NAME}; subject ${SUBJECT_ID}; continuing to next..."
                    continue
                fi
            fi
        done
    done
done

echo "inference complete!"
