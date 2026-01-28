#!/usr/bin/env bash

# Exit immediately if a command fails
set -e
today=$(date '+%Y%m%d')
# check if the log folder exists, if not create it
if [ ! -d log ]; then
  mkdir -p log;
fi

# check if the log folder for today exists, if not create it
if [ ! -d log/$today ]; then
  mkdir -p log/$today;
fi

# Path to python executable (adjust if needed)
PYTHON=python

# Fixed arguments
ANNDATA_FILE="data/species.h5ad"
OBS_CLASS_LABEL=("species" "condition")
NUM_COVERING_BOXES=100
ROOT="/Users/zeinab/Documents/MSR_internship/project/sc-AR-github-repo/sc-AR/"
MODEL="scgen"

# Loop variables
SEEDS=(100 101 102 103 104)
BALANCING_METHODS=("class_balancing" "geometric_sketching")

for SEED in "${SEEDS[@]}"; do

    for BAL_METHOD in "${BALANCING_METHODS[@]}"; do

        echo "Running seed=${SEED}, balancing_method=${BAL_METHOD}"

        ${PYTHON} ../scripts/balance_data.py \
        --anndata_file "${ANNDATA_FILE}" \
        --obs_class_label "${OBS_CLASS_LABEL[@]}" \
        --seed "${SEED}" \
        --balancing_method "${BAL_METHOD}" \
        --num_covering_boxes "${NUM_COVERING_BOXES}" \
        --root "${ROOT}" \
        --model "${MODEL}" >> log/$today/balance_data_${SEED}_${BAL_METHOD}.log 2> log/$today/balance_data_${SEED}_${BAL_METHOD}.err

    if grep -q "Traceback" log/$today/balance_data_${SEED}_${BAL_METHOD}.err; then
        echo "Error"
    fi

    done

done
