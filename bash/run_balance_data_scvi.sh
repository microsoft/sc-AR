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
DATA="sctab"
ANNDATA_BLOOD_FILE="data/sctab/BaseModel_scTabBloodOnly_seed0_bloodbase_TrainingData.h5ad"
ANNDATA_ATLAS_FILE="data/sctab/BaseModel_scTabAll_seed0_allbase_TrainingData.h5ad"
OBS_CLASS_LABEL=("tissue")
TRAINING_DATA_COUNT=100000
HVG_COUNT=2000
NUM_COVERING_BOXES=100
ROOT="/Users/zeinab/Documents/MSR_internship/project/sc-AR-github-repo/sc-AR/"
MODEL="scvi"

# Loop variables
SEEDS=(10 11 12 13 14)
ATLAS_COUNTS=(0 1 10 100 1000 10000 50000)
BALANCING_METHODS=("class_balancing" "geometric_sketching")

for SEED in "${SEEDS[@]}"; do
  for ATLAS_COUNT in "${ATLAS_COUNTS[@]}"; do
    for BAL_METHOD in "${BALANCING_METHODS[@]}"; do

      echo "Running seed=${SEED}, atlas_count=${ATLAS_COUNT}, balancing_method=${BAL_METHOD}"

      ${PYTHON} ../scripts/balance_data.py \
        --data "${DATA}" \
        --anndata_blood_file "${ANNDATA_BLOOD_FILE}" \
        --anndata_atlas_file "${ANNDATA_ATLAS_FILE}" \
        --obs_class_label "${OBS_CLASS_LABEL[@]}" \
        --training_data_count "${TRAINING_DATA_COUNT}" \
        --seed "${SEED}" \
        --atlas_count "${ATLAS_COUNT}" \
        --hvg_count "${HVG_COUNT}" \
        --balancing_method "${BAL_METHOD}" \
        --num_covering_boxes "${NUM_COVERING_BOXES}" \
        --root "${ROOT}" \
        --model "${MODEL}" >> log/$today/balance_data_${SEED}_${ATLAS_COUNT}_${BAL_METHOD}.log 2> log/$today/balance_data_${SEED}_${ATLAS_COUNT}_${BAL_METHOD}.err

    if grep -q "Traceback" log/$today/balance_data_${SEED}_${ATLAS_COUNT}_${BAL_METHOD}.err; then
      echo "Error"
    fi

    done
  done
done
