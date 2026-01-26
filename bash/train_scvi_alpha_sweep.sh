#!/bin/bash
set -e

############################
# Fixed parameters
############################
DATA="sctab"
DATA_PATH="/path/to/dataset/"
OUT_PATH="/path/to/saved_models/"
LOG_PATH="./log"
ROOT="/path/to/project/root"

MODEL_NAME="scvi"
LATENT_DIM=64
NUM_EPOCH=300
AR=True

############################
# Sweep parameters
############################
SEEDS=(42 43 44 45 46)
ATLAS_COUNTS=(0 1 10 100 1000 10000 50000)
ALPHAS=(0.001 0.00001)

############################
# Script to call
############################
TRAIN_SCRIPT="./train_scvi.sh"

############################
# Loop
############################
for SEED in "${SEEDS[@]}"; do
  for ATLAS_COUNT in "${ATLAS_COUNTS[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do

      echo "=============================================="
      echo "Running:"
      echo "  seed=${SEED}"
      echo "  atlas_count=${ATLAS_COUNT}"
      echo "  alpha=${ALPHA}"
      echo "=============================================="

      bash ${TRAIN_SCRIPT} \
        ${DATA} \
        ${DATA_PATH} \
        ${OUT_PATH} \
        ${SEED} \
        ${LOG_PATH} \
        ${ROOT} \
        ${MODEL_NAME} \
        ${LATENT_DIM} \
        ${NUM_EPOCH} \
        ${AR} \
        ${ATLAS_COUNT} \
        ${ALPHA}

    done
  done
done
