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
ALPHA=0.0001   # manuscript default; held fixed for bin sensitivity

############################
# Sweep parameters
############################
SEEDS=(42 43 44 45 46)
ATLAS_COUNTS=(0 1 10 100 1000 10000 50000)
BINS=(100) #(50 80 120 150)   # add 100 to compare against manuscript default

############################
# Script to call
############################
TRAIN_SCRIPT="./train_scvi_bins.sh"

############################
# Loop
############################
for SEED in "${SEEDS[@]}"; do
  for ATLAS_COUNT in "${ATLAS_COUNTS[@]}"; do
    for BIN in "${BINS[@]}"; do

      echo "=============================================="
      echo "Running:"
      echo "  seed=${SEED}"
      echo "  atlas_count=${ATLAS_COUNT}"
      echo "  bins=${BIN}"
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
        ${ALPHA} \
        ${BIN}

    done
  done
done
