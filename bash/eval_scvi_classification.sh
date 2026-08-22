#!/bin/bash

# Bin-sensitivity evaluation (rebuttal round 2): alpha fixed, sweep histogram bins.

seed_values=(42 43 44 45 46)
ARtype_values=("T") # "F")
latent_dim_values=(64)
Atlas_cell_count=(0 1 10 100 1000 10000 50000)
ALPHA=0.0001
EPOCH=300
BINS_VALUES=(50 80 120 150)

# activate the conda environment
eval "$(conda shell.bash hook)"
source activate scAR-env

for seed in "${seed_values[@]}"; do
    for ARtype in "${ARtype_values[@]}"; do
        AR='False'
        if [[ ${ARtype} == "T" ]]; then
            AR='True'
        fi

        for latent_dim in "${latent_dim_values[@]}"; do
            for Atlas_cell_count in "${Atlas_cell_count[@]}"; do
                for bins in "${BINS_VALUES[@]}"; do
                    echo "Running classification validation with:"
                    echo "seed=${seed}"
                    echo "ARtype=${ARtype}"
                    echo "latent_dim=${latent_dim}"
                    echo "Atlas_cell_count=${Atlas_cell_count}"
                    echo "alpha=${ALPHA}"
                    echo "bins=${bins}"
                    echo "epoch=${EPOCH}"

                    python -u ../eval_scripts/zero_shot_classification.py \
                        scVI \
                        "../saved_models/sctab/seed${seed}/bloodbase-${Atlas_cell_count}atlas-AR${ARtype}-lr5e-05-wd5e-05-bs4096-ldim${latent_dim}-alpha${ALPHA}-epoch${EPOCH}-scvi-hvg2000-bins${bins}-s${seed}-best" \
                        ${seed} \
                        "../result/test/scVI-classification-evals-bins" \
                        ${ARtype} \
                        ${latent_dim} \
                        ${Atlas_cell_count} \
                        "../data/sctab/bloodbase_${Atlas_cell_count}_atlas_seed${seed}_AR${AR}_train_adata_2000_2000HVGs.h5ad" \
                        ${ALPHA} \
                        ${EPOCH} \
                        ${bins}
                done
            done
        done
    done
done
