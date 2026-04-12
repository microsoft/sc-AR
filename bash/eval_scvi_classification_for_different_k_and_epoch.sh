#!/bin/bash

# Arrays of values for each variable
seed_values=(50 60 70 80 90)
ARtype_values=("F" "T")
Atlas_cell_counts=(0 1 10 100 1000 10000 50000)
Alpha_values=0.0001

# activate the conda environment
eval "$(conda shell.bash hook)"
source activate scAR-env


latent_dim_epoch_pairs=(
    "64 150"
    "64 600"
    "32 300"
    "128 300"
)

# Loop through each combination of the variables
for seed in "${seed_values[@]}"; do
    for latent_dim_epoch_pair in "${latent_dim_epoch_pairs[@]}"; do
        # Bash: scalar "64 150" has [0] == full string and [1] empty — split explicitly
        read -r latent_dim num_epoch <<< "${latent_dim_epoch_pair}"
        for ARtype in "${ARtype_values[@]}"; do
            AR='False'
            if [[ ${ARtype} == "T" ]]; then
                AR='True'
            fi
            for Atlas_cell_count in "${Atlas_cell_counts[@]}"; do

                echo "Running classification validation with:"
                echo "seed=${seed}"
                echo "ARtype=${ARtype}"
                echo "latent_dim=${latent_dim}"
                echo "Atlas_cell_count=${Atlas_cell_count}"
                echo "alpha=${Alpha_values}"
                echo "num_epoch=${num_epoch}"

                python -u ../eval_scripts/zero_shot_classification.py \
                    scVI \
                    "../saved_models/sctab/seed${seed}/bloodbase-${Atlas_cell_count}atlas-AR${ARtype}-lr5e-05-wd5e-05-bs4096-ldim${latent_dim}-alpha${Alpha_values}-epoch${num_epoch}-scvi-hvg2000-s${seed}-best" \
                    ${seed} \
                    "../result/test/scVI-classification-evals-for-different-k-and-epoch/" \
                    ${ARtype} \
                    ${latent_dim} \
                    ${Atlas_cell_count} \
                    "../data/sctab/bloodbase_${Atlas_cell_count}_atlas_seed${seed}_AR${AR}_train_adata_2000_2000HVGs.h5ad" \
                    ${Alpha_values}
            done
        done
    done
done
