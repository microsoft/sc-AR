#!/bin/bash

# Arrays of values for each variable
seed_values=(42 43 44 45 46)
ARtype_values=("T" "F")
latent_dim_values=(64)
Atlas_cell_count=(0 1 10 100 1000 10000 50000)

# activate the conda environment
eval "$(conda shell.bash hook)"
source activate scAR-env


# Loop through each combination of the variables
for seed in "${seed_values[@]}"; do
    for ARtype in "${ARtype_values[@]}"; do

        AR='False'
        # check if ARtype_values == "AR", then AR='T', if ARtype_values == "Naive", then AR='F'
        if [[ ${ARtype} == "T" ]]; then
            AR='True'
        fi
    
        for latent_dim in "${latent_dim_values[@]}"; do
            for Atlas_cell_count in "${Atlas_cell_count[@]}"; do
                echo "Running classification validation with:" 
                echo "seed=${seed}" 
                echo "ARtype=${ARtype}" 
                echo "latent_dim=${latent_dim}" 
                echo "Atlas_cell_count=${Atlas_cell_count}"

                python -u ../eval-scripts/zero_shot_classification.py \
                    scVI \
                    "../saved_models/sctab/seed${seed}/bloodbase-${Atlas_cell_count}atlas-AR${ARtype}-oodT-varconF-per1.0-lr5e-05-wd5e-05-bs4096-ldim${latent_dim}-epoch300-scvi-hvg2000-s${seed}-best" \
                    ${seed} \
                    "../result/test/scVI-classification-evals" \
                    ${ARtype} \
                    ${latent_dim} \
                    ${Atlas_cell_count} \
                    "../data/sctab/bloodbase_${Atlas_cell_count}_atlas_seed${seed}_AR${AR}_train_adata_2000_2000HVGs.h5ad"

            done
        done
    done
done
