#!/bin/bash

# Arrays of values for each variable
seed_values=(42 43 44 45 46)
ARtype_values=("T" "F")
latent_dim_values=(64)
Atlas_cell_count_values=(0 1 10 100 1000 10000 50000)

# Activate the conda environment
eval "$(conda shell.bash hook)"
source activate scar-env

# Loop through each combination of the variables
for seed in "${seed_values[@]}"; do
    for ARtype in "${ARtype_values[@]}"; do

        # Initialize AR variable based on ARtype
        if [[ ${ARtype} == "T" ]]; then
            AR='True'
        elif [[ ${ARtype} == "F" ]]; then
            AR='False'
        fi

        for latent_dim in "${latent_dim_values[@]}"; do
            for Atlas_cell_count in "${Atlas_cell_count_values[@]}"; do
                echo "Running reconstruction validation with:"
                echo "seed=${seed}"
                echo "ARtype=${ARtype}" 
                echo "latent_dim=${latent_dim}"
                echo "Atlas_cell_count=${Atlas_cell_count}"
                echo "AR=${AR}"

                python -u ../eval_scripts/LDVAE_eval.py \
                    "../data/sctab" \
                    "../data/stab/bloodbase_${Atlas_cell_count}_atlas_seed${seed}_AR${AR}_train_adata_2000_2000HVGs.h5ad" \
                    "../saved_models/sctab/seed${seed}/bloodbase-${Atlas_cell_count}atlas-AR${ARtype}-oodT-varconF-per1.0-lr5e-05-wd5e-05-bs4096-ldim${latent_dim}-epoch300-scvi-hvg2000-s${seed}-best" \
                    "../result/test/scVI-reconstruction-evals/" \
                    ${seed} \
                    ${ARtype} \
                    ${latent_dim} \
                    ${Atlas_cell_count}

            done
        done
    done
done
