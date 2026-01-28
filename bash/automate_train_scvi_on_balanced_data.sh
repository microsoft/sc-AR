#!/bin/bash

############################
# Fixed parameters
############################
data="sctab"
data_path="./data"
out_path="./saved_models/"
log_path="./log"
root=".."
model_name="scvi"
latent_dim=64
num_epoch=300
alpha=0.0001
AR=False

############################
# Parameter grids
############################
seeds=(42 43 44 45 46)
atlas_counts=(0 1 10 100 1000 10000 50000)
balancing_methods=("class_balancing" "geometric_sketching")

############################
# Loop over all combinations
############################
for balancing_method in "${balancing_methods[@]}"; do
    for seed in "${seeds[@]}"; do
        for atlas_count in "${atlas_counts[@]}"; do

        train_adata_path="${data_path}/${data}/balanced_data/${balancing_method}/${seed}/bloodbase_${atlas_count}atlas_seed${seed}_2000HVGs_balancing_method${balancing_method}_train_adata.h5ad"
        valid_adata_path="${data_path}/${data}/balanced_data/${balancing_method}/${seed}/bloodbase_${atlas_count}atlas_seed${seed}_2000HVGs_balancing_method${balancing_method}_valid_adata.h5ad"

        echo "======================================"
        echo "Running:"
        echo "  seed=${seed}"
        echo "  atlas_count=${atlas_count}"
        echo "  balancing_method=${balancing_method}"
        echo "  AR=${AR}"
        echo "======================================"

        bash train_scvi_on_balanced_data.sh \
            ${data} \
            ${data_path} \
            ${out_path} \
            ${seed} \
            ${log_path} \
            ${root} \
            ${model_name} \
            ${latent_dim} \
            ${num_epoch} \
            ${AR} \
            ${atlas_count} \
            ${alpha} \
            ${balancing_method} \
            ${train_adata_path} \
            ${valid_adata_path}

        done
    done
done
