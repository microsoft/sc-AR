#!/bin/bash
#SBATCH -n 1
#SBATCH --mem=300G
#SBATCH -t 04:00:00
#SBATCH -c 45
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH -J gpu_eval_scAR
#SBATCH -o output/%x_%j.out    
#SBATCH -e error/%x_%j.err    

# Email notifications
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=akshaya_thoutam@brown.edu

# Unload all modules (important)
module purge

## Please check the version of cuda and cudnn
module load cuda/11.8.0-kuhf
# module load cudnn/8.7.0.84-11.8-kff3

# activate the conda environment
eval "$(conda shell.bash hook)"
source activate scar-env
export WANDB_KEY=''

# Debug: confirm which python is being used
which python
# python -c "import sys; print('Python path:', sys.executable); import scanpy; print('Scanpy OK')"
# python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
python -c "import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"


# Arrays of values for each variable
seed_values=(50 60 70 80 90)
ARtype_values=("T" "F")
Atlas_cell_count_values=(0 1 10 100 1000 10000 50000)
Alpha_values=0.0001

# Activate the conda environment
eval "$(conda shell.bash hook)"
source activate scar-env

# (latent_dim, num_epoch) pairs to try
latent_dim_epoch_pairs=(
    "64 150"
    "64 600"
    "32 300"
    "128 300"
)

# Loop through each combination of the variables
for seed in "${seed_values[@]}"; do
    for latent_dim_epoch_pair in "${latent_dim_epoch_pairs[@]}"; do
        read -r latent_dim num_epoch <<< "${latent_dim_epoch_pair}"
        for ARtype in "${ARtype_values[@]}"; do
            if [[ ${ARtype} == "T" ]]; then
                AR='True'
            else
                AR='False'
            fi
            for Atlas_cell_count in "${Atlas_cell_count_values[@]}"; do
                echo "Running reconstruction validation with:"
                echo "seed=${seed}"
                echo "ARtype=${ARtype}"
                echo "latent_dim=${latent_dim}"
                echo "Atlas_cell_count=${Atlas_cell_count}"
                echo "AR=${AR}"
                echo "alpha=${Alpha_values}"
                echo "num_epoch=${num_epoch}"

                python -u ../eval_scripts/LDVAE_eval.py \
                    "../data/sctab" \
                    "../data/sctab/bloodbase_${Atlas_cell_count}_atlas_seed${seed}_AR${AR}_train_adata_2000_2000HVGs.h5ad" \
                    "../saved_models/sctab/seed${seed}/bloodbase-${Atlas_cell_count}atlas-AR${ARtype}-lr5e-05-wd5e-05-bs4096-ldim${latent_dim}-alpha${Alpha_values}-epoch${num_epoch}-scvi-hvg2000-s${seed}-best" \
                    "../result/test/scVI-reconstruction-evals-for-different-k-and-epoch/" \
                    ${seed} \
                    ${ARtype} \
                    ${latent_dim} \
                    ${Atlas_cell_count} \
                    ${Alpha_values} \
                    ${num_epoch}
            done
        done
    done
done
