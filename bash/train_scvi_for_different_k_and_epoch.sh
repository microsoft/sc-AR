#!/bin/bash
#SBATCH -n 1
#SBATCH --mem=120G
#SBATCH -t 1:00:00
#SBATCH -c 2
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -J gpu_train_scAR
#SBATCH -o output/%x_%j.out    
#SBATCH -e error/%x_%j.err    

# Email notifications
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=akshaya_thoutam@brown.edu

# Unload all modules (important)
module purge

## Please check the version of cuda and cudnn
module load cuda/11.6.0-wmwrm7d
module load cudnn/8.7.0.84-11.8-lg2dpd5

# activate the conda environment
eval "$(conda shell.bash hook)"
source activate scAR-env
export WANDB_KEY=''

# Debug: confirm which python is being used
which python
python -c "import sys; print('Python path:', sys.executable); import scanpy; print('Scanpy OK')"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"



# Assign input parameters to variables
data=sctab            # data base name, e.g., sctab
data_path=./data       # path to the /dataset/ folder
out_path=./saved_models/        # path to the folder for saving trained models (ends with /saved_models/)
seeds=(50 60 70 80 90)            # seed number
log_path=./log        # path to save logs
root=../            # root folder of the project
model_name=scvi      # model type: scvi or scgen
ARs=(True False)           # AR flag: True or False
atlas_counts=(0 1 10 100 1000 10000 50000)  # number of atlas cells, e.g., 0, 1, 10, 100, 1000, 10000, 50000
alpha=0.0001        # alpha value for smoothing


latent_dim_epoch_pairs=(
    (64 150)
    (64 600)
    (32 300)
    (128 300)
)

# create log folder
today=$(date '+%Y%m%d')
if [ ! -d log ]; then
  mkdir -p ${log_path};
fi

if [ ! -d log/$today ]; then
  mkdir -p ${log_path}/$today;
fi


## Other parameters ##
ood=True
batch_size=4096


############################
# Loop
############################
echo "Starting the loop"
for seed in "${seeds[@]}"; do
    for latent_dim_epoch_pair in "${latent_dim_epoch_pairs[@]}"; do
        latent_dim=${latent_dim_epoch_pair[0]}
        num_epoch=${latent_dim_epoch_pair[1]}
        for AR in "${ARs[@]}"; do
            for atlas_count in "${atlas_counts[@]}"; do
                echo "Running scvi for seed ${seed}, num_epoch ${num_epoch}, AR ${AR}, atlas_count ${atlas_count}"

                echo "=============================================="
                echo "Running:"
                echo "  seed=${seed}"
                echo "  num_epoch=${num_epoch}"
                echo "  latent_dim=${latent_dim}"
                echo "  AR=${AR}"
                echo "  atlas_count=${atlas_count}"
                echo "=============================================="

                CUDA_VISIBLE_DEVICES=0 python -u ${root}/main.py \
                    --root ${root} \
                    --data ${data} \
                    --data_path ${data_path} \
                    --out_path ${out_path} \
                    --train True \
                    --test False \
                    --predict False \
                    --latent_dim ${latent_dim} \
                    --AR ${AR} \
                    --ood ${ood} \
                    --num_epoch ${num_epoch} \
                    --seed ${seed} \
                    --model_name "scvi" \
                    --atlas_count ${atlas_count} \
                    --batch_size ${batch_size} \
                    --alpha ${alpha} >> ${log_path}/$today/${today}-${data}-AR${AR}-ood${ood}-seed${seed}-epoch${num_epoch}-ldim${latent_dim}-${model_name}-${atlas_count}atlas-alpha${alpha}-bs${batch_size}-train.txt 2> ${log_path}/$today/${today}-${data}-AR${AR}-ood${ood}-seed${seed}-epoch${num_epoch}-ldim${latent_dim}-${model_name}-${atlas_count}atlas-alpha${alpha}-bs${batch_size}-train.err 

                # Error checking
                if grep -q "Traceback" ${log_path}/$today/${today}-${data}-AR${AR}-ood${ood}-seed${seed}-epoch${num_epoch}-ldim${latent_dim}-${model_name}-${atlas_count}atlas-alpha${alpha}-bs${batch_size}-train.err; then
                    echo "Error"
                fi

                if grep -q "main.py: error:" ${log_path}/$today/${today}-${data}-AR${AR}-ood${ood}-seed${seed}-epoch${num_epoch}-ldim${latent_dim}-${model_name}-${atlas_count}atlas-alpha${alpha}-bs${batch_size}-train.err; then
                    echo "Error"
                fi

                done
            done
        done
    done
done
