#!/bin/bash

# Assign input parameters to variables
data=$1            # data base name, e.g., sctab
data_path=$2       # path to the /dataset/ folder
out_path=$3        # path to the folder for saving trained models (ends with /saved_models/)
seed=$4            # seed number
log_path=$5        # path to save logs
root=$6            # root folder of the project
model_name=$7      # model type: scvi or scgen
latent_dim=$8      # latent dimension
num_epoch=$9       # number of epochs
AR=${10}           # AR flag: True or False
atlas_count=${11}  # number of atlas cells, e.g., 0, 1, 10, 100, 1000, 10000, 50000


# activate the conda environment
eval "$(conda shell.bash hook)"
source activate scAR-env
export WANDB_KEY=''

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

# print the parameters
echo "data: $data"
echo "data_path: $data_path"
echo "out_path: $out_path"
echo "seed:" $seed
echo "log_path:" $log_path
echo "root:" $root
echo "model_name:" $model_name
echo "latent_dim:" $latent_dim
echo "num_epoch:" $num_epoch
echo "AR:" $AR
echo "atlas_count:" $atlas_count
echo "ood:" $ood
echo ""

python -u ${root}/main.py \
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
    --atlas_count ${atlas_count} >> ${log_path}/$today/${today}-${data}-AR${AR}-ood${ood}-seed${seed}-epoch${num_epoch}-${model_name}-${atlas_count}atlas-train.txt 2> ${log_path}/$today/${today}-${data}-AR${AR}-ood${ood}-seed${seed}-epoch${num_epoch}-${model_name}-${atlas_count}atlas-train.err 

# Error checking
if grep -q "Traceback" ${log_path}/$today/${today}-${data}-AR${AR}-ood${ood}-seed${seed}-epoch${num_epoch}-${model_name}-${atlas_count}atlas-train.err; then
    echo "Error"
fi

if grep -q "main.py: error:" ${log_path}/$today/${today}-${data}-AR${AR}-ood${ood}-seed${seed}-epoch${num_epoch}-${model_name}-${atlas_count}atlas-train.err; then
    echo "Error"
fi