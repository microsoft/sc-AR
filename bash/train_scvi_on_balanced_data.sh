#!/bin/bash

# Assign input parameters to variables
data=$1            # data: sctab
data_path=$2       # path to the ./data/ folder
out_path=$3        # path to the folder for saving trained models (ends with /saved_models/)
seed=$4            # seed number: 42 43 44 45 46
log_path=$5        # path to save logs ./log/
root=$6            # root folder of the project ../
model_name=$7      # model type: scvi
latent_dim=$8      # latent dimension: 64
num_epoch=$9       # number of epochs: 300
AR=${10}           # AR flag: True or False
atlas_count=${11}  # number of atlas cells, e.g., 0, 1, 10, 100, 1000, 10000, 50000
alpha=${12}        # alpha value for smoothing: 0.0001
balancing_method=${13} # balancing method
train_adata_path=${14} # path to train adata: $data_path/sctab/balanced_data/$balancing_method/$seed/bloodbase_${atlas_count}atlas_seed${seed}_2000HVGs_balancing_method${balancing_method}_train_adata.h5ad
valid_adata_path=${15} # path to valid adata: $data_path/sctab/balanced_data/$balancing_method/$seed/bloodbase_${atlas_count}atlas_seed${seed}_2000HVGs_balancing_method${balancing_method}_valid_adata.h5ad


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
batch_size=4096

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
echo "alpha:" $alpha
echo "balancing_method:" $balancing_method
echo "train_adata_path:" $train_adata_path
echo "valid_adata_path:" $valid_adata_path
echo ""

python -u ${root}/main.py \
    --root ${root} \
    --data ${data} \
    --data_path ${data_path} \
    --out_path ${out_path} \
    --train True \
    --balanced_data True \
    --balancing_method ${balancing_method} \
    --train_adata_path ${train_adata_path} \
    --valid_adata_path ${valid_adata_path} \
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
    --alpha ${alpha} >> ${log_path}/$today/${today}-${data}-AR${AR}-seed${seed}-${atlas_count}atlas-alpha${alpha}--balancing_method${balancing_method}-train.txt 2> ${log_path}/$today/${today}-${data}-AR${AR}-seed${seed}-${atlas_count}atlas-alpha${alpha}--balancing_method${balancing_method}-train.err 

# Error checking
if grep -q "Traceback" ${log_path}/$today/${today}-${data}-AR${AR}-seed${seed}-${atlas_count}atlas-alpha${alpha}--balancing_method${balancing_method}-train.err; then
    echo "Error"
fi