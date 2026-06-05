#!/usr/bin/env bash

set -u

if [ "$#" -lt 15 ]; then
  echo "Usage: $0 <data> <model> <num_epoch> <variable_con> <con_percent> <in_dist_group> <seed> <tracked_epoch> <model_name> <data_path> <out_path> <latent_dim> <root> <atlas_count> <weight_diagnostics_path>"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

data=$1
model=$2
num_epoch=$3
variable_con=$4
con_percent=$5
in_dist_group=$6
seed=$7
tracked_epoch=$8
model_name=$9
data_path=${10}
out_path=${11}
latent_dim=${12}
root=${13}
atlas_count=${14}
weight_diagnostics_path=${15}

## Sample command
# ./store_weight_diagnostics_scvi.sh \
#     sctab \
#     AR \
#     300 \
#     False \
#     1.0 \
#     "" \
#     42 \
#     0 \
#     scvi \
#     ../data/sctab/ \
#     ../saved_models/ \
#     64 \
#     ../ \
#     10000 \
#     ./weight_diagnostics/sctab/42/scvi/weight_diagnostics_sctab_AR_epoch300_seed42_scvi_ldim64_atlascount10000_trackedepoch0.csv

# print all input data
echo data: ${data}
echo data_path: ${data_path}
echo model: ${model}
echo num_epoch: ${num_epoch}
echo variable_con: ${variable_con}
echo con_percent: ${con_percent}
echo in_dist_group: ${in_dist_group}
echo seed: ${seed}
echo tracked_epoch: ${tracked_epoch}
echo model_name: ${model_name}
echo out_path: ${out_path}
echo latent_dim: ${latent_dim}
echo root: ${root}
echo atlas_count: ${atlas_count}
echo weight_diagnostics_path: ${weight_diagnostics_path}


eval "$(conda shell.bash hook)"
source activate scar-env

python "${SCRIPT_DIR}/../main.py" \
    --data "${data}" \
    --data_path "${data_path}" \
    --out_path "${out_path}" \
    --latent_dim "${latent_dim}" \
    --train False \
    --test True \
    --predict False \
    --check_scvi False \
    --model "${model}" \
    --num_epoch "${num_epoch}" \
    --variable_con "${variable_con}" \
    --con_percent "${con_percent}" \
    --in_dist_group "${in_dist_group}" \
    --seed "${seed}" \
    --plot_umap_annotated_with_w False \
    --store_resampling_weights False \
    --store_weight_diagnostics True \
    --weight_diagnostics_path "${weight_diagnostics_path}" \
    --tracked_epoch "${tracked_epoch}" \
    --degs_extraction_based_on_resampling_w False \
    --model_name "${model_name}" \
    --root "${root}" \
    --atlas_count "${atlas_count}" \
    --batch_size 4096

