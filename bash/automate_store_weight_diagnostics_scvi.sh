#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

today=$(date '+%Y%m%d')

# get the data, data_path, and seed from input
data=$1 # should be data name, sctab
data_path=$2 # should be path to the data/sctab/ folder
out_path=$3 # should be path to the folder for saving the trained models ending with /saved_models/
log_path=$4
root=$5 # path to the root folder of the project sc-uncertainty
weight_diagnostics_path=$6 # path to the folder for saving weight diagnostics ending with /weight_diagnostics/

if [ ! -d "${log_path}" ]; then
  mkdir -p "${log_path}";
fi

if [ ! -d "${log_path}/$today" ]; then
  mkdir -p "${log_path}/$today";
fi

## Common Parameters ##
num_epoch=300
model='AR'
variable_con=False
con_percent=1.0
in_dist_group=''
ood=True
model_name='scvi'
latent_dim=64

# Define seeds to iterate over
seeds=(42 43 44 45 46)

# Define atlas counts to iterate over
atlas_counts=(0 1 10 100 1000 10000 50000)

# Define tracked_epoch as every 20 epochs up to 280 and 'best'
tracked_epoch=(0 20 40 60 80 100 120 140 160 180 200 220 240 260 280 best)

for s in "${seeds[@]}"; do
  for atlas_count in "${atlas_counts[@]}"; do
    for t in "${tracked_epoch[@]}"; do

      weight_diagnostics_path_current="${weight_diagnostics_path}/$data/$s/$model_name/weight_diagnostics_${data}_${model}_epoch${num_epoch}_seed${s}_${model_name}_ldim${latent_dim}_atlascount${atlas_count}_trackedepoch${t}.csv"
      # check if ${weight_diagnostics_path}/$data/$s/$model_name/ exists, if not, create it
      if [ ! -d "${weight_diagnostics_path}/$data/$s/$model_name/" ]; then
        mkdir -p "${weight_diagnostics_path}/$data/$s/$model_name/"
      fi

      echo "data: ${data}"
      echo "model: ${model}"
      echo "num_epoch: ${num_epoch}"
      echo "variable_con: ${variable_con}"
      echo "con_percent: ${con_percent}"
      echo "in_dist_group: ${in_dist_group}"
      echo "seed: ${s}"
      echo "tracked_epoch: ${t}"
      echo "model_name: ${model_name}"
      echo "data_path: ${data_path}"
      echo "out_path: ${out_path}"
      echo "latent_dim: ${latent_dim}"
      echo "root: ${root}"
      echo "atlas_count: ${atlas_count}"
      echo "weight_diagnostics_path: ${weight_diagnostics_path_current}"

      log_file="${log_path}/$today/${today}-standard-${data}-${model}-epoch${num_epoch}-seed${s}-trackedepoch${t}-ldim${latent_dim}-atlascount${atlas_count}-${model_name}-weight-diagnostics.txt"
      err_file="${log_path}/$today/${today}-standard-${data}-${model}-epoch${num_epoch}-seed${s}-trackedepoch${t}-ldim${latent_dim}-atlascount${atlas_count}-${model_name}-weight-diagnostics.err"

      "${SCRIPT_DIR}/store_weight_diagnostics_scvi.sh" \
        "${data}" \
        "${model}" \
        "${num_epoch}" \
        "${variable_con}" \
        "${con_percent}" \
        "${in_dist_group}" \
        "${s}" \
        "${t}" \
        "${model_name}" \
        "${data_path}" \
        "${out_path}" \
        "${latent_dim}" \
        "${root}" \
        "${atlas_count}" \
        "${weight_diagnostics_path_current}" >> "${log_file}" 2> "${err_file}"

      if grep -q "Traceback" "${err_file}"; then
        echo "Error"
      fi

      if grep -q "main.py: error: " "${err_file}"; then
        echo "Error"
      fi

    done
  done
done
