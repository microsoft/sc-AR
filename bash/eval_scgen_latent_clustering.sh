#!/bin/bash
# Run latent-space cell-type clustering (train + test) for scGen Naive and AR
# models across all standard OOD datasets, held-out groups, and seeds.
#
# Results are written to:
#   result/test/{data}/clustering/{test_id}_{model}_all_latent_clustering_{label_col}_seed{seed}.csv
#
# Usage:
#   bash eval_scgen_latent_clustering.sh
#
# Requires trained checkpoints under saved_models/{data}/seed{seed}/ whose
# filenames match create_id() in scripts/utils.py (same hyperparameters as eval_scgen.sh).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

today=$(date '+%Y%m%d')

if [ ! -d log ]; then
  mkdir -p log
fi

if [ ! -d "log/${today}" ]; then
  mkdir -p "log/${today}"
fi

# activate environment
eval "$(conda shell.bash hook)"
source activate scar-env

# ── paths ─────────────────────────────────────────────────────────────────────
root="../"

# ── scGen experiment settings (must match training / eval_scgen.sh) ───────────
# These values are encoded in checkpoint filenames via create_id().
model="AR,Naive"
model_name="scgen"
ood=True
variable_con=False
con_percent=1.0
latent_dim=64
batch_size=2048
lr=5e-05
weight_decay=5e-05
seed_list=(100 101 102 103 104)

run_dataset() {
  local data="$1"
  shift
  local -a list=("$@")
  local num_epoch

  case "${data}" in
    pbmc) num_epoch=1800 ;;
    lps-hpoly) num_epoch=2000 ;;
    species) num_epoch=1000 ;;
    *)
      echo "Unknown dataset: ${data}"
      exit 1
      ;;
  esac

  echo "============================================================"
  echo "Dataset: ${data} | epochs: ${num_epoch} | seeds: ${seed_list[*]}"
  echo "============================================================"

  for i in "${!list[@]}"; do
    train_data=$(printf '%s\n' "${list[@]:0:$i}" "${list[@]:$((i + 1))}" | paste -sd ',' -)
    test_data=${list[$i]}

    for seed in "${seed_list[@]}"; do
      log_base="log/${today}/${today}-latent-clustering-${data}-test-${test_data}-${model}-OOD${ood}-epoch${num_epoch}-seed${seed}"

      echo ""
      echo "data:          ${data}"
      echo "train_data:    ${train_data}"
      echo "test_data:     ${test_data}"
      echo "model:         ${model}"
      echo "ood:           ${ood}"
      echo "variable_con:  ${variable_con}"
      echo "con_percent:   ${con_percent}"
      echo "num_epoch:     ${num_epoch}"
      echo "latent_dim:    ${latent_dim}"
      echo "batch_size:    ${batch_size}"
      echo "lr:            ${lr}"
      echo "weight_decay:  ${weight_decay}"
      echo "seed:          ${seed}"

      python ../main.py \
        --root "${root}" \
        --data "${data}" \
        --train_data "${train_data}" \
        --test_data "${test_data}" \
        --test True \
        --latent_clustering_all True \
        --model "${model}" \
        --model_name "${model_name}" \
        --ood "${ood}" \
        --variable_con "${variable_con}" \
        --con_percent "${con_percent}" \
        --num_epoch "${num_epoch}" \
        --latent_dim "${latent_dim}" \
        --batch_size "${batch_size}" \
        --lr "${lr}" \
        --weight_decay "${weight_decay}" \
        --seed "${seed}" \
        --rebuttal False \
        --rebuttal2 False \
        >> "${log_base}.txt" 2> "${log_base}.err"

      if grep -q "Traceback" "${log_base}.err"; then
        echo "Error: see ${log_base}.err"
      fi

      if grep -q "main.py: error: " "${log_base}.err"; then
        echo "Error: see ${log_base}.err"
      fi
    done
  done
}

# Species
run_dataset species mouse pig rabbit rat

# PBMC
run_dataset pbmc CD4T B NK CD14+Mono Dendritic CD8T FCGR3A+Mono

# LPS-Hpoly
run_dataset lps-hpoly Endocrine Enterocyte Enterocyte.Progenitor Goblet Stem TA TA.Early Tuft

echo ""
echo "Latent clustering evaluation completed."
