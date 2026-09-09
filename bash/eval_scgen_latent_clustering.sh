#!/bin/bash
# Run scGen latent-space clustering for Naive and AR across OOD datasets.
#
# For each held-out setting / seed, with --latent_clustering_all True:
#   1) Test cells  — KMeans vs perturbation condition (Control / Stimulated)
#   2) Train cells — KMeans vs cell-group labels
#      (pbmc: cell_type | species: species | lps-hpoly: cell_label)
#      Train set = all cells except held-out × stimulated (train+valid).
#   Separate GT and KMeans PCA-2 plots of the latent under:
#      figures/umap/{data}/seed{seed}/
#
# Metrics CSV (one file per model × data_type × GT label):
#   result/test/{data}/clustering/
#     {test_id}_{AR|Naive}_{train|test}_latent_clustering_{label}_seed{seed}.csv
#
# Usage (from repo root or this directory):
#   bash bash/eval_scgen_latent_clustering.sh
#   bash bash/eval_scgen_latent_clustering.sh --gpu
#
# Requires trained checkpoints under saved_models/{data}/seed{seed}/ whose
# filenames match create_id() in scripts/utils.py (same hyperparameters as
# bash/eval_scgen.sh / bash/train_scgen.sh).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

USE_GPU=False
if [[ "${1:-}" == "--gpu" ]]; then
  USE_GPU=True
fi

today=$(date '+%Y%m%d')
mkdir -p "log/${today}"

# activate environment
eval "$(conda shell.bash hook)"
source activate scar-env

# Avoid mixed OpenMP (libiomp + libomp) crashes during sklearn on macOS.
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

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
  echo "Dataset: ${data} | epochs: ${num_epoch} | seeds: ${seed_list[*]} | gpu=${USE_GPU}"
  echo "  tasks: test×condition + train×cell-group | models: ${model}"
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
      echo "gpu:           ${USE_GPU}"
      echo "latent_clustering_all: True"

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
        --gpu "${USE_GPU}" \
        --rebuttal False \
        --rebuttal2 False \
        >> "${log_base}.txt" 2> "${log_base}.err"

      if grep -q "Traceback" "${log_base}.err"; then
        echo "ERROR: Traceback in ${log_base}.err"
        tail -n 40 "${log_base}.err"
        exit 1
      fi

      if grep -q "main.py: error: " "${log_base}.err"; then
        echo "ERROR: argparse failure in ${log_base}.err"
        tail -n 40 "${log_base}.err"
        exit 1
      fi

      echo "OK: finished ${data} held-out=${test_data} seed=${seed}"
    done
  done
}

# Species (cell-group label: species)
run_dataset species mouse pig rabbit rat

# PBMC (cell-group label: cell_type)
run_dataset pbmc CD4T B NK CD14+Mono Dendritic CD8T FCGR3A+Mono

# LPS-Hpoly (cell-group label: cell_label)
run_dataset lps-hpoly Endocrine Enterocyte Enterocyte.Progenitor Goblet Stem TA TA.Early Tuft

echo ""
echo "Latent clustering evaluation completed."
echo "CSVs:  result/test/{data}/clustering/"
echo "Plots: bash/figures/umap/{data}/seed{seed}/ (*-pca2-color_{ground_truth,kmeans}*.png)"
