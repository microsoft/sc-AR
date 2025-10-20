import logging
import scanpy as sc
import pandas as pd
import numpy as np
import scvi
import anndata as ad
import os
import pickle
import sys
import torch
from pathlib import Path

from LDVAE_eval_class import LDVAE_eval


final_eval_path = sys.argv[1]
model_adata = sys.argv[2]
model_path = sys.argv[3]
eval_output_dir = sys.argv[4]
seed = int(sys.argv[5])
ARtype = sys.argv[6]
latent_dim = sys.argv[7]
Atlas_cell_count = sys.argv[8]

model_adata = sc.read_h5ad(model_adata)
# Load datasets
sctab_train_10pct_heart_nohematopoietic = sc.read(os.path.join(final_eval_path, "sctab_train_10pct_heart_nohematopoietic.h5ad"))
sctab_train_10pct_kidney_nohematopoietic = sc.read(os.path.join(final_eval_path, "sctab_train_10pct_kidney_nohematopoietic.h5ad"))
Neurons_H1830002_10Ksubset = sc.read(os.path.join(final_eval_path, "Neurons_H1830002_10Ksubset.h5ad"))

# Initialize LDVAE_eval object for evaluation
ldvae_eval = LDVAE_eval(model_path, model_adata)


# Run evaluations using LDVAE_eval (NOT scvi_model)
print(f"Running evaluation for Model {model_path}")

print("Reconstruction Evaluation")
print("scTab Heart")
ldvae_eval.get_reconstruction_r2(sctab_train_10pct_heart_nohematopoietic, ['cell_type'], 'scTab_Heart_reconstruction', 'All')

print("scTab Kidney")
ldvae_eval.get_reconstruction_r2(sctab_train_10pct_kidney_nohematopoietic, ['cell_type'], 'scTab_Kidney_reconstruction', 'All')

print("Neurons")
ldvae_eval.get_reconstruction_r2(Neurons_H1830002_10Ksubset, ['supercluster_term'], 'Neurons_reconstruction', 'All')

# Save results
output_file = os.path.join(eval_output_dir, f'Reconstruction_seed_{seed}_ARtype_{ARtype}_latent_dim_{latent_dim}_Atlas_cell_count_{Atlas_cell_count}_bloodbase_eval.pkl')
with open(output_file, 'wb') as f:
    pickle.dump(ldvae_eval.evals, f)

print(f"Evaluation results saved to {output_file}")
