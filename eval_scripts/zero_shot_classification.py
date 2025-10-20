"""zero_shot_classification.py evaluates the performance of a pre-trained model
on an unseen dataset without fine-tuning."""
from zero_shot_model_evaluators import SCVIZeroShotEvaluator
from sklearn.model_selection import train_test_split
from collections import defaultdict
import anndata as ad
import pandas as pd
import scvi
import sys
import os   
import random
import numpy as np
import torch


default_n_threads = 64
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"


def get_classification_metrics_df(train_adata_path,
                                  model_path,
                                  adata,
                                  dataset_name,
                                  seed,
                                  cell_type_col,
                                  method,
                                  ARtype,
                                  latent_dim,
                                  Atlas_cell_count):
    metrics_dict = defaultdict(list)

    if method == "scVI":
        zero_shot_evaluator = SCVIZeroShotEvaluator(model_path, train_adata_path)
        
    set_seed(seed)
    train_indices, test_indices = train_test_split(
        list(range(adata.n_obs)), train_size=0.5, test_size=0.5, random_state=seed)
    train_adata = adata[train_indices, :]
    test_adata = adata[test_indices, :]

    classification_metrics = zero_shot_evaluator.evaluate_classification(
        train_adata, test_adata, cell_type_col)
    print("classification_metrics")
    print(classification_metrics)

    classification_metrics["seed"] = seed
    classification_metrics["dataset"] = dataset_name
    classification_metrics["ARtype"] = ARtype
    classification_metrics["latent_dim"] = latent_dim
    classification_metrics["Atlas_cell_count"] = Atlas_cell_count

    for key in classification_metrics:
        metrics_dict[key].append(classification_metrics[key])

    metrics_df = pd.DataFrame.from_dict(metrics_dict)

    return metrics_df


def prep_for_evaluation(adata, model_path):
    
    adata.var_names = adata.var_names.str.split(".").str[0]

    adata_copy = adata.copy()
    adata_copy.obs_names_make_unique()
    adata_copy.var_names_make_unique()

    scvi.model.SCVI.prepare_query_anndata(adata_copy, model_path)
    return


def main():
    method = sys.argv[1]
    model_path = sys.argv[2]
    seed = int(sys.argv[3])
    out_dir = sys.argv[4]
    ARtype = sys.argv[5]
    latent_dim = sys.argv[6]
    Atlas_cell_count = sys.argv[7]
    train_adata_path = sys.argv[8]


    print(sys.argv)
    
    test_adata_path = {
        'Heart': '../data/ajay/sctab_train_10pct_heart_nohematopoietic.h5ad',
        'Kidney': '../data/ajay/sctab_train_10pct_kidney_nohematopoietic.h5ad',
        'Neurons': '../data/ajay/Neurons_H1830002_10Ksubset.h5ad'
    }
    
    
    # loop through the test_adata_path dictionary
    for dataset_name, test_adata_path in test_adata_path.items():
        print('Loading dataset')
        
        test_adata = ad.read_h5ad(test_adata_path)
        new_test_adata = test_adata.copy()
        
        if dataset_name == 'Neurons':
            label_col = 'supercluster_term'
        else:
            label_col = 'cell_type'

        metrics_df = get_classification_metrics_df(
            train_adata_path,
            model_path,
            new_test_adata,
            dataset_name,
            seed,
            label_col,
            method,
            ARtype,
            latent_dim,
            Atlas_cell_count)

        dataset_name = os.path.basename(dataset_name)

        metrics_csv = f"zero_shot_classification_metrics_{method}_{dataset_name}_seed_{seed}_ARtype_{ARtype}_latent_dim_{latent_dim}_Atlas_cell_count_{Atlas_cell_count}.csv"

        out_dir = out_dir

        print(out_dir + "/" + metrics_csv)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        metrics_df.to_csv(out_dir + "/" + metrics_csv)

def set_seed(seed):
    """Set seed for reproducibility.
    
    Args:
        seed (int): seed for reproducibility"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return
if __name__ == "__main__":
    main()
