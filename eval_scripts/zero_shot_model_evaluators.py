from evaluation_utils import eval_classification_metrics
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import scvi
import random
import os
import torch


def set_seed(seed):
    """Set seed for reproducibility.
    
    Args:
        seed (int): seed for reproducibility"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    

class ModelEmbeddingsMixin:
    def get_embeddings(self):
        raise NotImplementedError


class ZeroShotEvaluator(ModelEmbeddingsMixin):

    def evaluate_classification(self, adata_train, adata_test, cell_type_col, n_neighbors=5):
        
        print('getting embeddings')
        set_seed(42)
        train_latent_embeddings = self.get_embeddings(adata_train)
        np.random.seed(42)
        test_latent_embeddings = self.get_embeddings(adata_test)

        print('KNN CLASSIFICATION')
        set_seed(42)
        knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)

        print("Checking for NaNs in train data:", np.isnan(train_latent_embeddings).sum())
        print("Checking for NaNs in test data:", np.isnan(test_latent_embeddings).sum())
        print(f"Checking for NaNs in labels: {adata_train.obs[cell_type_col].isna().sum()}")

        set_seed(42)
        knn_model.fit(train_latent_embeddings, adata_train.obs[cell_type_col])

        print('predict')
        test_predictions = knn_model.predict(test_latent_embeddings)

        print('classification metrics')
        classification_metrics = eval_classification_metrics(
            adata_test.obs[cell_type_col], test_predictions)

        print('done')
        return classification_metrics


class SCVIZeroShotEvaluator(ZeroShotEvaluator):
    def __init__(self, model_path, train_adata_path):
        
        train_adata = scvi.data.read_h5ad(train_adata_path)
        self.model = scvi.model.SCVI.load(model_path, train_adata)

    def get_embeddings(self, adata):
        adata.var_names = adata.var_names.str.split(".").str[0]
        
        adata.obs_names_make_unique()
        adata.var_names_make_unique()
        np.random.seed(42)
        scvi.model.SCVI.prepare_query_anndata(adata, self.model)
        np.random.seed(42)
        latent_embeddings = self.model.get_latent_representation(adata)
        return latent_embeddings
