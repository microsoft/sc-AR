
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
import numpy as np
import random
import torch
import scvi


def set_seed(seed):
    """Set seed for reproducibility.
    
    Args:
        seed (int): seed for reproducibility"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return


class LDVAE_eval:
    def __init__(self, model_path, adata):
        """
        Initialize the LDVAE_eval class with a trained model.
        
        :param model_path: Path to the trained scVI model
        """
        set_seed(42)
        self.model = scvi.model.SCVI.load(model_path, adata)
        self.evals = {}

    def get_reconstruction_r2(self, adata, desired_obs_fields, eval_name, num_hvg = 'All'):
        adata.var_names = adata.var_names.str.split(".").str[0]
        
        print(adata)
        adata_copy = adata.copy()
        adata_copy.obs_names_make_unique()
        adata_copy.var_names_make_unique()

        print(num_hvg)
        set_seed(42)
        scvi.model.SCVI.prepare_query_anndata(adata_copy, self.model)
        print('adata_copy:', adata_copy)

        #get the nUMI and add to obs
        adata_copy.obs['nUMI'] = adata_copy.X.sum(axis = 1)

        #make a layer with the cp10k, log1p normalized counts
        adata_copy.layers['log1p_cp10k'] = adata_copy.X.copy()
        #print(adata_copy.layers['log1p_cp10k'][0,0])
        sc.pp.normalize_total(adata_copy, target_sum=1e4, layer = 'log1p_cp10k')
        #print(adata_copy.layers['log1p_cp10k'][0,0])
        sc.pp.log1p(adata_copy, layer = 'log1p_cp10k')
        #print(adata_copy.layers['log1p_cp10k'][0,0])

        #make a layer with the reconstructed gene expression
        np.random.seed(42)
        adata_copy.layers['reconstructed'] = np.log1p(self.model.get_normalized_expression(adata_copy, return_numpy = True, library_size = 1e4))

        #get the expression matrices
        expression_original = adata_copy.layers['log1p_cp10k'].copy().toarray()
        expression_reconstructed = adata_copy.layers['reconstructed'].copy()

        # Compute the means of the rows
        mean1 = np.mean(expression_original, axis=1)
        mean2 = np.mean(expression_reconstructed, axis=1)
        
        # Center the arrays by subtracting the mean of each row
        array1_centered = expression_original - mean1[:, np.newaxis]
        array2_centered = expression_reconstructed - mean2[:, np.newaxis]
        
        # Compute the numerator (covariance)
        numerator = np.sum(array1_centered * array2_centered, axis=1)
        
        # Compute the denominator (product of standard deviations)
        std1 = np.sqrt(np.sum(array1_centered ** 2, axis=1))
        std2 = np.sqrt(np.sum(array2_centered ** 2, axis=1))
        denominator = std1 * std2
        
        # Compute the correlation
        correlation = numerator / denominator
        
        #create a dataframe with desired obs fields and the correlation
        df = pd.DataFrame(correlation, columns = ['correlation'])
        #add the r^2
        df['r2'] = correlation**2
        for field in desired_obs_fields:
            df[field] = adata_copy.obs[field].values

        self.evals[eval_name] = df

        return correlation
