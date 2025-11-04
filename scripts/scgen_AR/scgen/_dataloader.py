from collections import Counter
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.stats import zscore
import plotly.express as px
from scripts.utils import set_seed
from umap import UMAP
import seaborn as sns
import scipy.stats
import pandas as pd
import numpy as np
import random
import torch
import math
import os
from pathlib import Path

path = Path(__file__).parents[2]
root = os.path.abspath(os.path.join(path, os.pardir))


class Dataset(torch.utils.data.Dataset):
    """Dataloder.

    Args_
        torch (_type_): _description_

    """

    def __init__(self, adata = [], groups = []):
        """_summary.

        Args:
            group (str, optional): _description_. Defaults to 'train'.
            cell_type (str, optional): _description_. Defaults to 'CD4T'.

        """
        self.data = torch.tensor(adata.X.toarray())
        self.resample_data = self.data
        self.w = np.zeros(shape=(self.data.shape[0], 1))
        self.groups = groups


    def update_training_sample_probabilities(self, z, bins=10, smoothing_fac=0.001): 
        """Updates the sampling probabilities for cells within a batch based on the maximum of
        latent variables' probablities.
        source: https://goodboychan.github.io/python/tensorflow/mit/2021/02/27/Debiasing.html
        
        Args:
            z (torch.Tensor): Latent variables.
            bins (int, optional): Number of bins to use for histogram. Defaults to 10.
            smoothing_fac (float, optional): Smoothing factor for histogram. Defaults to 0.001."""

        training_sample_p = np.zeros(z.shape[0])
        
        # consider the distribution for each latent variable 
        for i in range(z.shape[1]):

            latent_distribution = z[:,i]
            # generate a histogram of the latent distribution
            hist_density, bin_edges =  np.histogram(latent_distribution, density=True, bins=bins)

            # find which latent bin every data sample falls in 
            bin_edges[0] = -float('inf')
            bin_edges[-1] = float('inf')

            # call the digitize function to find which bins in the latent distribution 
            #    every data sample falls in to
            # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.digitize.html
            bin_idx = np.digitize(latent_distribution, bin_edges) 

            # smooth the density function
            hist_smoothed_density = hist_density + smoothing_fac
            hist_smoothed_density = hist_smoothed_density / np.sum(hist_smoothed_density)

            # invert the density function 
            p = 1.0/(hist_smoothed_density[bin_idx-1])

            # normalize all probabilities
            p = p / np.sum(p)

            # update sampling probabilities by considering whether the newly
            # computed p is greater than the existing sampling probabilities.
            training_sample_p = np.maximum(p, training_sample_p)

        # final normalization
        self.w = training_sample_p/np.sum(training_sample_p)

        return


    def resample(self, batch_size, args):
        """Resample data for one step/batch.
        
        Args:
            batch_size (int): Size of the resampling"""
            
        set_seed(args.seed)

        batch_dict = {}
        idx = random.choices(
            range(self.data.shape[0]),
            weights=self.w,
            k=batch_size)
            
        batch_dict['X'] = self.data[idx]

        return batch_dict
    
    
    def resample_all_batches(self):
        """Resample all batches at once."""

        idx = random.choices(
            range(self.data.shape[0]),
            weights=self.w,
            k=self.data.shape[0])

        assert len(idx) == self.data.shape[0]
            
        self.data = self.data[idx]

        return


    def __len__(self):
        """Denotes the total number of samples.

        Returns:
            int: Shape of the sample data passed to the class.

        """
        return self.data.shape[0]


    def __getitem__(self, idx):
        """Generate one sample of data.

        Args:
            index (int): index of the item to be returned.

        Returns:
            torch.Tensor: Tensor of the resampled data.

        """
        data_dict = {}
        data_dict['X'] = self.data[idx]
        return data_dict