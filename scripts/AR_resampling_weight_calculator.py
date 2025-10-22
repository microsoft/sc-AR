import numpy as np


def calculate_resampling_weights(z, bins=100, smoothing_fac=0.001): 
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
    w = training_sample_p/np.sum(training_sample_p)

    return w