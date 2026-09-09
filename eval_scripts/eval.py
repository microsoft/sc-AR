from scripts.dataloader import adjust_training_proportions
from scripts.scgen_AR.scgen._scgen import SCGEN
from scripts.utils import set_seed, create_id
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
import matplotlib.colors as mcolors
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from numpy.linalg import norm
from scipy.stats import ks_2samp
from scipy import sparse
import seaborn as sns
import pandas as pd
import scanpy as sc
import numpy as np
from umap import UMAP
import random
import scvi
import torch
import pca
import os
from scipy.stats import ttest_ind


test_metric = pd.DataFrame(
    columns=['experiment_id', 'test', 'AR', 'variable_con',
             'con_percent','in_dist_group', 'metric',
             'value', 'seed'])
path = os.getcwd()
root = os.path.abspath(os.path.join(path, os.pardir))+'/'


def _repo_root(args):
    """Resolve repository root from args.root (preferred) or caller cwd."""
    if getattr(args, 'root', None) and args.root not in ('.', ''):
        return args.root.rstrip('/') + '/'
    return root


def get_train_test_adata(args, adata):
    """Get the train and test data.
    
    Args:
        args (argparse.Namespace): Arguments passed to the script.
        adata (anndata.AnnData): Anndata object containing the data.
        
    Returns:
        train_adata (anndata.AnnData): Anndata object containing the train data.
        test_adata (anndata.AnnData): Anndata object containing the test data.
        unper_test_adata (anndata.AnnData): Anndata object containing the control test data.
        per_test_adata (anndata.AnnData): Anndata object containing the stimulated test data."""

    if args.variable_con:
        adjusted_adata = adjust_training_proportions(args, adata)
        train_valid_adata = adjusted_adata[~(
            (adjusted_adata.obs[args.adata_label_cell].isin(args.test_data)) &
            (adjusted_adata.obs.condition == args.adata_label_per))]
    else:
        train_valid_adata = adata[~(
            (adata.obs[args.adata_label_cell].isin(args.test_data)) &
            (adata.obs.condition == args.adata_label_per))]
    
    set_seed(args.seed)
    valid_index = random.sample(range(train_valid_adata.shape[0]),
                                int(train_valid_adata.shape[0]/5))
    train_index = \
        [x for x in range(train_valid_adata.shape[0])
         if x not in valid_index]

    train_adata = train_valid_adata[train_index]
    valid_adata = train_valid_adata[valid_index]
    assert train_adata.shape[0] + valid_adata.shape[0] == \
        train_valid_adata.shape[0]

    if args.in_dist_group == '':
        unper_test_adata = adata[(
            (adata.obs[args.adata_label_cell].isin(args.test_data)) &
            (adata.obs.condition == args.adata_label_unper))]
        per_test_adata = adata[(
            (adata.obs[args.adata_label_cell].isin(args.test_data)) &
            (adata.obs.condition == args.adata_label_per))]
        
    elif args.in_dist_group != '':
        print('in_dist_group: '+args.in_dist_group)

        in_dist_adata = \
            adata[adata.obs[args.adata_label_cell] == args.in_dist_group]
            
        unper_test_adata = in_dist_adata[(
            in_dist_adata.obs.condition == args.adata_label_unper)]
        per_test_adata = in_dist_adata[(
            in_dist_adata.obs.condition == args.adata_label_per)]

    test_adata = per_test_adata.concatenate(unper_test_adata)
    return train_adata, test_adata, unper_test_adata, per_test_adata


def _get_train_valid_adata(args, adata):
    """All cells except held-out × stimulated (train + validation combined)."""
    if args.variable_con:
        adjusted_adata = adjust_training_proportions(args, adata)
        return adjusted_adata[~(
            (adjusted_adata.obs[args.adata_label_cell].isin(args.test_data)) &
            (adjusted_adata.obs.condition == args.adata_label_per))].copy()
    return adata[~(
        (adata.obs[args.adata_label_cell].isin(args.test_data)) &
        (adata.obs.condition == args.adata_label_per))].copy()


def load_eval_adata(args, adata, model_name='Naive'):
    """Predict the perturbed gene expression values for the given test cell group.
    
    Args:
        args (argparse.Namespace): Arguments passed to the script.
        adata (anndata.AnnData): Anndata object containing the data.
        model (torch.nn.Module, optional): The model to use for prediction.
        model_name (str, optional): Name of the model.
        
    Returns:
        anndata.AnnData: Anndata object containing contro, real stimulated and predicted data.
        SCGEN: SCGEN object containing the trained model."""

    train_adata, test_adata, unper_test_adata, per_test_adata = \
        get_train_test_adata(args, adata)
        
    model_path = _repo_root(args)+"saved_models/"+args.data+"/seed"+str(args.seed)+'/'+args.id+"-best.pt"
    if args.plot_umap_annotated_with_w:
        if args.tracked_epoch != 'best':
            model_path = _repo_root(args)+"saved_models/"+args.data+"/seed"+str(args.seed)+'/'+args.id+"-epoch"+str(args.tracked_epoch)+".pt"
        print('loading model from: '+model_path)
    model = SCGEN.load(
        model_path,
        train_adata.copy(),
        accelerator="gpu" if args.gpu else "cpu",
        device='auto')
    model.is_trained = True
    
    if args.plot_umap_annotated_with_w | args.degs_extraction_based_on_resampling_w:
        print('successfully loaded the model in: '+model_path)
        return model

    if getattr(args, 'latent_clustering_all', False):
        print('successfully loaded the model in: '+model_path)
        return model

    pred_adata_file = ''
    if args.in_dist_group == '':
        pred_adata_file = root+'prediction/'+args.data+'/seed'+str(args.seed)+ \
                         '/'+args.id+'-pred-adata.h5ad'
    else:
        pred_adata_file = root+'prediction/'+args.data+'/seed'+str(args.seed)+ \
                         '/'+args.id+'-pred-adata-'+args.in_dist_group+'.h5ad'

    pred_adata = sc.read(pred_adata_file)
    assert pred_adata.obs['condition'].unique()[0] == model_name
    assert pred_adata.shape == unper_test_adata.shape
    eval_adata = pred_adata.concatenate(per_test_adata, unper_test_adata)

    return eval_adata, model, pred_adata


def evaluate_correlation_for_DEG(
    args, DEG_idx_dic, per_mean, pred_mean):
    """Calculate correlation between ground truth and predicted values for DEGs.
    
    Args:
        eval_adata (:obj:`anndata.AnnData`): Anndata object containing the data.
        model_name (:obj:`str`): Name of the model.
        args (:obj:`argparse.Namespace`): Arguments passed to the script.
        DEG_counts (:obj:`list`): List of number of DEGs to consider.
        per_mean (:obj:`np.ndarray`): Ground truth values.
        pred_mean (:obj:`np.ndarray`): Predicted values.
        corr_mean (:obj:`float`): Correlation between ground truth and predicted values."""

    global test_metric
    
    # loop over keys and values of DEG_idx_dic 
    for DEG_count, DEG_index in DEG_idx_dic.items():

        DEG_per_mean = per_mean[DEG_index]
        DEG_pred_mean = pred_mean[DEG_index]

        DEG_corr, p_value = pearsonr(
            DEG_per_mean,
            DEG_pred_mean)
        DEG_mse = F.mse_loss(
            torch.from_numpy(DEG_per_mean),
            torch.from_numpy(DEG_pred_mean))

        test_metric.loc[test_metric.shape[0]] = \
            [args.test_id, args.test_data[0], args.AR, args.variable_con, args.con_percent,
             args.in_dist_group, str(DEG_count)+' DEG R2', round(DEG_corr**2, 6),
             args.seed]
        test_metric.loc[test_metric.shape[0]] = \
            [args.test_id, args.test_data[0], args.AR, args.variable_con, args.con_percent,
             args.in_dist_group, str(DEG_count)+' DEG MSE', round(DEG_mse.item(), 6),
             args.seed]
            
    return


def evaluate_correlation(eval_adata_dic, args, DEG_idx_dic):
    """Calculate correlation between ground truth and predicted values.
    
    Args:
        eval_adata_dic (dict): Dictionary containing the eval_adata for each model.
        args (argparse.Namespace): Arguments passed to the script."""

    global test_metric

    for model_name, eval_adata in eval_adata_dic.items():
        args.AR = False if model_name == 'Naive' else True
        create_id(args)

        pred_adata = eval_adata[eval_adata.obs.condition == model_name]
        per_adata = eval_adata[eval_adata.obs.condition == args.adata_label_per]

        # All genes correlation/MSE of means
        per_mean = np.mean(per_adata.X.toarray().astype(float), axis=0)
        pred_mean = np.mean(pred_adata.X.toarray().astype(float), axis=0)
        assert per_mean.shape[0] == pred_mean.shape[0] == eval_adata.shape[1]

        corr, p_value = pearsonr(per_mean, pred_mean)
        mse = F.mse_loss(
            torch.from_numpy(per_mean),
            torch.from_numpy(pred_mean))

        test_metric.loc[test_metric.shape[0]] = \
            [args.test_id, args.test_data[0], args.AR, args.variable_con, args.con_percent,
             args.in_dist_group, 'R2', round(corr**2, 4), args.seed]
        test_metric.loc[test_metric.shape[0]] = \
            [args.test_id, args.test_data[0], args.AR, args.variable_con, args.con_percent,
             args.in_dist_group, 'MSE', round(mse.item(), 6), args.seed]

        evaluate_correlation_for_DEG(
            args, DEG_idx_dic, per_mean, pred_mean)
            
    return


def update_training_sample_probabilities(z, bins=10, smoothing_fac=0.001): 
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
    w = training_sample_p/np.sum(training_sample_p)

    return w


def extract_epoch_from_ckpt(args):
    """Extract the epoch from the checkpoint file.
    
    Args:
        args (argparse.Namespace): Arguments passed to the script.
        
    Returns:
        int: Epoch number."""
    parts = args.ckpt.split('-epoch')
    for part in parts:
        if '.pt' in part:
            epoch = int(part.split('.pt')[0])
            return epoch
    return


def extract_DEGs_based_on_resampling_weights(adata, args, model):
    """Extract the differentially expressed genes based on the resampling weights
    between cells with high resampling weights and low resampling weights 
    (across all cell groups).
    
    Args:
        adata (anndata.AnnData): Anndata object containing the data.
        args (argparse.Namespace): Arguments passed to the script.
        model (torch.nn.Module): Model used for prediction."""
        
    # create a .csv file to store the DEGs
    print('entered extract_DEGs_based_on_resampling_weights')
    if not os.path.isdir(root+'result/test/DEGs_based_on_resampling_weights/'+args.data+'/seed'+str(args.seed)+'/'):
        os.makedirs(root+'result/test/DEGs_based_on_resampling_weights/'+args.data+'/seed'+str(args.seed)+'/')
        
    f = root+'result/test/DEGs_based_on_resampling_weights/'+args.data+'/seed'+str(args.seed)+'/'+ \
        args.data+'-test'+args.test_data[0]+'-AR'+'-variablecon'+str(args.variable_con)+'-conpercent'+ \
        str(args.con_percent)+'-DEGs.csv'

    deg_df = pd.DataFrame(
        columns=['percentile', 'DEG_count']+['DEG-'+str(i) for i in range(DEG_len)])
    
    DEG_len = 50
    percentile_list = [50, 60, 70, 80, 90]
        
    # calculate the resampling weights
    with torch.no_grad():
        
        latent_X = model.get_latent_representation(adata)
        # calculate the resampling weights
        w = update_training_sample_probabilities(latent_X)
        # normalize the weights
        w = (w - np.min(w)) / (np.max(w) - np.min(w))
        adata.obs['normalized_resampling_weight'] = w
    
    for percentile in percentile_list:
        
        # divide the cells into two groups based on the resampling weights
        high_resampling_weight_cells = adata[adata.obs['normalized_resampling_weight'] >= \
            np.percentile(adata.obs['normalized_resampling_weight'], percentile)]
        low_resampling_weight_cells = adata[adata.obs['normalized_resampling_weight'] < \
            np.percentile(adata.obs['normalized_resampling_weight'], percentile)]
        
        # add a new column to the adata.obs to indicate the resampling weight group
        adata.obs['resampling_weight_group'] = 'Low Resampling Weight'
        adata.obs.loc[high_resampling_weight_cells.obs.index, 'resampling_weight_group'] = \
            'High Resampling Weight'

        # extract the DEGs based on the resampling weight group
        diff_genes = []
        sc.tl.rank_genes_groups(adata,
                                groupby="resampling_weight_group",
                                method="wilcoxon")
        diff_genes = \
            adata.uns["rank_genes_groups"]["names"][
                'High Resampling Weight'][:DEG_len].tolist()

        deg_df.loc[deg_df.shape[0]] = [percentile, DEG_len]+diff_genes
        
    # save the DEGs to a .csv file
    deg_df.to_csv(f, index=False)
    print('saved DEGs to: '+f)

    return


import matplotlib.pyplot as plt
import seaborn as sns

def save_resampling_histogram(
    adata,
    weight_key="log_normalized_resampling_weight",
    color_key="blood_atlas",
    bins=100,
    figsize=(8,6),
    save_path="resampling_histogram.png",
    dpi=300
):
    """
    Save histogram of resampling weights colored by a categorical variable.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing `.obs`.
    weight_key : str
        Column name in adata.obs for the resampling weight values.
    color_key : str
        Column name in adata.obs for the categorical variable to color/group by.
    bins : int
        Number of histogram bins.
    figsize : tuple
        Size of the figure.
    save_path : str
        Path to save the figure (e.g. 'output.png', 'output.pdf').
    dpi : int
        Resolution of saved figure.
    """
    df = adata.obs[[weight_key, color_key]].copy()
    df[color_key] = df[color_key].astype(str)
    print(df.head())

    # extract rows with color_key = Blood
    df_blood = df[df[color_key] == 'Blood']
    df_atlas = df[df[color_key] == 'Atlas']

    # print the number of rows in each group
    print(f"Number of rows in Blood group: {df_blood.shape}")
    print(f"Number of rows in Atlas group: {df_atlas.shape}")

    # calculate mean of resampling weights for each group
    mean_blood = df_blood[weight_key].mean()
    mean_atlas = df_atlas[weight_key].mean()

    # perform statistical test to see if the distributions are significantly different
    p_value = ttest_ind(df_blood[weight_key], df_atlas[weight_key], equal_var=False).pvalue
    print("p-value for difference between Blood and Atlas: "+str(p_value))
    # save p_value in a scientific notation
    p_value = "{:.10e}".format(p_value)

    atlas_count = df_atlas.shape[0]
    # randomely sample df_blood to have the same number of rows as df_atlas
    if df_blood.shape[0] > atlas_count:
        random.seed(42)
        df_blood = df_blood.sample(n=atlas_count, random_state=42)
        print(f"After sampling, number of rows in Blood group: {df_blood.shape}")

    # combine the two dataframes
    df = pd.concat([df_blood, df_atlas], axis=0)
    palette = {"Blood": "orchid", "Atlas": "cadetblue"}
    plt.figure(figsize=figsize)
    ax = sns.histplot(
        data=df,
        x=weight_key,
        hue=color_key,
        bins=bins,
        stat="count",
        kde=True,
        palette=palette,
    )
    sns.move_legend(
        ax, "lower center",
        bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False, fontsize=20
    )
    # add mean of resampling weights as vertical lines
    plt.axvline(mean_blood, color='orchid', linestyle='--', label='Blood Mean')
    plt.axvline(mean_atlas, color='cadetblue', linestyle='--', label='Atlas Mean')
    plt.xlabel('Log(10) Normalized Resampling Weight', fontsize=20)
    plt.ylabel("Count", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(False)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return


def plot_umap_with_weight_annotation(adata, args, model, dimension='high_dim'):
    """Plot UMAP visualization of adata for the input model 
    annotated with resampling weights.
    
    Args:
        adata (anndata.AnnData): Anndata object containing the data.
        args (argparse.Namespace): Arguments passed to the script.
        model (torch.nn.Module): Model used for prediction.
        dimension (str, optional): Dimension to use for UMAP visualization. Defaults to high_dim."""

    print('Tracked epoch: '+str(args.tracked_epoch))
    if not os.path.isdir('figures/umap_w_annotated/'+args.data+'/seed'+str(args.seed)+'/'+dimension+'/'):
        os.makedirs('figures/umap_w_annotated/'+args.data+'/seed'+str(args.seed)+'/'+dimension+'/')

    new_adata = adata.copy()

    with torch.no_grad():

        z = model.get_latent_representation(new_adata)
        # calculate the resampling weights
        w = update_training_sample_probabilities(z)
        # normalize the weights to be between 0 and 1
        new_adata.obs['resampling_weight'] = w
        w = (w - np.min(w)) / (np.max(w) - np.min(w))
        new_adata.obs['normalized_resampling_weight'] = w

        # print statistics of the resampling weights
        print('Resampling weights statistics:')
        print('Min: ', np.min(new_adata.obs['normalized_resampling_weight']))
        print('Max: ', np.max(new_adata.obs['normalized_resampling_weight']))
        print('Mean: ', np.mean(new_adata.obs['normalized_resampling_weight']))
        print('Std: ', np.std(new_adata.obs['normalized_resampling_weight']))
                
        if dimension == 'latent':
            new_adata = sc.AnnData(X=z, obs=new_adata.obs.copy())
        
        if args.model_name == 'scgen':
            # replace the condition labels with ['Control', 'Stimulated']
            adata_unper = new_adata[new_adata.obs.condition == args.adata_label_unper]
            adata_unper.obs.condition = 'Control'

            adata_per = new_adata[new_adata.obs.condition == args.adata_label_per]
            adata_per.obs.condition = 'Stimulated'
            
            new_adata = adata_unper.concatenate(adata_per)
            
        # plot UMAP with weight annotation
        
        # Extract the UMAP coordinates and the values for the feature to color by
        # fix seed for reproducibility
        set_seed(args.seed)
        sc.pp.neighbors(new_adata, random_state=42)
        sc.tl.umap(new_adata, random_state=42)
        
        # Sort the .obs DataFrame based on the 'your_feature' column
        # sorted_indices = new_adata.obs.sort_values(by='normalized_resampling_weight').index

        # # Reorder the AnnData object based on these sorted indices
        # new_adata = new_adata[sorted_indices,:]

        sc.set_figure_params(dpi_save=600, format='pdf', figsize=(6,6))
        red_cmap = mcolors.LinearSegmentedColormap.from_list('custom_red', ['lightgrey', 'darkred'])
        sc.pl.umap(
            new_adata, color="normalized_resampling_weight",
            wspace=0.4, 
            frameon=False,
            save='_w_annotated/'+args.data+'/seed'+str(args.seed)+'/'+dimension+'/'+ \
                args.id+'-w-annotated-epoch-'+args.tracked_epoch+'-umap.pdf',
                title='',
                cmap=red_cmap,
                show=False,
                size=18,
                alpha=1.0,              # transparency of points
                edgecolor="black",      # border color for points
                linewidth=0.1,           # border thickness)
        )

        # Extract UMAP coordinates
        umap = new_adata.obsm['X_umap']
        x, y = umap[:, 0], umap[:, 1]

        # Extract values
        values = new_adata.obs["normalized_resampling_weight"].values

        # Sort by values so low-weight (red) points are plotted last (on top)
        order = values.argsort()
        x, y, values = x[order], y[order], values[order]

        # Custom colormap
        red_cmap = mcolors.LinearSegmentedColormap.from_list('custom_red', ['lightgrey', 'darkred'])
        plt.figure(figsize=(6, 6), dpi=600)
        plt.scatter(
            x, y,
            c=values,
            cmap=red_cmap,
            s=11,                 # marker size
            alpha=1.0,            # transparency
            edgecolors="black",   # border color
            linewidths=0.05       # border thickness
        )
        # show color bar
        cbar = plt.colorbar()
        cbar.set_label('Normalized Resampling Weight', fontsize=25)
        cbar.ax.tick_params(labelsize=20)
        plt.axis("off")
        plt.tight_layout()
        out_path = args.root+'/figures/umap_w_annotated/'+args.data+'/seed'+str(args.seed)+'/'+dimension+'/' + \
                args.id+'-w-annotated-epoch-'+args.tracked_epoch+'-umap_matplotlib.png'
        plt.savefig(out_path, dpi=600, bbox_inches='tight')
        plt.close()


        if args.model_name == 'scgen':
            annotation_list = [args.adata_label_cell, 'condition']
            annotation_label = ['', '']
        else:
            annotation_list = ['blood_atlas']
            annotation_label = ['']
            palette = {"Blood": "orchid", "Atlas": "cadetblue"}


        # extract UMAP coordinates
        umap = new_adata.obsm["X_umap"]
        labels = new_adata.obs[annotation_list[0]]  # assuming annotation_list has one column

        fig, ax = plt.subplots(figsize=(6, 6))
        for group in labels.unique():
            mask = labels == group
            ax.scatter(
                umap[mask, 0],
                umap[mask, 1],
                s=11,
                c=[palette[group]],  # color from palette
                edgecolor="black",
                linewidth=0.1,
                alpha=0.3 if group == "Blood" else 0.7,
                label=group
            )

        ax.legend(fontsize=35, markerscale=8)
        ax.set_title("")
        ax.set_xlabel("UMAP1", fontsize=25)
        ax.set_ylabel("UMAP2", fontsize=25)
        ax.set_frame_on = False
        # remove the border of the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(left=False, bottom=False)
        plt.grid(False)
        # remove x and y ticks
        ax.set_xticks([])
        ax.set_yticks([])
        # adjust the size of legend
        ax.legend(fontsize=20, markerscale=3)


        # check if args.root+'/figures/umap_w_annotated/'+args.data+'/seed'+str(args.seed)+'/' +
        # dimension+'/' exists, if not create it
        if not os.path.isdir(args.root+'/figures/umap_w_annotated/'+args.data+'/seed'+str(args.seed)+'/' +
            dimension+'/'):
            os.makedirs(args.root+'/figures/umap_w_annotated/'+args.data+'/seed'+str(args.seed)+'/' +
            dimension+'/')

        plt.tight_layout()
        plt.savefig(
            args.root+'/figures/umap_w_annotated/'+args.data+'/seed'+str(args.seed)+'/' +
            dimension+'/'+args.id+'-cellytpe-annotated-umap.png',
            dpi=600
        )
        plt.close()


        # print('umap saved in figures/umap_w_annotated/'+args.data+ \
        #     '/seed'+str(args.seed)+'/'+ \
        #     dimension+'/'+args.id+'-cellytpe-annotated-umap.png')


        # # plot histogram of new_adata.obs['normalized_resampling_weight'] and color by new_adata.obs['blood_atlas']
        # if not os.path.isdir(args.root+'/figures/w_histogram/'+args.data+'/seed'+str(args.seed)+'/'):
        #     os.makedirs(args.root+'/figures/w_histogram/'+args.data+'/seed'+str(args.seed)+'/')

        # eps = 1e-6
        # new_adata.obs['log_normalized_resampling_weight'] = \
        #     np.log10(new_adata.obs['normalized_resampling_weight'] + eps)
        # save_resampling_histogram(new_adata,
        #                           save_path=args.root+'/figures/w_histogram/'+args.data+'/seed'+str(args.seed)+'/'+ \
        #                           args.id+'_balanced_histogram_log_normalized_resampling_weights_epoch'+args.tracked_epoch+'.png',
        #                           weight_key='log_normalized_resampling_weight',
        #                           dpi=600)
        # print('w histogram saved in '+args.root+'/figures/w_histogram/'+args.data+ \
        #     '/seed'+str(args.seed)+'/')

    return


def convert_matrix_to_tensor(X):
    """Convert a matrix to a tensor.
    
    Args:
        X (any array): Input matrix.
        
    Returns:
        X (torch.Tensor): Converted tensor."""
    
    if isinstance(X, (np.ndarray)):
        X = torch.from_numpy(X).float()
    else:
        X = torch.from_numpy(sparse.coo_matrix(X).toarray()).float()
    X = X.to(torch.float32)
    
    return X


def _harmonize_condition_labels(latent_adata, args):
    """Map raw condition labels to Control / Stimulated for evaluation."""
    latent_adata_unper = latent_adata[
        latent_adata.obs.condition == args.adata_label_unper]
    latent_adata_unper.obs.condition = 'Control'

    latent_adata_per = latent_adata[
        latent_adata.obs.condition == args.adata_label_per]
    latent_adata_per.obs.condition = 'Stimulated'

    latent_adata = latent_adata_unper.concatenate(latent_adata_per)
    assert latent_adata_unper.shape[0] + latent_adata_per.shape[0] == \
        latent_adata.shape[0]
    return latent_adata


def _save_latent_clustering_metrics(
    args,
    model_name,
    data_type,
    n_cells,
    n_clusters,
    label_col,
    sil_score,
    davies_score,
    calinski_score,
    ari_score,
    nmi_score,
    kmeans_seed,
):
    """Append latent-space clustering metrics to a per-experiment CSV."""
    out_dir = os.path.join(_repo_root(args), 'result', 'test', args.data, 'clustering')
    os.makedirs(out_dir, exist_ok=True)
    file = os.path.join(
        out_dir,
        f'{args.test_id}_{model_name}_{data_type}_latent_clustering_'
        f'{label_col}_seed{args.seed}.csv',
    )

    dist_df = pd.DataFrame([{
        'experiment_id': args.test_id,
        'test': args.test_data[0],
        'model': model_name,
        'AR': model_name == 'AR',
        'variable_con': args.variable_con,
        'con_percent': args.con_percent,
        'in_dist_group': args.in_dist_group,
        'data_type': data_type,
        'n_cells': n_cells,
        'n_clusters': n_clusters,
        'ground_truth_label': label_col,
        'silhouette_score': round(sil_score, 4),
        'davies_bouldin_score': round(davies_score, 4),
        'calinski_harabasz_score': round(calinski_score, 4),
        'ari': round(ari_score, 4),
        'nmi': round(nmi_score, 4),
        'seed': args.seed,
        'kmeans_seed': kmeans_seed,
    }])

    print(f'Saving latent clustering metrics to {file}')
    print(dist_df)
    if not os.path.isfile(file):
        dist_df.to_csv(file, index=False)
    else:
        dist_df.to_csv(file, mode='a', header=False, index=False)


def _match_clusters_to_true_labels(true_labels, cluster_labels):
    """Map each KMeans id to a unique GT label (Hungarian on contingency)."""
    true_labels = np.asarray(true_labels).astype(str)
    cluster_labels = np.asarray(cluster_labels).astype(str)
    true_cats = np.unique(true_labels)
    clust_cats = np.unique(cluster_labels)

    cont = np.zeros((len(clust_cats), len(true_cats)), dtype=np.int64)
    clust_index = {c: i for i, c in enumerate(clust_cats)}
    true_index = {t: i for i, t in enumerate(true_cats)}
    for t, c in zip(true_labels, cluster_labels):
        cont[clust_index[c], true_index[t]] += 1

    # Pad to square if needed so every cluster gets a distinct label when possible.
    n = max(cont.shape[0], cont.shape[1])
    cost = np.zeros((n, n), dtype=np.float64)
    cost[: cont.shape[0], : cont.shape[1]] = cont
    row_ind, col_ind = linear_sum_assignment(-cost)

    mapping = {}
    for r, c in zip(row_ind, col_ind):
        if r < len(clust_cats) and c < len(true_cats):
            mapping[clust_cats[r]] = true_cats[c]
    for c in clust_cats:
        if c not in mapping:
            mapping[c] = true_cats[int(np.argmax(cont[clust_index[c]]))]
    return np.asarray([mapping[c] for c in cluster_labels], dtype=object)


def evaluate_latent_clustering(
    latent_adata, args, model_name, data_type, label_col='condition'):
    """Cluster latent embeddings and compare to ground-truth labels.

    Writes ``kmeans_cluster`` and ``kmeans_matched_label`` (GT names via
    Hungarian matching) into ``latent_adata.obs`` for downstream plots.
    Returns a dict of scores, or ``None`` if clustering is skipped.
    """
    X = latent_adata.X
    true_labels = latent_adata.obs[label_col].astype(str).values
    n_clusters = len(np.unique(true_labels))

    if n_clusters < 2 or X.shape[0] <= n_clusters:
        print(
            f'Skipping latent clustering for {model_name} ({data_type}): '
            f'need >{n_clusters} cells for {n_clusters} clusters.'
        )
        return None

    kmeans_seed = args.seed
    set_seed(kmeans_seed)
    cluster_labels = KMeans(
        n_clusters=n_clusters,
        random_state=kmeans_seed,
        n_init=10,
    ).fit_predict(X)

    matched_labels = _match_clusters_to_true_labels(true_labels, cluster_labels)
    latent_adata.obs['kmeans_cluster'] = pd.Categorical(
        np.asarray(cluster_labels).astype(str)
    )
    latent_adata.obs['kmeans_matched_label'] = pd.Categorical(
        matched_labels.astype(str)
    )

    sil_score = silhouette_score(X, cluster_labels)
    davies_score = davies_bouldin_score(X, cluster_labels)
    calinski_score = calinski_harabasz_score(X, cluster_labels)
    ari_score = adjusted_rand_score(true_labels, cluster_labels)
    nmi_score = normalized_mutual_info_score(
        true_labels, cluster_labels, average_method='arithmetic')

    print(
        f'Latent clustering ({model_name}, {data_type}, '
        f'label={label_col}, n={X.shape[0]}, k={n_clusters}):'
    )
    print(f'  Silhouette score: {sil_score:.3f}')
    print(f'  Davies-Bouldin score: {davies_score:.3f}')
    print(f'  Calinski-Harabasz score: {calinski_score:.3f}')
    print(f'  ARI vs. {label_col}: {ari_score:.3f}')
    print(f'  NMI vs. {label_col}: {nmi_score:.3f}')

    _save_latent_clustering_metrics(
        args,
        model_name,
        data_type,
        X.shape[0],
        n_clusters,
        label_col,
        sil_score,
        davies_score,
        calinski_score,
        ari_score,
        nmi_score,
        kmeans_seed,
    )
    return {
        'ari': ari_score,
        'nmi': nmi_score,
        'silhouette': sil_score,
        'n_clusters': n_clusters,
    }


def _encode_latent_adata(model, cells_adata, args):
    """Project cells into model latent space (one forward pass)."""
    latent_X = model.get_latent_representation(cells_adata)
    latent_adata = sc.AnnData(X=latent_X, obs=cells_adata.obs.copy())
    return _harmonize_condition_labels(latent_adata, args)


def _assert_latent_task_labels(latent_adata, args, data_type, label_col):
    """Sanity-check cell sets and GT columns before clustering."""
    cond = latent_adata.obs['condition'].astype(str)
    cell = latent_adata.obs[args.adata_label_cell].astype(str)
    n = latent_adata.n_obs
    assert n > 0, f'{data_type}: empty latent AnnData'
    assert label_col in latent_adata.obs.columns, (
        f'{data_type}: missing label column {label_col!r}'
    )
    # Harmonize maps raw control/stim labels → Control / Stimulated.
    assert set(cond.unique()) <= {'Control', 'Stimulated'}, (
        f'{data_type}: unexpected condition labels {sorted(cond.unique())}'
    )

    if data_type == 'test':
        assert label_col == 'condition', (
            f'test task must use condition labels, got {label_col!r}'
        )
        assert set(cell.unique()) <= set(map(str, args.test_data)), (
            f'test cells must be held-out group(s) {args.test_data}, '
            f'got {sorted(cell.unique())}'
        )
        assert cond.nunique() == 2, (
            f'test condition clustering needs both Control and Stimulated, '
            f'got {sorted(cond.unique())}'
        )
    elif data_type == 'train':
        assert label_col == args.adata_label_cell, (
            f'train task must use cell-group column '
            f'{args.adata_label_cell!r}, got {label_col!r}'
        )
        # Same exclusion as training: no held-out × stimulated cells.
        held_out_stim = cell.isin(list(map(str, args.test_data))) & (
            cond == 'Stimulated'
        )
        assert not held_out_stim.any(), (
            'train set must not contain held-out stimulated cells '
            f'(found {int(held_out_stim.sum())})'
        )
        n_groups = latent_adata.obs[label_col].astype(str).nunique()
        assert n_groups >= 2, (
            f'train cell-group clustering needs ≥2 groups, got {n_groups}'
        )
    else:
        raise ValueError(f'Unknown data_type {data_type!r}')

    print(
        f'  label checks OK | n={n} | {label_col} levels='
        f'{sorted(latent_adata.obs[label_col].astype(str).unique())}',
        flush=True,
    )


def _ensure_latent_umap(latent_adata, seed):
    """Compute a 2D embedding of latent ``.X`` once; reuse if already present.

    Uses PCA → ``obsm['X_umap']`` so ``sc.pl.umap`` can plot it. On macOS,
    umap-learn / scanpy-neighbors often segfault under mixed OpenMP
    (libiomp + libomp) after torch/sklearn are both loaded; PCA avoids that.
    Latent dim is already low (e.g. 64), so PCA-2 is a reasonable viz.
    """
    if 'X_umap' in latent_adata.obsm:
        return
    set_seed(seed)
    X = np.asarray(latent_adata.X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f'Expected 2D latent matrix, got shape {X.shape}')
    if not np.isfinite(X).all():
        raise ValueError('Non-finite values in latent representation')
    n_comp = min(2, X.shape[0], X.shape[1])
    emb = PCA(n_components=n_comp, random_state=seed).fit_transform(X)
    if n_comp == 1:
        emb = np.hstack([emb, np.zeros((emb.shape[0], 1), dtype=emb.dtype)])
    latent_adata.obsm['X_umap'] = emb
    print(
        f'Computed PCA-2 embedding of latent for plotting '
        f'(n={X.shape[0]}, d={X.shape[1]})',
        flush=True,
    )


def _plot_latent_umap_true_vs_kmeans(
    latent_adata,
    args,
    model_name,
    data_type,
    label_col,
    scores=None,
    true_palette=None,
):
    """Save separate GT- and KMeans-colored PCA-2 plots of the latent (no titles).

    KMeans clusters are shown under matched GT label names with the same
    category order and colors as the ground-truth panel.
    """
    import matplotlib
    matplotlib.use('Agg', force=False)

    if 'kmeans_matched_label' not in latent_adata.obs.columns:
        print(
            f'Skipping latent embedding plots for {model_name} ({data_type}): '
            'no kmeans_matched_label in obs.'
        )
        return

    fig_dir = os.path.join('figures', 'umap', args.data, f'seed{args.seed}')
    os.makedirs(fig_dir, exist_ok=True)

    true_plot_col = 'true_label'
    km_plot_col = 'kmeans_matched_label'
    true_vals = latent_adata.obs[label_col].astype(str).values
    km_vals = latent_adata.obs[km_plot_col].astype(str).values

    if label_col == 'condition':
        cat_order = [
            c for c in ('Control', 'Stimulated')
            if c in set(true_vals) or c in set(km_vals)
        ]
        for c in sorted(set(true_vals) | set(km_vals)):
            if c not in cat_order:
                cat_order.append(c)
    else:
        cat_order = sorted(set(true_vals) | set(km_vals))

    latent_adata.obs[true_plot_col] = pd.Categorical(
        true_vals, categories=cat_order
    )
    latent_adata.obs[km_plot_col] = pd.Categorical(
        km_vals, categories=cat_order
    )

    if true_palette is not None and all(c in true_palette for c in cat_order):
        colors = [true_palette[c] for c in cat_order]
    else:
        # Stable default palette shared by GT and matched-KMeans panels.
        default = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
            '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
        ]
        colors = [default[i % len(default)] for i in range(len(cat_order))]

    latent_adata.uns[f'{true_plot_col}_colors'] = colors
    latent_adata.uns[f'{km_plot_col}_colors'] = list(colors)

    _ensure_latent_umap(latent_adata, args.seed)
    sc.set_figure_params(dpi=200, format='png')

    held_out = args.test_data[0]
    base = (
        f'{args.data}-heldout{held_out}-model{model_name}'
        f'-cells_{data_type}-label_{label_col}'
        f'-varcon{args.variable_con}-conpct{args.con_percent}'
        f'-seed{args.seed}-pca2'
    )

    plot_specs = [
        (true_plot_col, f'{base}-color_ground_truth.png'),
    ]
    ari = None if scores is None else scores.get('ari')
    nmi = None if scores is None else scores.get('nmi')
    km_suffix = '-color_kmeans'
    if ari is not None and nmi is not None:
        km_suffix += f'-ari{ari:.3f}-nmi{nmi:.3f}'
    plot_specs.append((km_plot_col, f'{base}{km_suffix}.png'))

    for color_key, filename in plot_specs:
        save_name = f'/{args.data}/seed{args.seed}/{filename}'
        sc.pl.umap(
            latent_adata,
            color=color_key,
            frameon=False,
            save=save_name,
            show=False,
            title='',
            legend_loc='right margin',
        )
        plt.close('all')
        print(
            f'Saved latent PCA-2 plot: figures/umap{save_name}',
            flush=True,
        )


def _cluster_latent_and_plot_umap(
    latent_adata,
    args,
    model_name,
    data_type,
    label_col,
    true_palette=None,
):
    """Run KMeans vs ``label_col``, save metrics, plot separate GT / KMeans figs."""
    scores = evaluate_latent_clustering(
        latent_adata,
        args,
        model_name,
        data_type,
        label_col=label_col,
    )
    if scores is None:
        return None
    _plot_latent_umap_true_vs_kmeans(
        latent_adata,
        args,
        model_name,
        data_type,
        label_col=label_col,
        scores=scores,
        true_palette=true_palette,
    )
    return scores


def _prepare_test_adata_for_latent(args, train_adata, test_adata):
    """Test cells with the same label overrides as the historical test path."""
    new_adata = test_adata.copy()
    if args.variable_con and args.con_percent == 0.0:
        # Match prior behaviour: briefly set then restore true test labels.
        new_adata.obs[args.adata_label_cell] = (
            test_adata.obs[args.adata_label_cell].values
        )
    if args.in_dist_group != '':
        new_adata.obs[args.adata_label_cell] = args.in_dist_group
    return new_adata


def run_latent_clustering_pipeline(model_dic, args, adata):
    """Full latent-clustering suite for ``--latent_clustering_all``.

    Cell splits:

    - **train**: every cell except held-out × stimulated (train+validation;
      held-out **control** cells are included). GT label =
      ``args.adata_label_cell`` (e.g. ``cell_type``).
    - **test**: held-out cell group, both control and stimulated. GT label
      = ``condition`` (Control vs Stimulated after harmonization).

    For each of AR and Naive: encode once → KMeans (k = #GT levels) in
    latent space → ARI/NMI vs GT + cluster-quality metrics → separate
    PCA-2 plots colored by GT and by Hungarian-matched KMeans labels.
    """
    fig_dir = os.path.join('figures', 'umap', args.data, f'seed{args.seed}')
    os.makedirs(fig_dir, exist_ok=True)

    _, test_adata, _, _ = get_train_test_adata(args, adata)
    train_cells = _get_train_valid_adata(args, adata)
    test_cells = _prepare_test_adata_for_latent(args, train_cells, test_adata)
    condition_palette = {'Stimulated': 'darkorange', 'Control': 'dodgerblue'}

    tasks = [
        ('test', test_cells, 'condition', condition_palette),
        ('train', train_cells, args.adata_label_cell, None),
    ]

    for model_name, model in model_dic.items():
        if model_name not in ('Naive', 'AR'):
            continue
        print('=' * 72)
        print(f'Latent clustering suite | model={model_name}')
        print('=' * 72)
        for data_type, cells_adata, label_col, palette in tasks:
            print(
                f'--> {data_type}: n={cells_adata.n_obs}, label_col={label_col}',
                flush=True,
            )
            latent_adata = _encode_latent_adata(model, cells_adata, args)
            _assert_latent_task_labels(
                latent_adata, args, data_type, label_col
            )
            _cluster_latent_and_plot_umap(
                latent_adata,
                args,
                model_name,
                data_type,
                label_col=label_col,
                true_palette=palette,
            )


def visualize_latent_space(
    model_dic,
    args,
    adata,
    data_type,
    cluster_label_col=None,
):
    """Cluster + UMAP for a single cell subset (train / test / all).

    Prefer :func:`run_latent_clustering_pipeline` when ``--latent_clustering_all``
    is set (runs both test-condition and train-cell-group tasks).
    """
    fig_dir = os.path.join('figures', 'umap', args.data, f'seed{args.seed}')
    os.makedirs(fig_dir, exist_ok=True)

    train_adata, test_adata, _, _ = get_train_test_adata(args, adata)
    if data_type == 'train':
        # Full seen set (train+valid): exclude only held-out stimulated cells.
        cells_adata = _get_train_valid_adata(args, adata)
        default_label = args.adata_label_cell
        true_palette = None
    elif data_type == 'test':
        cells_adata = _prepare_test_adata_for_latent(
            args, train_adata, test_adata
        )
        default_label = 'condition'
        true_palette = {'Stimulated': 'darkorange', 'Control': 'dodgerblue'}
    elif data_type == 'all':
        cells_adata = _get_train_valid_adata(args, adata).concatenate(
            test_adata
        )
        default_label = args.adata_label_cell
        true_palette = None
    else:
        raise ValueError(
            f"data_type must be 'train', 'test', or 'all', got {data_type!r}"
        )

    label_col = (
        cluster_label_col if cluster_label_col is not None else default_label
    )

    for model_name, model in model_dic.items():
        if model_name not in ('Naive', 'AR'):
            continue
        latent_adata = _encode_latent_adata(model, cells_adata, args)
        _cluster_latent_and_plot_umap(
            latent_adata,
            args,
            model_name,
            data_type,
            label_col=label_col,
            true_palette=true_palette,
        )
    return



def calculate_diff_corr(adata, args, DEG_idx):
    """Calculate the correlation between the real differnce of stimulated and control 
    versus model predictions and its difference with control.
    
    Args:
        adata (anndata.AnnData): Anndata object containing the data.
        args (argparse.Namespace): Arguments passed to the script.
        DEG_idx (list): Indices of the DEGs."""

    global test_metric
    # calculate R for difference between stimulated and control
    unper_mean = np.mean(adata[adata.obs.condition == \
        args.adata_label_unper].X.toarray().astype(float), axis=0)
    per_mean = np.mean(adata[adata.obs.condition == \
        args.adata_label_per].X.toarray().astype(float), axis=0)
    per_diff = per_mean - unper_mean
    assert unper_mean.shape == per_mean.shape == per_diff.shape

    for method in set(adata.obs['condition']):
        if method != args.adata_label_unper and method != args.adata_label_per:
            args.AR = False if method == 'Naive' else True
            create_id(args)

            pred_mean = np.mean(adata[adata.obs.condition == \
                method].X.toarray().astype(float), axis=0)
            pred_diff = pred_mean - unper_mean
            diff_corr, p_value = pearsonr(per_diff, pred_diff)
            test_metric.loc[test_metric.shape[0]] = \
                [args.test_id, args.test_data[0], args.AR, args.variable_con, args.con_percent,
                 args.in_dist_group, 'Diff R', round(diff_corr, 4), args.seed]
            
            mse = F.mse_loss(torch.from_numpy(pred_diff),
                             torch.from_numpy(per_diff))
            test_metric.loc[test_metric.shape[0]] = \
                [args.test_id, args.test_data[0], args.AR, args.variable_con, args.con_percent,
                 args.in_dist_group, 'Diff MSE', round(mse.item(), 6), args.seed]
                
            DEG_diff_corr, p_value = pearsonr(
                per_diff[DEG_idx],
                pred_diff[DEG_idx])
            
            test_metric.loc[test_metric.shape[0]] = \
                [args.test_id, args.test_data[0], args.AR, args.variable_con, args.con_percent,
                 args.in_dist_group, '20 DEG Diff R', round(DEG_diff_corr, 4),
                 args.seed]
                    
            mse = F.mse_loss(torch.from_numpy(pred_diff[DEG_idx]),
                             torch.from_numpy(per_diff[DEG_idx]))
            test_metric.loc[test_metric.shape[0]] = \
                [args.test_id, args.test_data[0], args.AR, args.variable_con, args.con_percent,
                 args.in_dist_group, '20 DEG Diff MSE', round(mse.item(), 6),
                 args.seed]

    return


def extract_DEG(test_adata, args, count=100):
    """Extract the differentially expressed genes.
    
    Args:
        test_adata (anndata.AnnData): Anndata object containing the data.
        args (argparse.Namespace): Arguments passed to the script.
        count (int, optional): Number of DEGs to extract.
        
    Returns:
        list: Indices of the DEGs."""
    diff_genes = []
    sc.tl.rank_genes_groups(test_adata,
                            groupby="condition",
                            method="wilcoxon")
    diff_genes = \
        test_adata.uns["rank_genes_groups"]["names"][
            args.adata_label_per][:count].tolist()

    if args.data in ['sciplex3', 'species', 'lps-hpoly']:
        gene_list = test_adata.var.index.tolist()
    else:
        gene_list = test_adata.var["gene_symbol"].tolist()

    DEG_idx = []
    for i in range(len(diff_genes)):
        for j in range(len(gene_list)):
            if gene_list[j] == diff_genes[i]:
                DEG_idx.append(j)

    return DEG_idx


def cosine_similarity(vector_a, vector_b):
    """Calculate the cosine similarity between two vectors.
    
    Args:
        vector_a (np.ndarray): First vector.
        vector_b (np.ndarray): Second vector.
        
    Returns:
        float: Cosine similarity between the two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = norm(vector_a)
    norm_b = norm(vector_b)

    similarity = dot_product / (norm_a * norm_b)
    return similarity


def compare_cosine_similarity(adata, args, DEG_idx):
    """Compare the cosine similarity between the real stimulated and model predictions.
    
    Args:
        adata (anndata.AnnData): Anndata object containing the data.
        args (argparse.Namespace): Arguments passed to the script.
        DEG_idx (list): Indices of the DEGs."""

    global test_metric
    for model_name in ['Naive', 'AR']:
        args.AR = False if model_name == 'Naive' else True
        create_id(args)
        per_mean = np.mean(adata[adata.obs.condition == \
            args.adata_label_per].X.toarray().astype(float), axis=0)

        pred_mean = np.mean(adata[adata.obs.condition == \
            model_name].X.toarray().astype(float), axis=0)

        cos_sim = cosine_similarity(
            per_mean,
            pred_mean)
        test_metric.loc[test_metric.shape[0]] = \
            [args.test_id, args.test_data[0], args.AR, args.variable_con, args.con_percent,
             args.in_dist_group, 'Cosine Similarity', cos_sim, args.seed]
            
        # calculate cosine similarity for DEGs
        cos_sim = cosine_similarity(
            per_mean[DEG_idx],
            pred_mean[DEG_idx])
        test_metric.loc[test_metric.shape[0]] = \
            [args.test_id, args.test_data[0], args.AR, args.variable_con, args.con_percent,
             args.in_dist_group, '20 DEG Cosine Similarity', cos_sim, args.seed]

    return


def create_test_id(args):
    """Create the test id.
    
    Args:
        args (argparse.Namespace): Arguments passed to the script."""

    args.test_id = args.data+ \
        '-ood'+str(args.ood)+ \
        '-variable_con'+str(args.variable_con)
        
    if args.in_dist_group != '':
        args.test_id = args.test_id+'-indist'
    return


def test_scgen(args, adata=None, model_dic={}):
    """Test the model.
    
    Args:
        args (argparse.Namespace): Arguments passed to the script.
        adata (anndata.AnnData, optional): Anndata object containing the data.
        model_dic (dict, optional): Dictionary containing the model for each method.
        """

    global root
    eval_adata_dic = {}

    train_adata, test_adata, unper_test_adata, per_test_adata = \
        get_train_test_adata(args, adata)
    create_test_id(args)
    
    if not getattr(args, 'latent_clustering_all', False):
        unper_test_adata.X = convert_matrix_to_tensor(unper_test_adata.X)
        per_test_adata.X = convert_matrix_to_tensor(per_test_adata.X)
        all_test_adata = per_test_adata.concatenate(unper_test_adata)
        all_adata = train_adata.concatenate(test_adata)
        
        assert set(unper_test_adata.obs[args.adata_label_cell]) == \
            set(per_test_adata.obs[args.adata_label_cell])

        # plot umap plots annotated by AR weights
        if args.plot_umap_annotated_with_w:
            if 'AR' in args.model:
                args.AR = True
                create_id(args)
                model = load_eval_adata(args, adata, 'AR')
                plot_umap_with_weight_annotation(all_adata, args, model)
                plot_umap_with_weight_annotation(all_adata, args, model, 'latent')
                return
            
        # extract DEGs based on resampling weights
        if (args.degs_extraction_based_on_resampling_w) & (args.tracked_epoch=='best'):
            if 'AR' in args.model:
                args.AR = True
                create_id(args)
                model = load_eval_adata(args, adata, 'AR')
                extract_DEGs_based_on_resampling_weights(all_adata, args, model)
                return

    # create two dictionaries to store the model and eval_adata for each method
    models = args.model.split(',')
    for model_name in models:
        args.AR = False if model_name == 'Naive' else True
        create_id(args)
        if getattr(args, 'latent_clustering_all', False):
            model = load_eval_adata(args, adata, model_name)
            model_dic[model_name] = model
        else:
            eval_adata, model, pred_adata = \
                load_eval_adata(args, adata, model_name)
            pred_adata.X = convert_matrix_to_tensor(pred_adata.X)
            all_test_adata = pred_adata.concatenate(all_test_adata)

            assert pred_adata.obs[args.adata_label_cell].tolist() == \
                unper_test_adata.obs[args.adata_label_cell].tolist()
            assert pred_adata.obs.condition.tolist() == \
                [model_name]*pred_adata.shape[0]
            
            eval_adata.X = convert_matrix_to_tensor(eval_adata.X)
            model_dic[model_name] = model
            eval_adata_dic[model_name] = eval_adata
        
    # print keys of the eval_adata_dic and model_dic
    # print('eval_adata_dic keys: ')
    # print(eval_adata_dic.keys())
    # print('model_dic keys: ')
    # print(model_dic.keys())

    if getattr(args, 'latent_clustering_all', False):
        # Test (condition) + train (cell-group) clustering + GT|KMeans UMAPs
        run_latent_clustering_pipeline(model_dic, args, adata)
    else:
        # extract DEG index
        DEG_idx_dic = {}
        DEG_idx = []
        for count in [20, 50, 100]:
            DEG_idx = extract_DEG(
                per_test_adata.concatenate(unper_test_adata),
                args, count=count)
            DEG_idx_dic[count] = DEG_idx

        # calculate correlation of model predictions with ground truth
        evaluate_correlation(eval_adata_dic, args, DEG_idx_dic)
        
        # calculate cosine similarity
        compare_cosine_similarity(all_test_adata, args, DEG_idx_dic[20])
        
        # visualize latent space and clustering metrics
        # visualize_latent_space(model_dic, args, adata, 'train')
        # visualize_latent_space(model_dic, args, adata, 'test')
        # visualize_latent_space(model_dic, args, adata, 'all')  # train+test, ARI/NMI vs cell type

        # R2 between differences of sitmulated and control
        # calculate_diff_corr(all_test_adata, args, DEG_idx_dic[20])

        # save results
        global test_metric
        file = root+'result/test/'+args.data+'/'+ \
            "{:%Y%m%d}".format(datetime.now())+'-'+args.test_id+'.csv'
        # check if there is a file with the same name
        if not os.path.isfile(file):
            test_metric.to_csv(file, index=False, header=True)
        else:
            test_metric.to_csv(file, mode='a', index=False, header=False)

    return


def test_scvi(args, train_adata):
    """Test the scVI model.
    
    Args:
        args (argparse.Namespace): Arguments passed to the script.
        train_adata (anndata.AnnData): Anndata object containing the training data."""
    
    train_adata_copy = train_adata.copy()
    print('train_adata_copy shape: ', train_adata_copy.shape)

    # plot umap plots annotated by AR weights
    if args.plot_umap_annotated_with_w:
        if 'AR' in args.model:
            args.AR = True
            create_id(args)
            path_to_model_folder = args.out_path+args.data+"/seed"+str(args.seed) + \
                '/'+args.id
            if args.tracked_epoch == 'best':
                path_to_model_folder = path_to_model_folder+'-best'
            else:
                path_to_model_folder = path_to_model_folder + \
                    '-epoch'+str(args.tracked_epoch)
            model = scvi.model.SCVI.load(path_to_model_folder, train_adata_copy)
            # plot_umap_with_weight_annotation(all_adata, args, model, 'latent')
            plot_umap_with_weight_annotation(train_adata_copy, args, model, 'high_dim')
            return
    
    # store resampling weights
    if args.store_resampling_weights:
        if 'AR' in args.model:
            args.AR = True
            create_id(args)
            path_to_model_folder = args.out_path+args.data+"/seed"+str(args.seed) + \
                '/'+args.id
            if args.tracked_epoch == 'best':
                path_to_model_folder = path_to_model_folder+'-best'
            else:
                path_to_model_folder = path_to_model_folder + \
                    '-epoch'+str(args.tracked_epoch)
            model = scvi.model.SCVI.load(path_to_model_folder, train_adata_copy)
            with torch.no_grad():

                z = model.get_latent_representation(train_adata_copy)
                # calculate the resampling weights
                w = update_training_sample_probabilities(z)

                # save the resampling weights in a csv file
                resampling_weights_df = pd.DataFrame({'resampling_weight': w})
                # assert number of rows in resampling_weights_df is equal to number of rows in train_adata_copy
                assert resampling_weights_df.shape[0] == train_adata_copy.shape[0]
                resampling_weights_df.to_csv(args.resampling_weight_path, index=False)
                print('saved resampling weights to: '+args.resampling_weight_path)

    # store weight diagnostics (normalized entropy and ESS)
    if args.store_weight_diagnostics:
        if 'AR' in args.model:
            args.AR = True
            create_id(args)
            path_to_model_folder = args.out_path+args.data+"/seed"+str(args.seed) + \
                '/'+args.id
            if args.tracked_epoch == 'best':
                path_to_model_folder = path_to_model_folder+'-best'
            else:
                path_to_model_folder = path_to_model_folder + \
                    '-epoch'+str(args.tracked_epoch)
            model = scvi.model.SCVI.load(path_to_model_folder, train_adata_copy)
            with torch.no_grad():
                z = model.get_latent_representation(train_adata_copy)
                w = update_training_sample_probabilities(z)

                n = len(w)
                # normalized Shannon entropy: 1 = uniform weights, 0 = all mass on one cell
                entropy = -np.sum(w * np.log(w + 1e-12))
                entropy_normalized = entropy / np.log(n)
                # effective sample size: how many cells are effectively being sampled
                ess = 1.0 / np.sum(w ** 2)
                ess_normalized = ess / n

                diagnostics_df = pd.DataFrame({
                    'entropy_normalized': [entropy_normalized],
                    'ess': [ess],
                    'ess_normalized': [ess_normalized],
                })
                diagnostics_df.to_csv(args.weight_diagnostics_path, index=False)
                print('entropy_normalized: ', entropy_normalized)
                print('ess_normalized: ', ess_normalized)
                print('saved weight diagnostics to: '+args.weight_diagnostics_path)
    return