from dataloader import adjust_training_proportions
from scgen_DB.scgen._scgen import SCGEN
from utils import set_seed, create_id
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
from scipy.stats import pearsonr
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from numpy.linalg import norm
from scipy import sparse
import seaborn as sns
import pandas as pd
import scanpy as sc
import numpy as np
from umap import UMAP
import scperturb
import random
import scvi
import torch
import pca
import os


test_metric = pd.DataFrame(
    columns=['experiment_id', 'test', 'AR', 'variable_con',
             'con_percent','in_dist_group', 'metric',
             'value', 'seed'])
path = os.getcwd()
root = os.path.abspath(os.path.join(path, os.pardir))+'/'


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

        in_dist_adata = \
            valid_adata[valid_adata.obs[args.adata_label_cell] == args.in_dist_group]
                
        # randomly sample from in_dist_adata
        set_seed(args.seed)
        valid_index = random.sample(range(in_dist_adata.shape[0]),
                                    int(in_dist_adata.shape[0]/2))
        not_valid_index = \
            [x for x in range(in_dist_adata.shape[0])
            if x not in valid_index]
        in_dist_not_valid_adata = in_dist_adata[not_valid_index]
            
        unper_test_adata = in_dist_not_valid_adata[(
            in_dist_not_valid_adata.obs.condition == args.adata_label_unper)]
        per_test_adata = in_dist_not_valid_adata[(
            in_dist_not_valid_adata.obs.condition == args.adata_label_per)]

    test_adata = per_test_adata.concatenate(unper_test_adata)
    return train_adata, test_adata, unper_test_adata, per_test_adata


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
        
    # root = '/Users/zeinab/Documents/MSR_internship/project/MSR_internship_new_git/sc-uncertainty/'
    model_path = root+"saved_models/"+args.data+"/seed"+str(args.seed)+'/'+args.id+"-best.pt"
    if args.plot_umap_annotated_with_w:
        if args.tracked_epoch != 'best':
            model_path = root+"saved_models/"+args.data+"/seed"+str(args.seed)+'/'+args.id+"-epoch"+str(args.tracked_epoch)+".pt"
    
    print('model_path: '+model_path)
    model = SCGEN.load(
        model_path,
        train_adata.copy(),
        use_gpu = args.gpu)
    model.is_trained = True
    
    if args.plot_umap_annotated_with_w | args.degs_extraction_based_on_resampling_w:
        print('successfully loaded the model in: '+model_path)
        return model

    pred_adata = sc.read(root+'prediction/'+args.data+'/seed'+str(args.seed)+
                         '/'+args.id+'-pred-adata.h5ad')
    print(pred_adata)
    print(pred_adata.obs['condition'])
    print(model_name)
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


def plot_pca(eval_adata, args, title):
    """Plot PCA of all cell groups: contorol, perturbed, Naive and AR model predictions.
    
    Args:
        eval_adata (anndata.AnnData): Anndata object containing the data.
        args (argparse.Namespace): Arguments passed to the script.
        title (str): Title of the plot."""

    color_dic = {}
    color_dic['Control'] = 'orange'
    color_dic['Naive'] = 'darkgray'
    color_dic['AR'] = 'brown'
    color_dic['Real Stimulated'] = 'darkcyan'
    
    condition = ['Control', 'Naive', 'AR', 'Real Stimulated']

    per_adata = eval_adata[eval_adata.obs.condition == args.adata_label_per]
    per_adata.obs.condition = 'Real Stimulated'
    per_adata.obs['color'] = 'darkcyan'

    unper_adata = eval_adata[eval_adata.obs.condition == args.adata_label_unper]
    unper_adata.obs.condition = 'Control'
    unper_adata.obs['color'] = 'orange'

    for c in condition:
        if c != args.adata_label_per and \
            c != args.adata_label_unper:
            pred_adata = eval_adata[eval_adata.obs.condition == c]
            pred_adata.obs['color'] = color_dic[c]
            print('condition: '+c)
            print(pred_adata.X)
            print()
            unper_adata = unper_adata.concatenate(pred_adata)
            
    new_eval_adata = unper_adata.concatenate(per_adata)

    sc.tl.pca(new_eval_adata)
    
    title = title+'-conp'+str(args.con_percent)
    sc.pl.pca(new_eval_adata, color="condition", frameon=False,
              title=[title+' PC [1,2]', title+' PC [2,3]', title+' PC [3,4]'],
              palette=color_dic,
              components= ["1,2", "2,3", "3,4"],
              save="/"+args.data+"/seed"+str(args.seed)+"/"+title.replace(' ', '')+'_pca.png',
              show=False)

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
        w = (w - np.min(w)) / (np.max(w) - np.min(w))
        new_adata.obs['normalized_resampling_weight'] = w
                
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
        sc.pp.neighbors(new_adata)
        set_seed(args.seed)
        sc.tl.umap(new_adata)
        
        # Sort the .obs DataFrame based on the 'your_feature' column
        # sorted_indices = new_adata.obs.sort_values(by='normalized_resampling_weight').index

        # # Reorder the AnnData object based on these sorted indices
        # new_adata = new_adata[sorted_indices,:]

        sc.set_figure_params(dpi=600, format='png')
        red_cmap = mcolors.LinearSegmentedColormap.from_list('custom_red', ['lightgrey', 'red'])
        sc.pl.umap(
            new_adata, color="normalized_resampling_weight",
            wspace=0.4, 
            frameon=False,
            save='_w_annotated/'+args.data+'/seed'+str(args.seed)+'/'+dimension+'/'+ \
                args.id+'-w-annotated-epoch-'+args.tracked_epoch+'-umap.png',
                title='',
                cmap=red_cmap,
                show=False,
                size=11)
            
        # plot UMAP with condition and cell type annotations
        if args.model_name == 'scgen':
            annotation_list = [args.adata_label_cell, 'condition']
            annotation_label = ['', '']
        else:
            annotation_list = ['blood_atlas']
            annotation_label = ['']
            palette = {"Blood": "lightblue", "Atlas": "darkslategrey"}

        sc.pl.umap(
            new_adata,
            color=annotation_list,
            wspace=0.4,
            frameon=False,
                save='_w_annotated/'+args.data+'/seed'+str(args.seed)+'/'+ \
                    dimension+'/'+args.id+'-cellytpe-annotated-umap.png',
            title=annotation_label,
            show=False,
            size=11,
            palette=palette,
            legend_fontsize=12)
        print('umap saved in figures/umap_w_annotated/'+args.data+ \
            '/seed'+str(args.seed)+'/'+ \
            dimension+'/'+args.id+'-cellytpe-annotated-umap.png')
    return


def scvi_reg_plot(eval_adata_dic, args, model_dic):
    """Plot the predicted vs. ground truth values using scvi function.
    
    Args:
        eval_adata_dic (dict): Dictionary containing the eval_adata for each model.
        args (argparse.Namespace): Arguments passed to the script.
        model_dic (dict): Dictionary containing the model for each method."""

    for model_name, eval_adata in eval_adata_dic.items():
        args.AR = False if model_name == 'Naive' else True
        create_id(args)

        model = model_dic[model_name]
        unper_per_adata = eval_adata[eval_adata.obs['condition'] != model_name]
        
        sc.tl.rank_genes_groups(unper_per_adata,
                                groupby="condition",
                                method="wilcoxon")
        diff_genes = unper_per_adata.uns["rank_genes_groups"]["names"][args.adata_label_per]

        r2_value = model.reg_mean_plot(
            eval_adata,
            axis_keys={"x": model_name, "y": args.adata_label_per},
            gene_list=diff_genes[:10],
            top_100_genes= diff_genes[:100],
            labels={"x": "Predicted","y": "Ground Truth"},
            path_to_save=root+'figures/test/'+args.data+'/seed'+str(args.seed)+'/reg/'+args.id+'_'+model_name+'_reg_scvi.png',
            legend=False,
            title='Test: '+'-'.join(args.test_data)
        )
        test_metric.loc[test_metric.shape[0]] = \
            [args.test_id, args.test_data[0], args.AR, args.variable_con, args.con_percent,
             args.in_dist_group, 'R2 (scvi)', round(r2_value[0], 3), args.seed]
        test_metric.loc[test_metric.shape[0]] = \
            [args.test_id, args.test_data[0], args.AR, args.variable_con, args.con_percent,
             args.in_dist_group, '100 DEG R2 (scvi)', round(r2_value[1], 3),
             args.seed]
        
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


def plot_pc_variance_explained(adata_all, args):
    """Print the variance explained by each principal component.
    
    Args:
        adata_all (anndata.AnnData): Anndata object containing the data.
        args (argparse.Namespace): Arguments passed to the script."""

    model = PCA()
    # Fit pca on all data including control, real stimulated, and model predictions
    model.fit(convert_matrix_to_tensor(adata_all.X))

    # Extract explained variance and cumulative explained variance
    explained_variance = model.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(explained_variance) + 1), explained_variance,
             marker='o', linestyle='-', color='b')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.title('Scree Plot: Variance Explained by Each Principal Component')

    # Plotting the cumulative explained variance
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
             marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Variance Explained')
    plt.title('Cumulative Variance Explained by Principal Components')

    plt.tight_layout()
    title = args.data+'-Test'+'_'.join(args.test_data)+'-conp'+str(args.con_percent)+ \
                '-group'+str(args.group_percent)
    if args.in_dist_test_group != '':
        title = title+'-in_dist_test'+args.in_dist_test_group
    if args.in_dist_group != '':
        title = title+'-in_dist'+args.in_dist_group
    plt.savefig(root+'figures/test/'+args.data+'/seed'+str(args.seed)+
                "/pca/"+title+'_all_data_pca_variance.png')


    # Plotting the explained variance for first 20 principal component
    model = pca.pca(n_components=20)
    # pca.fit(convert_matrix_to_tensor(adata_all.X))
    results = model.fit_transform(convert_matrix_to_tensor(adata_all.X))

    # Extract explained variance and cumulative explained variance
    explained_variance = model.results['variance_ratio'] #pca.explained_variance_ratio_
    cumulative_variance = model.results['explained_var'] #np.cumsum(explained_variance)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, 
             marker='o', linestyle='-', color='b')
    plt.xlabel('Frist 20 Principal Component')
    plt.ylabel('Variance Explained')
    plt.title('Variance Explained by Each Principal Component')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
             marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Variance Explained For First 20 PCs')
    plt.title('Cumulative Variance Explained by Principal Components')

    plt.tight_layout()
    title = args.data+'-Test'+'_'.join(args.test_data)+'-conp'+str(args.con_percent)+ \
                '-groupp'+str(args.group_percent)
    if args.in_dist_test_group != '':
        title = title+'-in_dist_test'+args.in_dist_test_group
    if args.in_dist_group != '':
        title = title+'-in_dist'+args.in_dist_group
    plt.savefig(root+'figures/test/'+args.data+'/seed'+str(args.seed)+
                "/pca/"+title+'_all_data_20pc_variance.png')

    return


def compare_edit_distance(adata, args):
    """Compare the edit distance between the real stimulated and model predictions.
    
    Args:
        adata (anndata.AnnData): Anndata object containing the data.
        args (argparse.Namespace): Arguments passed to the script."""

    estats = scperturb.edist(adata, obs_key='condition')
    print('estats')
    print(estats)
    
    for model_name in ['Naive', 'AR']:
        args.AR = False if model_name == 'Naive' else True
        create_id(args)
        test_metric.loc[test_metric.shape[0]] = \
            [args.test_id, args.test_data[0], args.AR, args.variable_con, args.con_percent,
             args.in_dist_group, 'E-distance',
             estats[model_name][args.adata_label_per], args.seed]

    return


def visualize_latent_space(model_dic, args, adata, data_type):
    """Visualize the latent space of the models.
    
    Args:
        model_dic (dict): Dictionary containing the model for each method.
        args (argparse.Namespace): Arguments passed to the script.
        adata (anndata.AnnData): Anndata object containing the data.
        data_type (str): Type of the data, either train or test."""

    if not os.path.isdir('figures/umap/'+args.data+'/seed'+str(args.seed)+'/'):
        os.makedirs('figures/umap/'+args.data+'/seed'+str(args.seed)+'/')

    for model_name, model in model_dic.items():
        if model_name in ['Naive', 'AR']:
            
            train_adata, test_adata, unper_test_adata, per_test_adata = \
                get_train_test_adata(args, adata)
            
            if data_type == 'train':
                new_adata = train_adata
            elif data_type == 'test':
                new_adata = test_adata.copy()
                if args.variable_con and args.con_percent == 0.0:
                    new_adata.obs[args.adata_label_cell] = \
                        train_adata.obs[args.adata_label_cell].unique()[0]

            latent_X = model.get_latent_representation(new_adata)
            
            if data_type=='test':

                if args.variable_con and args.con_percent==0.0:
                    new_adata.obs[args.adata_label_cell] = \
                        test_adata.obs[args.adata_label_cell].unique()[0]

                if args.in_dist_group != '':
                    new_adata.obs[args.adata_label_cell] = args.in_dist_group

            latent_adata = sc.AnnData(X=latent_X, obs=new_adata.obs.copy())
            # replace the condition with ['Control', 'Stimhlated']
            latent_adata_unper = latent_adata[latent_adata.obs.condition == \
                args.adata_label_unper]
            latent_adata_unper.obs.condition = 'Control'

            latent_adata_per = latent_adata[latent_adata.obs.condition == \
                args.adata_label_per]
            latent_adata_per.obs.condition = 'Stimulated'
            
            latent_adata = latent_adata_unper.concatenate(latent_adata_per)
                
            assert latent_adata_unper.shape[0] + latent_adata_per.shape[0] == \
                latent_adata.shape[0]
            print('latent_adata')
            print(latent_adata.obs.condition.unique())
            
            palette = {'Stimulated':'darkorange',
                       'Control':'dodgerblue'}

            set_seed(args.seed)
            sc.pp.neighbors(latent_adata)
            sc.set_figure_params(dpi=600, format='png')
            sc.tl.umap(latent_adata)
            sc.pl.umap(latent_adata, 
                       color=['condition'], 
                       wspace=0.4, 
                       frameon=False,
                       save='/'+args.data+'/seed'+str(args.seed)+'/'+ \
                            args.data+'-test'+args.test_data[0]+'-'+model_name+ \
                            '-variablecon'+str(args.variable_con)+'-conpercent'+ \
                            str(args.con_percent)+'-seed'+str(args.seed)+'-'+ \
                            data_type+'-condition-umap.png',
                       show=False,
                       title=[''],
                       palette=palette)
            
            
            # sc.pl.umap(latent_adata, 
            #            color=[args.adata_label_cell], 
            #            wspace=0.4, 
            #            frameon=False,
            #            save='/'+args.data+'/seed'+str(args.seed)+'/'+ \
            #                 args.data+'-test'+args.test_data[0]+'-'+model_name+ \
            #                 '-variablecon'+str(args.variable_con)+'-conpercent'+ \
            #                 str(args.con_percent)+'-'+data_type+'-celltype-umap.pdf',
            #            show=False,
            #            title=[''])
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
        args.test_id = args.test_id+'-indistTrue'
    return


def test(args, adata=None, model_dic={}):
    """Test the model.
    
    Args:
        args (argparse.Namespace): Arguments passed to the script.
        adata (anndata.AnnData, optional): Anndata object containing the data.
        model_dic (dict, optional): Dictionary containing the model for each method.
        test_label_con (list, optional): List of control cell labels.
        test_label_stim (list, optional): List of stimulated cell labels."""

    global root
    eval_adata_dic = {}

    train_adata, test_adata, unper_test_adata, per_test_adata = \
        get_train_test_adata(args, adata)
    create_test_id(args)
    
    unper_test_adata.X = convert_matrix_to_tensor(unper_test_adata.X)
    per_test_adata.X = convert_matrix_to_tensor(per_test_adata.X)
    all_test_adata = per_test_adata.concatenate(unper_test_adata)
    all_adata = train_adata.concatenate(test_adata)
    
    assert set(unper_test_adata.obs[args.adata_label_cell]) == \
        set(per_test_adata.obs[args.adata_label_cell])
        
        
    # # plot umap plots annotated by AR weights
    # if args.plot_umap_annotated_with_w:
    #     if 'AR' in args.model:
    #         args.AR = True
    #         create_id(args)
    #         model = load_eval_adata(args, adata, 'AR')
    #         plot_umap_with_weight_annotation(all_adata, args, model)
    #         plot_umap_with_weight_annotation(all_adata, args, model, 'latent')
    #         return
        
    # extract DEGs based on resampling weights
    # if (args.degs_extraction_based_on_resampling_w) & (args.tracked_epoch=='best'):
    #     if 'AR' in args.model:
    #         args.AR = True
    #         create_id(args)
    #         model = load_eval_adata(args, adata, 'AR')
    #         extract_DEGs_based_on_resampling_weights(all_adata, args, model)
    #         return

    # create two dictionaries to store the model and eval_adata for each method
    models = args.model.split(',')
    for model_name in models:
        args.AR = False if model_name == 'Naive' else True
        create_id(args)
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
    print('eval_adata_dic keys: ')
    print(eval_adata_dic.keys())
    print('model_dic keys: ')
    print(model_dic.keys())
        
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
    
    ## plot scvi regression plot
    # scvi_reg_plot(eval_adata_dic, args, model_dic)
    
    # plot PCA for all test samples together
    # if args.ood:
    #     pca_title = args.data+'-Test'+'_'.join(args.test_data)
    # else:
    #     pca_title = args.data+'-OODFalse'
    # if args.in_dist_group != '':
    #     pca_title = pca_title+'-indist'+args.in_dist_group
        
    # plot_pca(test_adata, args, pca_title)
    
    # Plot variance explained of test data
    # plot_pc_variance_explained(all_test_adata, args)
    
    # caluclate edit distance
    # compare_edit_distance(all_test_adata, args)
    
    # calculate cosine similarity
    compare_cosine_similarity(all_test_adata, args, DEG_idx_dic[20])
    
    # visualize latent space
    # visualize_latent_space(model_dic, args, adata, 'train')
    visualize_latent_space(model_dic, args, adata, 'test')

    # R2 between differences of sitmulated and control
    calculate_diff_corr(all_test_adata, args, DEG_idx_dic[20])

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


def test_scvi(args, train_adata, test_adata):
    print('entered test_scvi')
    
    # all_adata = train_adata.concatenate(test_adata)
    all_adata = train_adata.copy()

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
            model = scvi.model.SCVI.load(path_to_model_folder, all_adata)
            # plot_umap_with_weight_annotation(all_adata, args, model, 'latent')
            plot_umap_with_weight_annotation(all_adata, args, model, 'high_dim')
            return
    
    
    return