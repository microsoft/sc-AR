import plotly.graph_objects as go
import matplotlib.pyplot as plt
import scipy.spatial as sp
import seaborn.objects as so
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import scanpy as sc
import pandas as pd
import scperturb
import argparse
import random
from datetime import datetime
import wandb
import torch
import os


def set_seed(seed):
    """Set seed for reproducibility.
    
    Args:
        seed (int): seed for reproducibility"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return


def str2bool(v):
    """Convert string to boolean.
    
    Args:
        v (str): string to convert to boolean"""

    if isinstance(v, bool):
        return v
    if v.lower() in ('True', 'TRUE', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('False', 'FALSE', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

    return
    
    
def create_id(args):
    """Create id for the experiment based on the input arguments.
    
    Args:
        args (argparse): input arguments"""
    
    test_data = ''
    if args.ood:
        test_data = '-'.join(args.test_data)

    if args.model_name == 'scgen':
        args.id = args.data+ \
            '-test'+test_data + \
            '-AR'+str(args.AR)+ \
            '-ood'+str(args.ood)+ \
            '-varcon'+str(args.variable_con)+ \
            '-per'+str(args.con_percent)+ \
            '-lr'+str(args.lr)+ \
            '-wd'+str(args.weight_decay)+ \
            '-bs'+str(args.batch_size)+ \
            '-ldim'+str(args.latent_dim)+ \
            '-epoch'+str(args.num_epoch)+ \
            '-seed'+str(args.seed)
            
    if args.model_name == 'scvi':
        if args.data == 'sctab':
            args.id = 'bloodbase-'+str(args.atlas_count)+'atlas'
        elif 'idx_' in args.data and args.sctab_subset != 'normal':
            args.id += args.data+'-'+str(args.sctab_subset)


        args.id = args.id + \
            '-AR'+str(args.AR)[0]+ \
            '-ood'+str(args.ood)[0]+ \
            '-varcon'+str(args.variable_con)[0]+ \
            '-per'+str(args.con_percent)+ \
            '-lr'+str(args.lr)+ \
            '-wd'+str(args.weight_decay)+ \
            '-bs'+str(args.batch_size)+ \
            '-ldim'+str(args.latent_dim)+ \
            '-epoch'+str(args.num_epoch)+'-scvi'

        if args.check_scvi:
            args.id = args.id+'-check'
        
        if args.cell_count != None:
            args.id = args.id+'-cell'+str(args.cell_count)
            
        if args.hvg_count != None:
            args.id = args.id+'-hvg'+str(args.hvg_count)
        args.id = args.id+'-s'+str(args.seed)

    return


def create_optimizers(model, args):
    """Create and return optimizer.

    Args_
        model (nn.Module): Neural network model to train.
        optimizer (torch.optim): the optimizer function to be used during
        training.
        args (_type_): _description_

    Returns_
        _type_: _description_

    """
    if args.optimizer.lower() == 'sgd':
        print("SGD Initialized")
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.beta1,
            weight_decay=args.weight_decay,
            nesterov=False)

    elif args.optimizer.lower() == 'adam':
        print("Adam Initialized")
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999))

    elif args.optimizer.lower() == 'radam':
        print("RAdam Initialized")
        optimizer = torch.optim.RAdam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,)

    elif args.optimizer.lower() == 'adamw':
        print("AdamW Initialized")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,)

    return optimizer


def create_directory(args):
    """Create the directory to store the results.
    
    Args:
        args (argparse.Namespace): Arguments passed to the script."""
    
    if args.train:
        if not os.path.exists(args.root+'/result/train/'+str(args.data)+'/'):
            os.makedirs(args.root+'/result/train/'+str(args.data)+'/')

        if not os.path.exists(args.root+"/saved_models/"+args.data +
                              "/seed"+str(args.seed)+'/'):
            os.makedirs(args.root+"/saved_models/"+args.data +
                        "/seed"+str(args.seed)+'/')
            
        if not os.path.exists(args.root+"/saved_models/"+args.data +
                              "/seed"+str(args.seed)+'/config/'):
            os.makedirs(args.root+"/saved_models/"+args.data +
                        "/seed"+str(args.seed)+'/config/')

    if args.test:
        if not os.path.isdir(args.root+'/result/test/'+args.data):
            os.makedirs(args.root+'/result/test/'+args.data)

    return


def relatedness_score(adata, args, pca_performed = True, metric='cosine'):
    """Computes the relatedeness between celltypes using cosine/E-distance distance in PCA space
    
    Args:
        adata (AnnData): AnnData object containing relevant count information with celltype
            and batch observations.
        pca_performed (bool): True or False value indicating whether PCA decomposition steps
            have been performed already for AnnData object. Default is True.
    """
    # Perform PCA if not already performed
    if pca_performed is False:
        # sc.pp.normalize_total(adata, target_sum=1e4)
        # sc.pp.log1p(adata)
        # sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2500)
        sc.pp.pca(adata, n_comps=args.latent_dim)
        
    print("PCA performed")
        
    # Get the batch and celltype information from AnnData object
    condition_vals = np.unique(adata.obs["condition"].__array__())
    if len(condition_vals) > 1:
        raise ValueError("More than one batch found in AnnData object")
    condition = condition_vals[0]
    celltypes = np.unique(adata.obs[args.adata_label_cell].__array__())
    
    # Utilize the distance between the average PCA embedding for celltype i and 
    # average PCA embedding for celltype j
    pca_top = adata.obsm["X_pca"]#[:, 0:20]
    top_pc_weights = adata.uns["pca"]["variance_ratio"]#[:, 0:20]
    celltype_is = []
    celltype_js = []
    pca_dists = []
    for celltype_i in celltypes:
        for celltype_j in celltypes:
            celltype_is.append(celltype_i)
            celltype_js.append(celltype_j)
            pca_celltype_i = pca_top[
                adata.obs[args.adata_label_cell] == celltype_i
            ]
            pca_celltype_j = pca_top[
                adata.obs[args.adata_label_cell] == celltype_j
            ]
            pca_celltype_i_avg = np.sum(pca_celltype_i, axis = 0)/len(pca_celltype_i)
            pca_celltype_j_avg = np.sum(pca_celltype_j, axis = 0)/len(pca_celltype_j)
            if metric == 'cosine':
                pca_dist = sp.distance.cosine(
                    pca_celltype_i_avg,
                    pca_celltype_j_avg,
                    w = top_pc_weights
                )
                pca_dists.append(pca_dist)
            elif metric == 'e-distance':
                # select subset of adata with adata.obs[args.adata_cell_label] == celltype_i or celltype_j
                pca_celltype_i_j = pca_top[
                    (adata.obs[args.adata_label_cell] == celltype_i) | 
                    (adata.obs[args.adata_label_cell] == celltype_j)]
                latent_adata = sc.AnnData(
                    X=pca_celltype_i_j,
                    obs=adata[
                        (adata.obs[args.adata_label_cell] == celltype_i) |
                        (adata.obs[args.adata_label_cell] == celltype_j)].obs.copy()
                    )
                pca_dist = scperturb.edist(latent_adata, 
                                           obs_key=args.adata_label_cell)
                pca_dists.append(pca_dist[celltype_i][celltype_j])
            
            # pca_dists.append(pca_dist)
            
    # Gather the cosine distance results in a dataframe and return  
    if metric == 'cosine':
        dist_results_df = pd.DataFrame({
            "Cell Group 1": celltype_is,
            "Cell Group 2": celltype_js,
            "PCA cosine dist": pca_dists,
            "Condition": condition
        })
    elif metric == 'e-distance':
        dist_results_df = pd.DataFrame({
            "Cell Group 1": celltype_is,
            "Cell Group 2": celltype_js,
            "PCA e-distance": pca_dists,
            "Condition": condition
        })
    
    dist_results_df = dist_results_df.pivot(
        index = "Cell Group 1",
        columns = "Cell Group 2",
        values = "PCA "+metric+ (" dist" if metric == 'cosine' else '')
        )
    print(dist_results_df)
    # add labels to heatmap
    # path = os.getcwd()
    # root = os.path.abspath(os.path.join(path, os.pardir))
    address = args.root+'/figures/qq/'+args.data+'/'
    print("Saving heatmap to: ", address+metric+'_heatmap_'+args.data+'_'+condition+'.png')
    plt.clf()
    sns.heatmap(dist_results_df, cmap = "Blues", annot = True, fmt = ".2f")
    # add title
    plt.title("PCA "+metric+" distance between cell groups in "+args.data+" "+condition)
    plt.savefig(address+metric+'_heatmap_'+args.data+'_'+condition+'.png', dpi = 300,
                bbox_inches = "tight")
    return dist_results_df


def plot_sample_count(adata, args, path):
    
    # print adata with count number of each cell group a and condition
    print(adata.obs.groupby(['condition', args.adata_label_cell]).size())
    
    count_df = pd.DataFrame(columns=['Cell group', 'Condition', 'Number of cells'])
    per = adata[adata.obs['condition'] == args.adata_label_per]
    unper = adata[adata.obs['condition'] == args.adata_label_unper]
    y_per = list(set(per.obs[args.adata_label_cell].tolist()))
    y_unper = list(set(unper.obs[args.adata_label_cell].tolist()))
    # replace 'Enterocyte.Progenitor' with 'Ent.Progenitor'
    y_per = [x if x != 'Enterocyte.Progenitor' else 'Ent.Progenitor' for x in y_per]
    y_unper = [x if x != 'Enterocyte.Progenitor' else 'Ent.Progenitor' for x in y_unper]
    
    print(y_per)
    print(y_unper)
    
    # sort the cell groups
    y_per.sort()
    y_unper.sort()

    x_per = list()
    for group in y_per:
        original_group = group
        if group =='Ent.Progenitor':
            original_group = 'Enterocyte.Progenitor'
        x_per.append(per.obs[args.adata_label_cell].tolist().count(original_group))
        count_df.loc[count_df.shape[0]] = {'Cell group': group, 'Condition': 'Perturbed', 'Number of cells': per.obs[args.adata_label_cell].tolist().count(original_group)}
    x_unper = list()
    for group in y_unper:
        original_group = group
        if group =='Ent.Progenitor':
            original_group = 'Enterocyte.Progenitor'
        x_unper.append(unper.obs[args.adata_label_cell].tolist().count(original_group))
        count_df.loc[count_df.shape[0]] = {'Cell group': group, 'Condition': 'Control', 'Number of cells': unper.obs[args.adata_label_cell].tolist().count(original_group)}

    colors = {'Perturbed':'darkorange',
              'Control':'dodgerblue'}

    # create stacked barplot for the number of cells in each cell group with seaborn
    plt.clf()
    fig, ax = plt.subplots()
    assert y_per == y_unper
    cell_groups = tuple(y_unper)
    print(cell_groups)
    counts = {
        "Control": np.array(x_unper),
        "Perturbed": np.array(x_per),
    }
    width = 0.5

    fig, ax = plt.subplots()
    bottom = np.zeros(len(cell_groups))

    for boolean, count in counts.items():
        p = ax.bar(cell_groups, count, width, label=boolean, bottom=bottom, color=colors[boolean])
        bottom += count
        
    # adjust font size
    ax.set_ylabel('Number of cells', fontsize = 17)
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=13, labelrotation=40)
    ax.tick_params(axis='y', labelsize=12)
    # set font size of legend

    # locate legend outside of the plot
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=13)
    
    plt.savefig(path+args.data+'-sample-count.png', dpi=600, bbox_inches="tight")
    return


def plot_sample_count_scvi(adata, args, path):
    
    # print adata with count number of each cell group a and condition
    print(adata.obs.groupby(['cell_type']).size())
    count_df = adata.obs.groupby(['cell_type']).size()
    
    count_df.columns = ['Cell group', 'Number of cells']
    # plot a barplot for the number of cells in each cell group
    plt.clf()
    fig, ax = plt.subplots()
    cell_groups = tuple(count_df.index)
    counts = count_df.values
    width = 0.5
    p = ax.bar(cell_groups, counts, width, color='dodgerblue')
        
    # adjust font size
    ax.set_ylabel('Number of cells', fontsize = 17)
    ax.set_xlabel('Cell group', fontsize = 17)
    # ax.tick_params(axis='x', labelsize=12, labelrotation=20)
    # ax.tick_params(axis='y', labelsize=16)
    # remove axis labels
    ax.set_xticklabels([])
    ax.legend()
    plt.savefig(path+args.data+'-sample-count.png', dpi=600, bbox_inches="tight")
    return


def quality_control_adata(adata, args):
    """Performs quality control on AnnData object and outputs relevant plots and statistics
    
    Args:
        adata (AnnData): AnnData object containing relevant count information with celltype
            and condition observations.
        args (argparse.ArgumentParser): It is a container for argument specifications.
    """
    print(args.data+' quality control')
    if args.data != 'lps-hpoly':
        adata.X = adata.X.toarray()

    path = os.getcwd()
    root = os.path.abspath(os.path.join(path, os.pardir))

    address = 'figures/qq/'+args.data+'/'
    if not os.path.isdir(address):
        os.makedirs(address)
        
    if args.data == 'species':
        # capitilize the first letter of the cell type
        adata.obs[args.adata_label_cell] = adata.obs[args.adata_label_cell].str.capitalize()

    if args.model_name == 'scvi':
        plot_sample_count_scvi(adata, args, 'figures/qq/'+args.data+'/')
        return
    else:
        plot_sample_count(adata, args, 'figures/qq/'+args.data+'/')

    # Obtain PCA of adata.X in args.latent_dim dimension and plot Umap of PCs
    # sc.pp.pca(adata, n_comps=args.latent_dim)
    # latent_adata = sc.AnnData(X=adata.obsm['X_pca'],
    #                           obs=adata.obs.copy())
    # unper_adata = adata[adata.obs['condition'] == args.adata_label_unper]
    # per_adata = adata[adata.obs['condition'] == args.adata_label_per]
    if args.data == 'species':
        # capitilize the first letter of the cell type
        adata.obs[args.adata_label_cell] = adata.obs[args.adata_label_cell].str.capitalize()

    # replace args.adata_label_unper in condition with 'Control' and args.adata_label_per with 'Perturbed'
    adata.obs['condition'] = adata.obs['condition'].replace(args.adata_label_unper, 'Control')
    adata.obs['condition'] = adata.obs['condition'].replace(args.adata_label_per, 'Perturbed')

    adata.obs[args.adata_label_cell] = adata.obs[args.adata_label_cell].replace('Enterocyte.Progenitor', 'Enterocyte.P')
    set_seed(args.seed)
    sc.pp.neighbors(adata)
    sc.set_figure_params(dpi=600, format='png', fontsize=13)
    
    palette = {'Perturbed':'darkorange',
              'Control':'dodgerblue'}
    
    sc.tl.umap(adata)
    sc.pl.umap(
        adata,
        color=['condition'],
        wspace=0.4, frameon=False,
        save='-'+args.data+'-conditions',
        title='',
        legend_fontsize=16,
        palette=palette)
    sc.pl.umap(
        adata,
        color=[args.adata_label_cell],
        wspace=0.4, frameon=False,
        save='-'+args.data+'-celltypes',
        title='',
        legend_fontsize=16)
    
    return
    # modify font size of labels and title
    
    # plt.rc('axes', titlesize=16)
    # plt.rc('axes', labelsize=18)
    # plt.rc('xtick', labelsize=18)
    # plt.rc('ytick', labelsize=18)
    # plt.rc('legend', fontsize=18)
    # plt.rc('figure', titlesize=18)
    

    # Calculate similarity matric for each cell type and condition
    # and plot heatmap
    # condition_values = np.unique(adata.obs["condition"].__array__())
    # for condition in condition_values:
    #     adata_copy = adata[adata.obs['condition'] == condition]
    #     print(adata_copy)
    #     relatedness_score(adata_copy, args, pca_performed = True, metric='cosine')
    #     relatedness_score(adata_copy, args, pca_performed = True, metric='e-distance')

    for celltype in np.unique(adata.obs[args.adata_label_cell].__array__()):
        cell_adata = adata[adata.obs[args.adata_label_cell] == celltype]       
        dist = scperturb.edist(cell_adata,
                               obs_key='condition')
        print(celltype+" e-distance: "+
              str(dist[args.adata_label_per][args.adata_label_unper]))

        per = cell_adata[cell_adata.obs['condition'] == args.adata_label_per].X
        unper = cell_adata[cell_adata.obs['condition'] == args.adata_label_unper].X
        per_avg = np.sum(per, axis = 0)/len(per)
        unper_avg = np.sum(unper, axis = 0)/len(unper)
        print(per_avg.shape)
        print(unper_avg.shape)
        print()

        pca_dist = sp.distance.cosine(
            per_avg, unper_avg)
        print(celltype+" cosine distance: "+str(pca_dist))

    return

def create_id_old(args, percent):
    """Create id for the experiment based on the input arguments.
    
    Args:
        args (argparse): input arguments"""
    
    if args.ood:
        id = args.data+'-test-'+'-'.join(args.test_data) + \
            '-AR'+str(args.AR)+'-varconTrue-per'+str(percent)+'-vargroupNone-per1.0-seed'+str(args.seed)

    return id


def updateـold_model_name(args):
    """Update the name of the model.
    
    Args:
        args (argparse.ArgumentParser): It is a container for argument specifications."""
        
    path = os.getcwd()
    root = os.path.abspath(os.path.join(path, os.pardir))
    
    path = args.root+"/saved_models/"+args.data +\
        "/seed"+str(args.seed)+'/'

    if not os.path.exists(path):
        os.makedirs(path)
        
    if not os.path.exists(path+'/config/'):
        os.makedirs(path+'/config/')
    
    for percent in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        
        args.variable_con = True
        args.con_percent = percent
        
        id = create_id_old(args, percent)
        create_id(args)
 
        wandb_id = "MSR-"+"{:%m%d}".format(datetime.now())+'-'+args.id
        wandb.init(project=wandb_id,
                   config=args,
                   dir=args.root+'/',)
        config_dict = wandb.config._items

        old_model_path = args.root+"/saved_models_old/"+args.data +\
            "/seed"+str(args.seed)+'/'+id+"_best.pt"
            
        updated_model_path = args.root+"/saved_models/"+args.data +\
            "/seed"+str(args.seed)+'/'+args.id+"-best.pt"
            
        print("cp -r "+old_model_path+" "+updated_model_path)
        # print args
        for key, val in vars(args).items():
            print("{:16} {}".format(key, val))
            
        # copy the model to the new directory
        os.system("cp -r "+old_model_path+" "+updated_model_path)
        
        # Save the configuration to a local file
        with open(args.root+"/saved_models/"+args.data +
                  "/seed"+str(args.seed)+'/config/'+
                  args.id+"_wandb_config.yaml", "w") as f:
            for key, val in vars(args).items():
                f.write(f"{key}: {val}\n")

    return