from utils import set_seed
from geosketch import gs
import scanpy as sc
import numpy as np
import argparse
import time
import os



def select_hvg(
    adata,
    hvg_count,
    original_train_adata_standard,
    original_train_adata_AR=None,
):
    """Select hvg from training data.
    
    Args:
        adata (AnnData): AnnData object containing the data
        hvg_count (int): number of hvg to select
        original_train_adata_standard (AnnData): AnnData object containing the original training data
        original_train_adata_AR (AnnData): AnnData object containing the original training data with AR
        
    Returns:
        adata (AnnData): AnnData object containing the data with the hvg selected
    """
    # select the genes from the original train_adata file
    gene_list = original_train_adata_standard.var_names.tolist()
    assert len(gene_list) == hvg_count
    adata = adata[:, gene_list]
    assert adata.shape[1] == original_train_adata_standard.shape[1]
    assert all(adata.var_names == original_train_adata_standard.var_names)
    # assert all(adata.var_names == original_train_adata_AR.var_names)

    # check if set of all cells in original_train_adata_AR is a subset of the set of all cells in the training data adata
    # assert set(original_train_adata_AR.obs_names.tolist()) <= set(adata.obs_names.tolist())

    return adata


def balance_data_class_balancing(
    adata,
    seed,
    obs_class_label,
):
    """Balance the data using class balancing.
    
    Args:
        adata (AnnData): AnnData object containing the data
        seed (int): seed for random number generator
        obs_class_label (str): obs class label to check
    """
    set_seed(seed)

    # get the unique values of the obs_class_label
    unique_values = adata.obs[obs_class_label].unique()
    sample_count_per_class = int(adata.shape[0] / len(unique_values))

    all_indices = []
    for value in unique_values:
        # get the indices of the cells with the obs_class_label equal to value
        mask = adata.obs[obs_class_label] == value
        indices = np.where(mask)[0]
        # sample from the indices with replacement
        # check if the number of cells in the class is greater than the sample_count_per_class, if so, sample from the class without replacement
        if len(indices) >= sample_count_per_class:
            sampled_indices = np.random.choice(indices, size=sample_count_per_class, replace=False)  
        else:
            sampled_indices = np.random.choice(indices, size=sample_count_per_class, replace=True)  
        
        all_indices.extend(sampled_indices)

    # check if balanced_adata.shape[0] != adata.shape[0], count the difference and randomly sample from the difference to make the number of cells equal to the original number of cells
    if len(all_indices) != adata.shape[0]:
        difference = adata.shape[0] - len(all_indices)
        sampled_indices = np.random.choice(adata.shape[0], size=difference, replace=True)
        all_indices.extend(sampled_indices)

    balanced_adata = adata[np.array(all_indices)].copy()
    assert balanced_adata.shape[0] == adata.shape[0]
    # assert there are the same number of classes as it was in the original adata
    assert len(balanced_adata.obs[obs_class_label].unique()) == len(unique_values)
    # Check the number of cells per class (may not be exactly equal due to rounding)
    for value in unique_values:
        actual_count = balanced_adata.obs[obs_class_label].value_counts()[value]
        print(f"Class {value}: {actual_count} cells (target: {sample_count_per_class})")
        # Note: Due to rounding and the adjustment step, counts may not be exactly equal
        # The assertion is commented out as it may fail due to integer division rounding
    return balanced_adata


def balance_data_geometric_sketching(
    adata,
    seed,
    num_covering_boxes=100,
):
    """Balance the data using geometric sketching.
    
    Args:
        adata (AnnData): AnnData object containing the data with PCA representation
        seed (int): seed for random number generator
        num_covering_boxes (int): number of covering boxes
    
    Returns:
        adata (AnnData): AnnData object containing the geometrically sketched data
    """
    set_seed(seed)

    # derived params
    sketch_size = adata.shape[0]
    num_covering_boxes = int(np.sqrt(adata.shape[0]))

    print("Starting geometric sketching")
    start = time.time()

    sketch_index = gs(
        adata.obsm['X_pca'],
        sketch_size,
        seed=seed,
        k=num_covering_boxes,
        replace=True,
    )

    end = time.time()
    print("Geometric sketching complete")
    print("Time Taken", end - start)

    # Create balanced AnnData object with only the sketched cells
    balanced_adata = adata[sketch_index].copy()
    balanced_adata.obs_names = [
        f"{name}_gs{i}" for i, name in enumerate(balanced_adata.obs_names)
    ]
    print(f"Balanced data shape: {balanced_adata.shape} (original: {adata.shape})")
    
    return balanced_adata, num_covering_boxes


def split_data(
    adata,
    seed,
):
    """Split the data into training and validation sets.
    
    Args:
        adata (AnnData): AnnData object containing the data
        seed (int): seed for random number generator

    Returns:
        train_adata (AnnData): AnnData object containing the training data
        valid_adata (AnnData): AnnData object containing the validation data
    """
    set_seed(seed)

    train_ratio = 0.9
    train_index = np.random.choice(adata.shape[0], int(train_ratio*adata.shape[0]), replace=False)
    train_index = np.array(train_index)
    valid_index = [x for x in range(adata.shape[0]) if x not in train_index]
    valid_index = np.array(valid_index)
    valid_adata = adata[valid_index]
    train_adata = adata[train_index]
    assert train_adata.shape[0] + valid_adata.shape[0] == adata.shape[0]
    return train_adata, valid_adata


def read_and_preprocess_data_for_scvi_sctab(
    atlas_adata, 
    blood_adata,
    atlas_count,
    hvg_count,
    seed,
    root,
    data
):
    """Read and preprocess data for scvi.
    
    Args:
        atlas_adata (AnnData): AnnData object containing the atlas data
        blood_adata (AnnData): AnnData object containing the blood data
        atlas_count (int): number of atlas cells
        hvg_count (int): number of hvg
        seed (int): seed for random number generator
        root (str): root directory

    Returns:
        train_adata (AnnData): AnnData object containing the training data
    """
    all_train_cells = 100000

    # read atlas data
    atlas_adata = sc.read_h5ad(atlas_adata)
    # read blood data
    blood_adata = sc.read_h5ad(blood_adata)

    # randomely select args.atlas_count number of cells from the atlas data
    set_seed(seed)
    index = np.random.choice(atlas_adata.shape[0], atlas_count, replace=False)
    train_adata_altas = atlas_adata[index]
            
    set_seed(seed)
    blood_count = all_train_cells - atlas_count
    index = np.random.choice(blood_adata.shape[0], blood_count, replace=False)
    train_adata_blood = blood_adata[index]
    assert train_adata_blood.shape[0]+train_adata_altas.shape[0] == all_train_cells
    
    # add a new field to the obs of the train_adata_blood
    train_adata_blood.obs['blood_atlas'] = 'Blood'
    train_adata_altas.obs['blood_atlas'] = 'Atlas'

    train_adata_blood.X = train_adata_blood.X.astype('float64')
    train_adata_altas.X = train_adata_altas.X.astype('float64')

    train_adata = train_adata_blood.concatenate(train_adata_altas)
    assert train_adata.shape[0] == all_train_cells

    train_adata.X = train_adata.X.astype('float64')
    assert train_adata.obs[train_adata.obs['blood_atlas'] == 'Atlas'].shape[0] == atlas_count

    # select the same hvg from training data
    path_to_original_train_adata_file = root+"/data/"+data+"/bloodbase_"+ \
        str(atlas_count)+'_atlas'+\
        "_seed"+str(seed)+"_ARTrue"+\
        "_train_adata_"+str(hvg_count)+"_"+str(hvg_count)+"HVGs.h5ad"
    original_train_adata_standard = sc.read_h5ad(path_to_original_train_adata_file)
    # Read AR version by replacing ARFalse with ARTrue
    # path_to_original_train_adata_AR_file = path_to_original_train_adata_file.replace("ARFalse", "ARTrue")
    # original_train_adata_AR = sc.read_h5ad(path_to_original_train_adata_AR_file)
    # assert original_train_adata_standard.shape[1] == original_train_adata_AR.shape[1]

    train_adata = select_hvg(
        train_adata,
        hvg_count,
        original_train_adata_standard
    )

    return train_adata


def create_PCA_representation_for_geometric_sketching(
    adata,
    seed,
):
    """Create PCA representation for geometric sketching.
    
    Args:
        adata (AnnData): AnnData object containing the data (expected to be raw counts)
        seed (int): seed for random number generator

    Returns:
        adata (AnnData): AnnData object containing the data with the PCA representation
    """
    set_seed(seed)

    # Normalize and log-transform
    # Note: Assumes input data is raw counts. If data is already normalized/log-transformed,
    # these steps will be applied again which may not be desired.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata,
              svd_solver="arpack",
              random_state=seed)
    # check if the PCA representation is created
    assert adata.obsm['X_pca'] is not None
    assert adata.obsm['X_pca'].shape[0] == adata.shape[0]
    return adata


def read_and_preprocess_data_for_scgen(
    anndata_file,
    obs_class_label
):
    """Read and preprocess data for scgen.
    
    Args:
        anndata_file (str): path to the anndata file
        obs_class_label (list): obs class label to check

    Returns:
        adata (AnnData): AnnData object containing the data
        class_label (str): class label to check
    """
    adata = sc.read_h5ad(anndata_file)
    # create a new field in the obs of the adata, and assign new category labels based on labels in the obs_class_label list
    adata.obs['class_label'] = adata.obs[obs_class_label[0]]
    for label in obs_class_label[1:]:
        adata.obs['class_label'] += '_'+adata.obs[label]
    # check if the new field is created
    assert 'class_label' in adata.obs.columns
    # print unique values of the new field
    print(adata.obs['class_label'].unique())
    # print the number of cells per category
    print(adata.obs['class_label'].value_counts())

    return adata, "class_label"



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str, nargs='?',
                        default="scvi",
                        help="model: scvi, scgen")
    parser.add_argument("--data", type=str, nargs='?',
                        default="sctab",
                        help="data: sctab, species")
    parser.add_argument('--anndata_blood_file', default="data/sctab/BaseModel_scTabBloodOnly_seed0_bloodbase_TrainingData.h5ad",
                        help="path to the anndata file")
    parser.add_argument('--anndata_atlas_file', default="data/sctab/BaseModel_scTabAll_seed0_allbase_TrainingData.h5ad",
                        help="path to the anndata file")
    parser.add_argument("--anndata_file", type=str, nargs='?',
                        default="data/species.h5ad",
                        help="path to the anndata file")
    parser.add_argument("--obs_class_label",
                        nargs="+",
                        type=str,
                        required=True,
                        help="obs class label to check")
    parser.add_argument("--training_data_count", type=int, nargs='?',
                        default=100000,
                        help="training data count")
    parser.add_argument("--seed", type=int, nargs='?',
                        default=42,
                        help="seed for random number generator")
    parser.add_argument("--atlas_count", type=int, nargs='?',
                        default=0,
                        help="atlas count")
    parser.add_argument("--hvg_count", type=int, nargs='?',
                        default=2000,
                        help="hvg count")
    parser.add_argument("--balancing_method", type=str, nargs='?',
                        default="class_balancing",
                        help="balancing method: class_balancing, geometric_sketching")
    parser.add_argument("--num_covering_boxes", type=float, nargs='?',
                        default=100,
                        help="number of covering boxes")
    parser.add_argument("--root", type=str, nargs='?',
                        default="/Users/zeinab/Documents/MSR_internship/project/sc-AR-github-repo/sc-AR/",
                        help="root directory")

    # parse and preprocess args
    args = parser.parse_args()
    args.anndata_atlas_file = args.root+args.anndata_atlas_file
    args.anndata_blood_file = args.root+args.anndata_blood_file
    args.anndata_file = args.root+args.anndata_file

    # check if len of obs_class_label is 1
    if len(args.obs_class_label) == 1:
        args.obs_class_label = args.obs_class_label[0]

    ## read and preprocess data
    if args.model == "scvi":
        # create original atlas and blood combination of training data
        adata = read_and_preprocess_data_for_scvi_sctab(
            args.anndata_atlas_file,
            args.anndata_blood_file,
            args.atlas_count,
            args.hvg_count,
            args.seed,
            args.root,
            args.data,
        )
    elif args.model == "scgen":
        train_adata, args.obs_class_label = read_and_preprocess_data_for_scgen(
            args.anndata_file,
            args.obs_class_label
        )
    else:
        raise ValueError("Invalid model")
    
    # balance the data
    if args.balancing_method == "geometric_sketching":
        start_time = time.time()
        # Create PCA representation for geometric sketching (required for this method)
        adata = create_PCA_representation_for_geometric_sketching(
            adata,
            args.seed,
        )
        adata, args.num_covering_boxes = balance_data_geometric_sketching(
            adata,
            args.seed,
            args.num_covering_boxes,
        )
        end_time = time.time()
        print("Time taken to create PCA representation and balance data for geometric sketching: ", end_time - start_time, "seconds")
    elif args.balancing_method == "class_balancing":
        adata = balance_data_class_balancing(
            adata,
            args.seed,
            args.obs_class_label,
        )
    else:
        raise ValueError("Invalid balancing method")


    # split the data into training and validation sets
    if args.model == "scvi":
        train_adata, valid_adata = split_data(
            adata,
            args.seed,
        )

    if args.model == 'scvi':
        # check if args.root+"/data/$data/balanced_data/" exists, if not create it
        output_path = args.root+"/data/"+args.data+"/balanced_data/"+args.balancing_method+"/"+str(args.seed)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # save the balanced train and validation data
        path_to_train_adata_file = output_path+"/bloodbase_"+ \
            str(args.atlas_count)+'_atlas'+\
            "_seed"+str(args.seed)+\
            "_"+str(args.hvg_count)+"HVGs"+\
            "_"+args.balancing_method
            
        if args.balancing_method == "geometric_sketching":
            "_"+str(args.num_covering_boxes)+"covering_boxes"
            
        path_to_train_adata_file = path_to_train_adata_file+"_train_adata.h5ad"
        train_adata.write(path_to_train_adata_file)

        path_to_valid_adata_file = output_path+"/bloodbase_"+ \
            str(args.atlas_count)+'_atlas'+\
            "_seed"+str(args.seed)+\
            "_"+str(args.hvg_count)+"HVGs"+\
            "_"+args.balancing_method
            
        if args.balancing_method == "geometric_sketching":
            "_"+str(args.num_covering_boxes)+"covering_boxes"
        path_to_valid_adata_file = path_to_valid_adata_file+"_valid_adata.h5ad"
        valid_adata.write(path_to_valid_adata_file)
    
    elif args.model == 'scgen':
        # check if args.root+"/data/$data/balanced_data/" exists, if not create it
        output_path = args.root+"/data/"+args.data+"/balanced_data/"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # save the balanced train and validation data
        path_to_train_adata_file = output_path+"/"+args.data+"_seed"+str(args.seed)+"_"+args.balancing_method
        
        if args.balancing_method == "geometric_sketching":
            path_to_train_adata_file = path_to_train_adata_file+"_"+str(args.num_covering_boxes)+"covering_boxes"
        path_to_train_adata_file = path_to_train_adata_file+"_adata.h5ad"
        train_adata.write(path_to_train_adata_file)
