from numpy.random import RandomState
from scripts.utils import set_seed
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import random
import os



def load_pbmc(args):
    """Load PBMC data.
    
    Args:
        args (argparse): arguments
        
    Returns:
        adata (AnnData): AnnData object containing the data"""

    adata = sc.read_h5ad(args.root+"/data/pbmc.h5ad")
    args.adata_label_cell = 'cell_type'
    args.adata_label_per = 'stimulated'
    args.adata_label_unper = 'control'
    print(adata.obs['cell_type'].value_counts())

    return adata


def load_species(args):
    """Load species data.
    
    Args:
        args (argparse): arguments
        
    Returns:
        adata (AnnData): AnnData object containing the data"""

    adata = sc.read_h5ad(args.root+"/data/species.h5ad")
    args.adata_label_cell = 'species'
    args.adata_label_per = 'LPS6'
    args.adata_label_unper = 'unst'

    return adata


def load_hpoly(args):
    """Load hpoly data.
    
    Args:
        args (argparse): arguments
        
    Returns:
        adata (AnnData): AnnData object containing the data"""

    adata = sc.read_h5ad(args.root+"/data/hpoly.h5ad")
    args.adata_label_cell = 'cell_label'
    args.adata_label_per = 'Hpoly.Day10'
    args.adata_label_unper = 'Control'

    return adata


def split_randomely(args, adata):
    """Randomely split the data into training and testing sets.
    
    Args:
        args (argparse): arguments
        adata (AnnData): AnnData object containing the data
        
    Returns:
        new_adata (AnnData): AnnData object containing the data
        after spliting into training and testing sets"""
    
    adata.obs['original_group'] = adata.obs[args.adata_label_cell]

    set_seed(args.seed)
    test_index = random.sample(range(adata.shape[0]), int(adata.shape[0]/5))
    train_index = [x for x in range(adata.shape[0]) if x not in test_index]
    
    train_adata = adata[train_index]
    test_adata = adata[test_index]
    assert train_adata.shape[0] + test_adata.shape[0] == adata.shape[0]

    train_adata.obs[args.adata_label_cell] = \
        list(map(str.__add__,
                 train_adata.obs[args.adata_label_cell],
                 ['-train']*train_adata.shape[0]))
    test_adata.obs[args.adata_label_cell] = ['test-data']*test_adata.shape[0]
    new_adata = train_adata.concatenate(test_adata)
    
    args.train_data = list(set(train_adata.obs[args.adata_label_cell]))
    args.test_data = list(set(test_adata.obs[args.adata_label_cell]))

    print(new_adata.obs.groupby(['condition', args.adata_label_cell]).size())
    assert train_adata.shape[0] + test_adata.shape[0] == adata.shape[0]
    
    return new_adata


def adjust_training_proportions(args, adata):
    """Adjust the training proportions of the control samples from OOD test group.
    
    Args:
        args (argparse): arguments
        adata (AnnData): AnnData object containing the data"""

    new_adata = adata
    set_seed(args.seed)
    prng = RandomState(args.seed)

    if args.variable_con and args.con_percent != 1.0:

        test_unper_adata = adata[
            (adata.obs[args.adata_label_cell].isin(args.test_data)) &
            (adata.obs['condition'] == args.adata_label_unper)]
        train_adata = adata[~(
            (adata.obs[args.adata_label_cell].isin(args.test_data)) &
            (adata.obs['condition'] == args.adata_label_unper))]
        
        index = prng.randint(0, test_unper_adata.shape[0]-1,
                             size=int(args.con_percent*test_unper_adata.shape[0]))

        new_test_unper_adata = test_unper_adata[index]
        new_adata = train_adata.concatenate(new_test_unper_adata)

    return new_adata


def read_and_preprocess_data_for_scvi_sctab(args):
    """Read and preprocess data for scvi.
    
    Args:
        args (argparse): arguments
        
    Returns:
        train_adata (AnnData): AnnData object containing the training data
        valid_adata (AnnData): AnnData object containing the validation data
        test_adata (AnnData): AnnData object containing the testing data
    """
    all_train_cells = 100000

    # read atlas data
    atlas_adata = sc.read_h5ad(
        args.root+
        "/data/sctab/BaseModel_scTabAll_seed0_allbase_TrainingData.h5ad")
    # read blood data
    blood_adata = sc.read_h5ad(
        args.root+
        "/data/sctab/BaseModel_scTabBloodOnly_seed0_bloodbase_TrainingData.h5ad")

    # randomely select args.atlas_count number of cells from the atlas data
    set_seed(args.seed)
    index = np.random.choice(atlas_adata.shape[0], args.atlas_count, replace=False)
    train_adata_altas = atlas_adata[index]
            
    set_seed(args.seed)
    blood_count = all_train_cells - args.atlas_count
    index = np.random.choice(blood_adata.shape[0], blood_count, replace=False)
    train_adata_blood = blood_adata[index]
    assert train_adata_blood.shape[0]+train_adata_altas.shape[0] == all_train_cells

    # randomely select 10% of the blood data for validation
    set_seed(args.seed)
    valid_index = np.random.choice(
        train_adata_blood.shape[0],
        int(0.1*all_train_cells),
        replace=False)
    train_index = [x for x in range(train_adata_blood.shape[0]) if x not in valid_index]
    # convert train_index to a numpy array
    train_index = np.array(train_index)
    valid_index = np.array(valid_index)
    
    # add a new field to the obs of the train_adata_blood
    train_adata_blood.obs['blood_atlas'] = 'Blood'
    train_adata_altas.obs['blood_atlas'] = 'Atlas'
            
    train_adata = train_adata_blood[train_index].concatenate(train_adata_altas)
    valid_adata = train_adata_blood[valid_index]
    assert train_adata.shape[0]+valid_adata.shape[0] == all_train_cells

    train_adata.X = train_adata.X.astype('float64')
    valid_adata.X = valid_adata.X.astype('float64')

    # assert the number of Atlas cells in the training data is equal to args.atlas_count
    assert train_adata.obs[train_adata.obs['blood_atlas'] == 'Atlas'].shape[0] == args.atlas_count
    
    # select hvg from training data
    sc.pp.highly_variable_genes(
        train_adata,
        n_top_genes=args.hvg_count,
        subset=True,
        flavor='seurat_v3'
    )

    gene_list = train_adata.var_names.tolist()
    # save the genes used for training in a csv file
    path_to_train_adata_gene_file = args.root+"/data/sctab/"+\
        "bloodbase_"+str(args.atlas_count)+'_atlas'+\
        "_seed"+str(args.seed)+"_AR"+str(args.AR)+\
        "_train_adata_"+str(args.hvg_count)+"_gene_names.csv"
            
    # check if the file exists
    if not os.path.exists(path_to_train_adata_gene_file):
        with open(path_to_train_adata_gene_file, 'w') as f:
            for item in gene_list:
                f.write("%s\n" % item)
                
    # save train_adata file
    train_adata.write(
        args.root+"/data/sctab/bloodbase_"+str(args.atlas_count)+'_atlas'+\
        "_seed"+str(args.seed)+"_AR"+str(args.AR)+\
        "_train_adata_"+str(args.hvg_count)+"_"+str(args.hvg_count)+"HVGs.h5ad")

    # select hvg from training data in validation data
    valid_adata = valid_adata[:, gene_list]
    assert train_adata.shape[1] == valid_adata.shape[1]
    assert all(train_adata.var_names == valid_adata.var_names)

    test_adata = None
    if (args.predict or args.test) and (not args.plot_umap_annotated_with_w):

        gene_list_saved = pd.read_csv(
            path_to_train_adata_gene_file, header=None)[0].tolist()
        assert len(gene_list_saved) == args.hvg_count
        # assert if gene_list and gene_list_saved are the same
        assert all(x == y for x, y in zip(gene_list_saved, gene_list)) and \
            len(gene_list_saved) == len(gene_list), "Lists are different"
        
        # load args.test_adata_path file
        test_adata = sc.read_h5ad(args.test_adata_path)
        test_adata.X = test_adata.X.astype('float64')
        
        if 'neurons' in args.test_adata_path.lower():
            # get the names of all genes in the train_adata
            train_gene_names = train_adata.var['feature_id'].tolist()
            print("train_gene_names len: ", len(train_gene_names))
            # assert gene_list_saved and train_gene_names are the same
            assert all(x == y for x, y in zip(gene_list_saved, train_gene_names))
            
            # get the names of all genes in the test_adata
            test_gene_names = test_adata.var.index.tolist()
            test_gene_names = [x.split('.')[0] for x in test_gene_names]
            
            # remove duplicates from test_gene_names
            test_adata.var.index = test_gene_names
            test_adata = test_adata[:, ~test_adata.var.index.duplicated()]
            test_gene_names = test_adata.var.index.tolist()
            
            # get the names of the genes in the train_adata that are not in the test_adata
            not_present_test_gene_names = [
                x for x in train_gene_names if x not in test_gene_names]
            
            # get the index of the genes that are not in the test gene ids in the train gene ids
            not_present_train_gene_idx = [
                i for i, x in enumerate(train_gene_names) if x in not_present_test_gene_names]
    
            not_present_train_gene_idx = not_present_train_gene_idx
            
            full_test_gene_names = []
            neuron_new_X = np.array([])
            neuron_test_common_genes = []
            
            for i in range(len(not_present_train_gene_idx)):

                if i == 0:
                    train_gene_names_subset = train_gene_names[:not_present_train_gene_idx[i]]
                else:
                    train_gene_names_subset = train_gene_names[
                        not_present_train_gene_idx[i-1]+1:not_present_train_gene_idx[i]]
                assert not_present_test_gene_names[i] not in train_gene_names_subset

                test_gene_idx = [i for i, x in enumerate(test_gene_names) if x in train_gene_names_subset]
                assert len(test_gene_idx) == len(train_gene_names_subset)

                print('neuron_new_X shape before appending: ', neuron_new_X.shape)
                if i == 0:
                    neuron_new_X = test_adata.X[:, test_gene_idx].toarray()
                else:
                    neuron_new_X = np.concatenate(
                        [neuron_new_X,
                         test_adata.X[:, test_gene_idx].toarray()],
                        axis=1)
                print('neuron_new_X shape after appending: ', neuron_new_X.shape)
                
                full_test_gene_names += train_gene_names_subset

                print(" added zero instead of not_present_test_gene_names[i]: ", not_present_test_gene_names[i])
                # insert a new column of zeros for the genes that are not in the test gene ids
                neuron_new_X = np.concatenate(
                    [neuron_new_X, np.zeros((test_adata.shape[0], 1))],
                    axis=1)
                assert neuron_new_X.shape[1] == not_present_train_gene_idx[i]+1
                assert neuron_new_X[:,-1].sum() == 0
                full_test_gene_names += [not_present_test_gene_names[i]]

                if i == len(not_present_train_gene_idx)-1:
                    train_gene_names_subset = train_gene_names[not_present_train_gene_idx[i]+1:]
                    test_gene_idx = [i for i, x in enumerate(test_gene_names) if x in train_gene_names_subset]
                    assert len(test_gene_idx) == len(train_gene_names_subset)
                    neuron_new_X = np.concatenate(
                        [neuron_new_X,
                         test_adata.X[:, test_gene_idx].toarray()],
                        axis=1)
                    assert neuron_new_X.shape[1] == len(train_gene_names)
                    full_test_gene_names += train_gene_names_subset

                neuron_test_common_genes += train_gene_names_subset
            
            # check if all elements of full_test_gene_names and train_gene_names are the same in order
            assert all(x == y for x, y in zip(full_test_gene_names, train_gene_names))
            # assert if neuron_test_common_genes is a subset of train_gene_names
            assert all(x in train_gene_names for x in neuron_test_common_genes)

            test_adata = ad.AnnData(
                X=neuron_new_X,
                obs=test_adata.obs,
                var=pd.DataFrame(index=train_gene_names)
            )
            
            # save the genes used for training in a csv file
            path_to_test_adata_gene_file = args.root+"/data/sctab/"+\
                "bloodbase_"+str(args.atlas_count)+'_atlas'+\
                "_seed"+str(args.seed)+"_AR"+str(args.AR)[0]+\
                "_neuron_test_adata_"+str(args.hvg_count)+"_common_gene_names.csv"
            
            # write the genes to a file
            with open(path_to_test_adata_gene_file, 'w') as f:
                for item in neuron_test_common_genes:
                    f.write("%s\n" % item)
            
        else:
            test_adata = test_adata[:, gene_list]

        assert train_adata.shape[1] == test_adata.shape[1]
        assert all(train_adata.var_names == test_adata.var_names)
    
    return train_adata, valid_adata, test_adata
        

def count_evaluation_cell_counts(args, train_adata):
    """Count the number of cells which contian "heart", "neuron", and "kidney" 
    in the cell_type, tissue, or tissue_general columns of the concatenation 
    of train and valid adata.
    
    Args:
        args (argparse): arguments
        train_adata (AnnData): AnnData object containing the training data"""

    cell_groups = ['heart', 'neuron', 'kidney']
    # assert that the size of train_adata is 100000
    assert train_adata.shape[0] == 90000, "train_adata shape is not 100000"

    cell_count_df = pd.DataFrame(columns=['atlas_count','cell_group', 'count', 'seed'])
    metadata_df = train_adata.obs

    for cell_group in cell_groups:
        group_metadata_df = metadata_df[metadata_df['tissue_general'].str.contains(cell_group, case=False) |
                                        metadata_df['cell_type'].str.contains(cell_group, case=False) |
                                        metadata_df['tissue'].str.contains(cell_group, case=False)]
        cell_count = group_metadata_df.shape[0]
        cell_count_df = pd.concat([cell_count_df, pd.DataFrame(
            [[args.atlas_count, cell_group, cell_count, args.seed]],
            columns=['atlas_count','cell_group', 'count', 'seed'])])

    path_to_cell_count_file = args.root+"/data/sctab/"+\
        "training_atlas_cell_counts.csv"

    if not os.path.exists(path_to_cell_count_file):
        cell_count_df.to_csv(path_to_cell_count_file, index=False)
        print("Saved cell count to: ", path_to_cell_count_file)
    else: # append to the existing file
        existing_df = pd.read_csv(path_to_cell_count_file)
        combined_df = pd.concat([existing_df, cell_count_df], ignore_index=True)
        combined_df.to_csv(path_to_cell_count_file, index=False)
        print("Appended cell count to: ", path_to_cell_count_file)

    return


def load_and_preprocess_data(args):
    """Load data and adjust control test samples if needed.
    
    Args:
        args (argparse): arguments"""

    if args.data == 'pbmc':
        adata = load_pbmc(args)
    
    elif args.data == 'species':
        adata = load_species(args)
    
    elif args.data == 'lps-hpoly':
        adata = load_hpoly(args)
        
    elif args.data == 'sctab':
        assert args.model_name == 'scvi', "Model name should be scvi for sctab data"
        train_adata, valid_adata, test_adata = read_and_preprocess_data_for_scvi_sctab(args)

        if args.count_atlas_cells:
            count_evaluation_cell_counts(args, train_adata)
        return train_adata, valid_adata, test_adata
    
    else: # new dataset
        if args.h5ad_adata_file != None:
            adata = sc.read_h5ad(args.h5ad_adata_file)

            if args.model_name == 'scgen':
                assert args.adata_label_cell != None, "adata_label_cell should not be None"
                assert args.adata_label_per != None, "adata_label_per should not be None"
                assert args.adata_label_unper != None, "adata_label_unper should not be None"
                assert args.train_data != None, "train_data should not be None."+ \
                    "args.train_data should be a string concatenating cell groups from the " + \
                    "args.h5ad_adata_file in the args.adata_label_cell field that are used for training, concatenated by ','"
                assert args.test_data != None, "test_data should not be None. "+ \
                    "args.test_data should be a string inclduing the cell group from the " + \
                    "args.h5ad_adata_file in the args.adata_label_cell field that is exclueded durring training."

                if args.train:
                    adata = adjust_training_proportions(args, adata)
            else:
                # split the data into training and validation sets
                
                train_adata = None
                valid_adata = None
                test_adata = None

                if args.train_adata_path is not None:
                    # read the h5ad file from train_adata_path
                    train_adata = ad.read_h5ad(args.train_adata_path)
                    print('Loaded train_adata from:', args.train_adata_path)
                    
                if args.valid_adata_path is not None:
                    # read the h5ad file from valid_adata_path
                    valid_adata = ad.read_h5ad(args.valid_adata_path)
                    print('Loaded valid_adata from:', args.valid_adata_path)
                    
                if args.test_adata_path is not None:
                    # read the h5ad file from test_adata_path
                    test_adata = ad.read_h5ad(args.test_adata_path)
                    print('Loaded test_adata from:', args.test_adata_path)  
                
                return train_adata, valid_adata, test_adata

    return adata

