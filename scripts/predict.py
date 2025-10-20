from scripts.dataloader import adjust_training_proportions
from scripts.scgen_AR.scgen._scgen import SCGEN
from scripts.utils import set_seed, create_id
import random
import torch
import scvi
import os


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
            adata[adata.obs[args.adata_label_cell] == args.in_dist_group]
            
        unper_test_adata = in_dist_adata[(
            in_dist_adata.obs.condition == args.adata_label_unper)]
        per_test_adata = in_dist_adata[(
            in_dist_adata.obs.condition == args.adata_label_per)]

    test_adata = per_test_adata.concatenate(unper_test_adata)
    return train_adata, test_adata, unper_test_adata, per_test_adata


def predict_and_save_anndata_scgen(args, adata, model_name='Naive'):
    """Predict the perturbed gene expression values for the given test cell group
    and save it.
    
    Args:
        args (argparse.Namespace): Arguments passed to the script.
        adata (anndata.AnnData): Anndata object containing the data.
        model_name (str, optional): Name of the model."""

    train_adata, test_adata, unper_test_adata, per_test_adata = \
        get_train_test_adata(args, adata)
    
    model = SCGEN.load(
        args.root+"saved_models/"+args.data+"/seed"+str(args.seed)+'/'+args.id+"-best.pt",
        train_adata.copy(),
        use_gpu = args.gpu)
    model.is_trained = True

    if args.variable_con and args.con_percent == 0.0:
        unper_test_adata.obs[args.adata_label_cell] = \
            train_adata.obs[args.adata_label_cell].unique()[0]

    set_seed(args.seed)
    pred_adata, delta = model.predict(
        ctrl_key=args.adata_label_unper,
        stim_key=args.adata_label_per,
        adata_to_predict=unper_test_adata
    )

    if args.in_dist_group == '':
        test_cell = args.test_data[0]
    elif args.in_dist_group != '':
        test_cell = args.in_dist_group

    unper_test_adata.obs[args.adata_label_cell] = test_cell
    pred_adata.obs[args.adata_label_cell] = test_cell
    pred_adata.obs.condition = model_name
    assert pred_adata.shape == unper_test_adata.shape

    if args.in_dist_group == '':
        pred_adata.write(
            args.root+"prediction/"+args.data+"/seed"+str(args.seed)+'/'+args.id+"-pred-adata.h5ad")
    elif args.in_dist_group != '':
        pred_adata.write(
            args.root+"prediction/"+args.data+"/seed"+str(args.seed)+'/'+args.id+"-pred-adata-"+args.in_dist_group+".h5ad")

    return


def predict_and_save_anndata_scvi(args, test_adata):
    """Predict the perturbed gene expression values for the given test cell group
    and save it.    
    
    Args:
        args (argparse.Namespace): Arguments passed to the script.
        adata (anndata.AnnData): Anndata object containing the data.
        model_name (str, optional): Name of the model."""
    
    model_path = args.out_path+args.data+"/seed"+str(args.seed) + \
        '/'+args.id+"-best/"

    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True
    else:
        use_gpu = False
    
    test_adata_copy = test_adata.copy()
    scvi.model.SCVI.setup_anndata(test_adata_copy)
    model = scvi.model.SCVI.load(model_path, adata=test_adata_copy, use_gpu=use_gpu)
    print('Loaded model from:', model_path)
    
    new_test_adata = test_adata.copy()
    
    latent = model.get_latent_representation(test_adata)
    new_test_adata.obsm['scvi_latent'] = latent
    
    new_test_adata.layers["scvi_1e4_normalized"] = \
        model.get_normalized_expression(test_adata, return_numpy=True, library_size=1e4)

    # save the predicted data to a file
    test_dataset = ''
    if 'heart' in args.test_adata_path.lower():
        test_dataset = 'heart'
    elif 'kidney' in args.test_adata_path.lower():
        test_dataset = 'kidney'
    elif 'neurons' in args.test_adata_path.lower():
        test_dataset = 'neurons'
    else:
        # raise error that the test dataset is not recognized
        raise ValueError('The test dataset is not recognized.')
    
    # print sum of new_test_adata.layers["scvi_1e4_normalized"] rows
    print("Sum of new_test_adata.layers[\"scvi_1e4_normalized\"] rows:")
    print(new_test_adata.layers["scvi_1e4_normalized"].sum(axis=1))
    
    
        
    new_test_adata.write(
        args.root+"/prediction/"+args.data+"/seed"+str(args.seed)+
        '/'+args.id+"-pred-adata-scvi-best-"+test_dataset+".h5ad")

    return


def predict(args, adata=None):
    """Make prediction.
    
    Args:
        args (argparse.Namespace): Arguments passed to the script.
        adata (anndata.AnnData, optional): Anndata object containing the data."""

    if not os.path.exists(args.root+"/prediction/"+args.data+"/seed"+str(args.seed)):
        os.makedirs(args.root+"/prediction/"+args.data+"/seed"+str(args.seed))

    models = args.model.split(',')
    for model_name in models:
        args.AR = False if model_name == 'Naive' else True
        create_id(args)
        
        if args.model_name == 'scgen':
            predict_and_save_anndata_scgen(args, adata, model_name)

        elif args.model_name == 'scvi':
            predict_and_save_anndata_scvi(args, adata)

    return