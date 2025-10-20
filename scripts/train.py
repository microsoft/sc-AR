from scripts.scgen_AR.scgen._scgen import SCGEN
from scripts.scvi_train import scAR
from scripts.utils import set_seed
import torch
import scvi


def train(args, adata, valid_adata=None):
    """Calls train function based on args.AR values.
    
    Args:
        args (argparse): input arguments
        adata (AnnData): AnnData object containing all the data (for scgen), 
        and training data (for scvi)
        valid_adata (AnnData): AnnData object containing validation data (for scvi)
        
    Returns:
        dict: dictionary of trained models"""

    model = None
    set_seed(args.seed)

    if args.model_name == 'scgen':
        
        # extract the train data
        train_adata = adata[~(
            (adata.obs[args.adata_label_cell].isin(args.test_data)) &
            (adata.obs.condition == args.adata_label_per))]
        train_adata = train_adata.copy()

        SCGEN.setup_anndata(train_adata,
                            batch_key="condition",
                            labels_key=args.adata_label_cell)
        model = SCGEN(train_adata, n_latent=args.latent_dim)

        if args.AR:
            model.mytrain_AR(args, adata)

        else:
            model.mytrain_naive(args, adata)
            
    elif args.model_name == 'scvi':

        root = args.root
        
        scAR_obj = scAR(
            train_adata=adata,
            valid_adata=valid_adata,
            id=args.id,
            seed=args.seed,
            model_name=args.model_name,
            root=root,
            AR=args.AR,
            data=args.data,
            num_epoch=args.num_epoch,
            batch_size=args.batch_size,
            bins=args.bins,
            smoothing_fac=args.alpha,
            checkpoint=args.checkpoint,
            debug=args.debug,
            lr=args.lr,
            n_latent=args.latent_dim,
            out_path= args.out_path,
        )
        scAR_obj.train()

    return