"""Main file."""
import anndata as ad
import psutil
import torch

# Check if GPUs are available
if torch.cuda.is_available():
    # Limit GPU memory growth (PyTorch automatically handles memory allocation)
    print(f"GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{i}")
        # PyTorch doesn't have a direct equivalent for TensorFlow's memory growth setting,
        # but we can limit the memory usage manually using environment variables or control the memory growth indirectly.
        # Here's an example to set the GPU memory fraction (for example, limit to 50% memory usage per GPU)
        torch.cuda.set_device(device)
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
else:
    print("No GPUs available.")


# CPU usage monitoring
cpu_percent = psutil.cpu_percent(interval=1)  # Measure CPU usage over 1 second
cpu_count = psutil.cpu_count(logical=True)  # Total number of logical CPUs
virtual_memory = psutil.virtual_memory()  # Virtual memory stats

print(f"CPU Usage: {cpu_percent}%")
print(f"Total CPUs: {cpu_count}")
print(f"Total Memory: {virtual_memory.total / (1024**3):.2f} GB")
print(f"Available Memory: {virtual_memory.available / (1024**3):.2f} GB")
print(f"Used Memory: {virtual_memory.used / (1024**3):.2f} GB")

from scripts.utils import str2bool, create_id, create_directory, quality_control_adata
from scripts.dataloader import load_and_preprocess_data, split_randomely
from scripts.train import train
from eval_scripts.eval import test_scgen, test_scvi
from scripts.predict import predict
from datetime import datetime
import argparse
import torch
import wandb
import time
import os




# Print message indicating the script is being accessed
print("This is testing that this script is getting accessed")

# List available GPUs in PyTorch
if torch.cuda.is_available():
    print(f"Available GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
else:
    print("No GPUs available.")


def main(args):
    """Preprocess/load data, and train or test data based on input.
    
    Args:
        args (argparse): input arguments"""
    wandb.login()
    start_time = time.time()
    
    ## create id and directory
    create_id(args)
    create_directory(args)
    
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))
    print('_____________________________________________________________')
    print()

    ## load and prepare data
    
    if args.model_name == 'scgen':
        adata = load_and_preprocess_data(args)
        
        if not args.ood:
            adata = split_randomely(args, adata)
            
    if args.model_name == 'scvi':
        train_adata, valid_adata, test_adata = load_and_preprocess_data(args)
        adata = train_adata.copy()
  
    
    print('Data loaded and preprocessed')
    
    ## quality control
    if args.qq:
        assert args.model_name == 'scgen', 'Quality control is only implemented for scgen'
        quality_control_adata(adata, args)
        return

    ## debug mode
    if args.debug:
        if not os.path.exists(args.root+'/debug/'+args.id):
            os.mkdir(args.root+'/debug/'+args.id)

    ## train
    if args.train:
        # set up wandb
        wandb_id = "MSR-"+"{:%m%d}".format(datetime.now())+'-'+args.id
        wandb.init(project=wandb_id,
                   config=args,
                   dir=args.root+'/',)
        config_dict = wandb.config._items

        # Save the configuration to a local file
        with open(args.out_path+args.data +
                  "/seed"+str(args.seed)+'/config/'+
                  args.id+"_wandb_config.yaml", "w") as f:
            for key, value in config_dict.items():
                f.write(f"{key}: {value}\n")
        
        if args.model_name == 'scgen':
            train(args, adata)

        elif args.model_name == 'scvi':
            train(args, train_adata, valid_adata)
        
        if not os.path.isfile(args.root+'/result/train/trained_models_log.csv'):
            file = open(args.root+'/result/train/trained_models_log.csv', 'w')
            
        with open(args.root+'/result/train/trained_models_log.csv', 'a') as f:
            f.write(args.id+','+"{:%Y%m%d}".format(datetime.now())+'\n')

    ## predict
    elif args.predict:
        if args.model_name == 'scgen':
            predict(args, adata)
        elif args.model_name == 'scvi':
            predict(args, test_adata)

    ## test
    elif args.test:
        if args.model_name == 'scvi':
            test_scvi(args, train_adata)

        elif args.model_name == 'scgen':
            model_dic={}
            test_scgen(args, adata, model_dic)
        
    print("--- Task completed in %s seconds ---" % (time.time() - start_time))

    return


if __name__ == '__main__':
    print('Started main.py')
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--root', default=".",
                        help="path to the sc-uncertainty directory")

    # task related arguments
    parser.add_argument("--train", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Train model. True or False")
    parser.add_argument("--predict", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Make prediction. True or False")
    parser.add_argument("--test", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Run test evaluations. True or False")
    parser.add_argument("--qq", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Perform quality control. True or False")
    parser.add_argument("--debug", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Debug mode or not")
    parser.add_argument("--check_scvi", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Debug mode or not")
    parser.add_argument("--count_atlas_cells", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Count atlas cells in training data")
    parser.add_argument("--plot_umap_annotated_with_w", ## TODO: set default to False
                        type=str2bool, nargs='?', 
                        const=True, default=False,
                        help="True or False")
    parser.add_argument("--degs_extraction_based_on_resampling_w",
                        type=str2bool, nargs='?',
                        const=True, default=False,
                        help="True or False")

    # model related arguments
    parser.add_argument('--id', default="",
                        help="a name for identifying the model")
    parser.add_argument('--test_id', default="",
                        help="a name for saving test results")
    parser.add_argument('--model', default='AR',
                        help="AR or Naive")
    parser.add_argument('--model_name', default='scgen',
                        help="scgen,scvi")
    parser.add_argument("--AR", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="True or False")
    parser.add_argument('--latent_dim', default=64,
                        help="latent dimension")
    parser.add_argument('--num_epoch', default=300,
                        help="number of epochs for training")
    parser.add_argument('--optimizer', default='adam',
                        help="optimizer for training")
    parser.add_argument('--batch_size', default=4096 , #2048
                        help="optimizer for training")
    parser.add_argument('--lr', default=0.00005,
                        help="optimizer for training")
    parser.add_argument('--weight_decay', default=0.00005,
                        help="weight_decay")
    parser.add_argument('--cell_count', default=None,
                        help="Number of cells to subsample")
    parser.add_argument('--hvg_count', default=2000,
                        help="Number of highly variable genes to subsample")

    # AR model related arguments
    parser.add_argument('--bins', default=100,
                        help="weight_decay")
    parser.add_argument('--alpha', default=0.0001,
                        help="alpha")

    # data related argument
    parser.add_argument('--data', type=str, default="ajay",
                        help='name or id of the input data')
    parser.add_argument('--data_path', type=str, default='../data/ajay/',
                        help='if not None, the path to the data folder')
    parser.add_argument('--h5ad_adata_file', type=str, default=None,
                        help='if not None, the path to input adata file')
    # parser.add_argument('--valid_adata_path', type=str, default=None,
    #                     help='path to valid adata')
    parser.add_argument('--test_adata_path', type=str, default=None,
                        help='if not None, the path to scvi test adata')
    parser.add_argument('--train_data', nargs="+", default='', 
                        help="train cell types, concatenated with ','")
    parser.add_argument('--test_data', nargs="+", default='',
                        help="test cell type")
    parser.add_argument("--ood", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="True or False")
    parser.add_argument("--variable_con", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="if variable control cells from the test cell group are included in the training set")
    parser.add_argument('--con_percent', default=1.0, type=float,
                        help='percent of test group contorl cell included in training, 0, 0.2, 0.4, 0.6, 1.0')
    parser.add_argument('--in_dist_group', default='',
                        help="in distirbution test cell group")
    parser.add_argument('--out_path', default='./saved_models/',
                        help="path to save the model ckpts")
    parser.add_argument('--atlas_count', default=0, type=int,
                        help='number of atlas cells included in training, combined with blood cells')

    # misc arguments
    parser.add_argument('--seed', default=0, type=int, help='manual seed')
    parser.add_argument('--adata_label_cell', default=None, 
                        help='label for cell types in adata')
    parser.add_argument('--adata_label_per', default=None,
                        help='value of  adata_label_cell for perturbed cells in adata')
    parser.add_argument('--adata_label_unper', default=None,
                        help='value of adata_label_cell for unperturbed cells in adata')
    parser.add_argument('--checkpoint', default=20,
                        help="checkpoint for saving model")
    parser.add_argument("--gpu", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="True or False")
    parser.add_argument('--tracked_epoch', default='best')
    

    # parse and preprocess args
    args = parser.parse_args()
    args.num_epoch = int(args.num_epoch)
    args.latent_dim = int(args.latent_dim)
    if args.train_data != '':
        args.train_data = args.train_data[0].split(',')
    if args.test_data != '':
        args.test_data = args.test_data[0].split(',')
    if args.hvg_count != None:
        args.hvg_count = int(args.hvg_count)
    if args.cell_count != None:
        args.cell_count = int(args.cell_count)
    if args.atlas_count != None:
        args.atlas_count = int(args.atlas_count)
    if args.con_percent != None:
        args.con_percent = float(args.con_percent)
    if args.seed != None:
        args.seed = int(args.seed)
        
    args.gpu = torch.cuda.is_available()
    
    if not args.test:
        if args.AR:
            args.model = 'AR'
        else:
            args.model = 'Naive'

    main(args)
    print('Finished main.py')