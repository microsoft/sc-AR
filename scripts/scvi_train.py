from datetime import datetime
from matplotlib import pyplot
from scripts.utils import set_seed
import numpy as np
import random
import torch
import wandb
import scvi
import os


class scAR:
    def __init__(self,
                 train_adata,
                 valid_adata,
                 model_name='scvi',
                 root='/',
                 seed=42,
                 num_epoch=10,
                 checkpoint=5,
                 AR=False,
                 bins=10,
                 smoothing_fac=0.001,
                 data=None,
                 id=None,
                 batch_size=32,
                 debug=False,
                 lr=0.001,
                 n_latent=10,
                 out_path=None): 
        """Initialize scAR object.
        
        Args:
            train_adata (AnnData): AnnData object containing training data
            valid_adata (AnnData): AnnData object containing validation data
            model_name (str, optional): Name of the model. Defaults to 'scvi'.
            root (str, optional): Root directory. Defaults to '/'.
            seed (int, optional): Random seed. Defaults to 42.
            num_epoch (int, optional): Number of epochs for training. Defaults to 10.
            checkpoint (int, optional): Checkpoint for saving model. Defaults to 5.
            AR (bool, optional): AR technique enabled or not. Defaults to False.
            bins (int, optional): Number of bins for histogram. Defaults to 10.
            smoothing_fac (float, optional): Smoothing factor for histogram. Defaults to 0.001.
            data (str, optional): Data name. Defaults to None.
            id (str, optional): ID for the model. Defaults to None.
            batch_size (int, optional): Batch size. Defaults to 32.
            debug (bool, optional): Debug mode enabled or not. Defaults to False.
            lr (float, optional): Learning rate. Defaults to 0.001.
            n_latent (int, optional): Number of latent dimensions. Defaults to 10."""
        self.train_adata = train_adata
        self.valid_adata = valid_adata
        self.model_name = model_name
        self.root = root
        self.seed = seed
        self.num_epoch = num_epoch
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.lr = lr
        
        # AR parameters
        self.AR = AR
        self.bins = bins
        self.smoothing_fac = smoothing_fac
        
        # scgen parameters
        self.n_latent = n_latent
        self.data = data
        self.id = id
        self.debug = debug
        self.out_path = out_path
        
        # parameters to be updated during training
        self.model = None
        self.train_loss_elbo_list = []
        self.train_loss_recon_list = []
        self.valid_loss_elbo_list = []
        self.valid_loss_recon_list = []
        self.best_loss = 10e7
        self.best_epoch = 0

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        if self.AR:
            print("Created scAR object with AR technique enabled")
        else:
            print("Created scAR model with naive setting")
        
    def initialize_model(self):
        """Initialize scVI model."""
        
        if self.model_name == 'scvi':
            print("setting up anndata")
            scvi.model.SCVI.setup_anndata(self.train_adata)

            self.model = scvi.model.SCVI(
                self.train_adata, n_latent=self.n_latent)
            self.model.module = self.model.module.to(self.device)
            print("created scvi model")

        return
        
    def get_latent_representation(self):
        """Get the latent representation of the data."""
        
        if self.model_name == 'scvi':
            return self.model.get_latent_representation(
                self.train_adata)

        return

    def update_resampling_weights(self, z): 
        """Updates the sampling probabilities for cells within a batch based on the maximum of
        latent variables' probablities.
        source: https://goodboychan.github.io/python/tensorflow/mit/2021/02/27/Debiasing.html
            
        Args:
            z (torch.Tensor): Latent variables.
            bins (int, optional): Number of bins to use for histogram. Defaults to 10.
            smoothing_fac (float, optional): Smoothing factor for histogram. Defaults to 0.001."""
            
        bins = self.bins
        smoothing_fac = self.smoothing_fac

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
    
    def resample_all_batches(self, w):
        """Resample from original training adata based on resampling weights w.
        
        Args:
            w (np.array): Resampling weights.
            
        Returns:
            AnnData: Resampled training data."""
        idx = random.choices(
            range(self.train_adata.shape[0]),
            weights=w,
            k=self.train_adata.shape[0])

        assert len(idx) == self.train_adata.shape[0]
        adata = self.train_adata[idx]

        return adata
    
    def load_pretrained_model(self, adata):
        """Load the pretrained model with the resampled data and save in 
        self.model variable.
        
        Args:
            adata (AnnData): Resampled training data."""
        
        if self.model_name == 'scvi':

            scvi.model.SCVI.prepare_query_anndata(adata, self.model)
            query_model = scvi.model.SCVI.load_query_data(
                adata.copy(),
                self.model,
                unfrozen=True)
            self.model = query_model

        return
            
    def train_one_epoch(self, train_size):
        """Train the model for one epoch.
        
        Args:
            train_size (float): Training size."""
        
        if self.model_name == 'scvi':
            # check if gpu is available
            if torch.cuda.is_available():
                self.model.train(
                    max_epochs=1,
                    train_size=train_size,
                    batch_size=self.batch_size,
                    shuffle_set_split=False,
                    check_val_every_n_epoch=1,
                    plan_kwargs={"n_steps_kl_warmup": None,
                                 "n_epochs_kl_warmup": None,
                                 "lr": self.lr},
                    accelerator="gpu",
                    )
            else:
                self.model.train(
                    max_epochs=1,
                    train_size=train_size,
                    batch_size=self.batch_size,
                    shuffle_set_split=False,
                    check_val_every_n_epoch=1,
                    plan_kwargs={"n_steps_kl_warmup": None,
                                 "n_epochs_kl_warmup": None,
                                 "lr": self.lr}
                    )

        return
            
    def update_train_loss(self):
        """Update the training loss."""
        
        if self.model_name == 'scvi':
            train_loss_elbo = self.model.history[
                'elbo_train']['elbo_train'][0]
            train_loss_recon = self.model.history[
                'reconstruction_loss_train']['reconstruction_loss_train'][0]
        
            self.train_loss_elbo_list.append(train_loss_elbo)
            self.train_loss_recon_list.append(train_loss_recon)

        return

    def update_validation_loss(self):
        """Update the validation loss."""
        
        if self.model_name == 'scvi':

            # Compute the ELBO (lower bound on log-likelihood) as a proxy for loss
            valid_loss_elbo = self.model.history[
                'elbo_validation']['elbo_validation'][0]
            valid_loss_recon = self.model.history[
                'reconstruction_loss_validation']['reconstruction_loss_validation'][0]

            self.valid_loss_elbo_list.append(valid_loss_elbo)
            self.valid_loss_recon_list.append(valid_loss_recon)

        return

    def save_checkpoint(self, epoch):
        """Save the model checkpoint.
        
        Args:
            epoch (int): Epoch number."""
        
        ## save the best model
        if self.valid_loss_elbo_list[-1] < self.best_loss:
            print("Saving the best model")
            self.model.save(
                self.out_path+self.data+"/seed"+str(self.seed) + \
                '/'+self.id+"-best", overwrite=True)
            self.best_loss = self.valid_loss_elbo_list[-1]
            self.best_epoch = epoch
        
        if epoch % self.checkpoint == 0:
            self.model.save(self.out_path+self.data+ \
                "/seed"+str(self.seed)+'/'+self.id+"-epoch"+ \
                str(epoch),
                overwrite=True)
                
            ## save the model at the end of eacg epoch
            self.model.save(
                self.out_path+self.data+"/seed"+str(self.seed) + \
                '/'+self.id, overwrite=True)
        return
            
    def perform_final_documentation(self):
        """Perform final documentation."""

        file = self.out_path+self.data+"/seed"+str(self.seed)+'/' + \
            "{:%Y%m%d}".format(datetime.now())+'_'+self.id + \
            '_valid_loss.csv'
        np.savetxt(file, np.array(self.valid_loss_elbo_list, dtype=float), delimiter=',')

        ## save the best epoch in the config file
        with open(self.out_path+self.data +
                "/seed"+str(self.seed)+'/config/'+
                self.id+"_wandb_config.yaml", "a") as f:
            f.write("best_epoch: "+str(self.best_epoch)+"\n")

        return

    def train(self):
        """Train scVI model."""
        set_seed(self.seed)

        ## setting up anndata: first 10% of the data is validation set, rest is training set
        # in scvi.dataloaders.DataSplitter, if data shuffling is set to False, If False, 
        # the val, train, and test set are split in the sequential order of the data according 
        # to validation_size and train_size percentages.
        adata = self.valid_adata.concatenate(self.train_adata).copy()
        train_size = self.train_adata.shape[0] / adata.shape[0]
        
        ## setting up model
        self.initialize_model()

        set_seed(self.seed)
        for epoch in range(self.num_epoch):

            if epoch > 0:
        
                if self.AR:
                    ## update the adata based on AR algorithm
                    print("Updating resampling weights")
                    z = self.get_latent_representation()
                    w = self.update_resampling_weights(z)
                    resampled_train_adata = self.resample_all_batches(w)
                    assert resampled_train_adata.shape[0] == self.train_adata.shape[0]
                        
                    if self.debug:
                        # plot the histogram of the resampling weights
                        pyplot.figure()
                        fig = pyplot.hist(2**w, bins=100) # could be 2**w
                        pyplot.savefig(
                            self.root+'/debug/'+self.id+'/w_hist_epoch'+str(epoch)+'.png',
                            dpi=300, bbox_inches='tight')

                        if os.path.exists(self.root+'/debug/' + \
                            self.id+'/w_z_scvi.txt'):
                            with open(self.root+'/debug/'+self.id + \
                                '/w_z_scvi.txt', 'a') as f:
                                f.write(str(epoch)+'\n')
                                f.write('w[0:10]: '+str(w[0:10])+'\n')
                                f.write('z[0:2]: '+str(z[0:2])+'\n')
                                f.write('w min, max, mean: '+str(w.min())+', '+str(w.max())+', '+str(w.mean())+'\n')
                                f.write('-'*100+'\n')
                        else:
                            with open(self.root+'/debug/' + \
                                self.id +'/w_z_scvi.txt', 'w') as f:
                                f.write(str(epoch)+'\n')
                                f.write('w[0:10]: '+str(w[0:10])+'\n')
                                f.write('z[0:2]: '+str(z[0:2])+'\n')
                                f.write('w min, max, mean: '+str(w.min())+', '+str(w.max())+', '+str(w.mean())+'\n')
                                f.write('-'*100+'\n')
                                
                    adata = self.valid_adata.concatenate(resampled_train_adata).copy()

                # load the model with the resampled data
                self.load_pretrained_model(adata)

            if self.debug:
                # append the model parameters in a file with epoch number
                if os.path.exists(self.root+'/debug/'+self.id+'/model_param.txt'):
                    with open(self.root+'/debug/'+self.id+'/model_param.txt', 'a') as f:
                        f.write(str(epoch)+'\n')
                        for param in self.model.module.parameters():
                            f.write(str(param)+'\n')
                        f.write('-'*100+'\n')
                else:
                    with open(self.root+'/debug/'+self.id+'/model_param.txt', 'w') as f:
                        f.write(str(epoch)+'\n')
                        for param in self.model.module.parameters():
                            f.write(str(param)+'\n')
                        f.write('-'*100+'\n')
            
            ## perform one training step 
            print("Training scVI epoch {}".format(epoch))
            self.train_one_epoch(train_size)

            ## log loss values
            self.update_train_loss()
            self.update_validation_loss()
            wandb.log({
                "ELBO Train Loss": self.train_loss_elbo_list[-1],
                "RECON Train Loss": self.train_loss_recon_list[-1],
                "ELBO Valid Loss": self.valid_loss_elbo_list[-1],
                "RECON Valid Loss": self.valid_loss_recon_list[-1],
                "Epoch": epoch})
                
            ## save the model every checkpoint
            self.save_checkpoint(epoch)
            
        self.perform_final_documentation()

        # ## plot UMAP of the latent representation
        # if not os.path.exists('figures/umap/'+self.data):
        #     os.makedirs('figures/umap/'+self.data)
            
        # latent = self.model.get_latent_representation(self.train_adata)
        # self.train_adata.obsm["X_scVI"] = latent

        # sc.pp.neighbors(self.train_adata, use_rep="X_scVI")
        # sc.tl.umap(self.train_adata, min_dist=0.3)

        # sc.pl.umap(
        #     self.train_adata,
        #     color=["tissue_general"],
        #     frameon=False,
        #     save="/"+self.data+ \
        #         '/'+self.id+".png",
        # )
        print('--- scAR training completed ---')
        
        return

