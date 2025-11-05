# Adaptive Resampling For Improved Machine Learning In Imbalanced Single-cell Datasets

This repository contains the code that accompanies our paper, "Adaptive Resampling For Improved Machine Learning In Imbalanced Single-Cell Datasets". You can find the link to the preprint here.

## Abstract
While machine learning models trained on single-cell transcriptomics data have shown great promise in providing biological insights, existing tools struggle to effectively model underrepresented and out-of-distribution cellular features or states. We present a generalizable Adaptive Resampling (AR) approach that addresses these limitations and enhances single-cell representation learning by resampling data based on its learned latent structure in an online, adaptive manner concurrent with model training. Experiments on gene expression reconstruction, cell type classification, and perturbation response prediction tasks demonstrate that the proposed AR training approach leads to significantly improved downstream performance across datasets and metrics. Additionally, it enhances the quality of learned cellular embeddings compared to standard training methods. Our results suggest that AR may serve as a valuable technique for improving representation learning and predictive performance in single-cell transcriptomic models. 

---

## Environmental Setting

Create a new virtual environment (python 3.10.13 was used in our experiments), and install using the following commands:

```bash
conda env create -f environment.yaml
conda activate scar-env
```


## Codebase overview

Code used for training and validaiton presented in the manuscript are provided in this repository.

```bash

├── bash # Containing bash scripts for model training/evaluation with the hyper-parameters used for each experiment
├── eval_scripts # Containing scripts for evaluation
├── scripts # Containing main Python scripts for training models
├── data # Create a data folder where h5ad data files should be located in this path
├── saved_models # Model checkpoints will be saved in this directory
```

## Reproducing Results

All model checkpoints and scRNA-seq datasets used in our study are publicly available in the project's Zenodo dashboard: https://zenodo.org/records/15186018. Detailed guidance for each section is provided in the following.

## Datasets

You can download the datasets used for modelling from the Zenodo link provided above. More details about datasets used for each of the analysis are provided below:

For scGen models, you can run the following command to download the Species, PBMC, and H.poly datasets used for training and evaluation of scRNA-seq perturbation response prediction models. This script is a modified version of the https://github.com/theislab/scgen-reproducibility/blob/master/code/DataDownloader.py, and each h5ad file will be saved in `./data/` directory for the training and validation scripts to load them. The link to the public google Drive of the h5ad files provided by the original study is https://drive.google.com/drive/folders/1v3qySFECxtqWLRhRTSbfQDFqdUCAXql3.

```bash
python scripts/DownloadData.py
```

For training and evaluation of the scVI models, we used the following files that are available in the Zenodo and should be located in `./data/sctab/`:

- BaseModel_scTabAll_seed0_allbase_TrainingData.h5ad: generated using [https://github.com/microsoft/scFM-datamix/blob/main/Preprocess/preprocess_data_allbase.py]

- BaseModel_scTabBloodOnly_seed0_bloodbase_TrainingData.h5ad: generated using [https://github.com/microsoft/scFM-datamix/blob/main/Preprocess/preprocess_data_bloodbase.py]

- Neurons_H1830002_10Ksubset.h5ad: generated using [https://github.com/microsoft/scFM-datamix/blob/main/Preprocess/preprocess_eval_data.py]

- sctab_train_10pct_heart_nohematopoietic.h5ad: The dataset termed "Heart" was created by subsampling all the cells labeled "heart" in the scTab data and retaining only those with the following cell type labels: cardiac muscle cell; pericyte; capillary endothelial cell; fibroblast of cardiac tissue; endothelial cell of artery; smooth muscle cell; endothelial cell; vein endothelial cell; neuron; endothelial cell of lymphatic vessel; cardiac neuron; and mesothelial cell. This left the final evaluation dataset with 13,571 cells.

- sctab_train_10pct_kidney_nohematopoietic.h5ad: The dataset termed "Kidney" was created by subsampling all the cells labeled "kidney" in the scTab data and retaining only those with the following cell type labels: epithelial cell of proximal tubule; kidney loop of Henle thick ascending limb epithelial cell; endothelial cell; kidney collecting duct principal cell; kidney distal convoluted tubule epithelial cell; kidney interstitial fibroblast; kidney collecting duct intercalated cell; kidney connecting tubule epithelial cell; kidney loop of Henle thin descending limb epithelial cell; kidney loop of Henle thin ascending limb epithelial cell; renal interstitial pericyte; and vascular associated smooth muscle cell. This left the final evaluation dataset with 8,468 cells.



## Training

### scGen
The bash scripts used for training models are provided in the `bash/` directory. The parameters of the training scripts for each model can be adjusted, which train for both AR and Naive setting as indicated in `AR_list=(True False)`. The provided train bash scripts loop over different settings for each model with details provided in the comments, and trains separate model for each combination.

For scGen, the `bash/train_scgen.sh` loops over different cell groups to be excluded during training as the test set, seed values, and AR/Standard settings. Paramters can be adjusted as described in the bash scirpt if intend to run training on other datasets.

```bash
./train_scgen.sh
```

### scVI
For scVI models, read the input arguments of the `bash/train_scvi.sh` script and their descriptions. The following command is used to train scvi model on the scTab data files located in `/data/sctab/` directory, which trains models for different combination of blood base and atlas cells as training samples. The input arguments should be adjusted. 

```bash
./train_scvi.sh \
    data="sctab" \
    data_path="../data/sctab/" \
    out_path="../saved_models/" \
    seed=42 \
    log_path="/bash/log" \
    root="../" \
    model_name="scvi" \
    latent_dim=64 \
    num_epoch=300 \
    AR=True \
    atlas_count=0
```


## Prediction

### scGen
For perturbation response prediction (before running evaluation), run the following script, where you can adjust the dataset parameters, seeds, and AR vs. Standard settings based on model variables. The output would be a .h5ad file including model prediction, which will be usef for evaluation and is saved in `./prediction/${dataset}` folder.

```bash
./predict_scgen.sh
```

### scVI
Model prediction for the scVI models are done in the same script used for running evaluations, as described below. 

## Evaluation

### scGen
For reproducing evaluation results of the scGen models, run the following script, where you can adjust the dataset parameters, seeds, and AR vs. Standard settings based on model variables. The output includes all different evaluation metrics, which will be saved in `result/test/${dataset}` folder. The `eval_scripts/generate_plot.py` script is used for generating the plots for scGen evaluation.

```bash
./eval_scgen.sh
```

### scVI
For evalaution of scVI models, there are two bash scripts inside the `eval_scripts` folder for cell type classification and gene expression reconstruction metrics, which directly loads model files, makes prediction, and generate results. 

For cell type classification, run the following command:

```bash
./eval-scripts/eval-classification.sh \

# where the script loops over the following parameters for the scVI models, that can be adjusted.
seed_values=(42)
ARtype_values=("T" "F")
latent_dim_values=(64)
Atlas_cell_count=(0 1 10 100 1000 10000 50000)
```


For gene expression reconstruction evaluation, run the following command:
```bash
./eval-scripts/eval-reconstruction.sh \
```

where the script loops over the same parameters as in classification script, and can be adjusted. The eval outputs will be saved in `./result/test/scVI-reconstruction-evals/` and `./result/test/scVI-classification-evals/` by default, where you can modify the path in the bash scripts.


## Adaptive Resampling Module
We provide an Adaptive Resampling (AR) module that can be easily integrated into other machine learning pipelines aiming to incorporate adaptive resampling during training.
You can use the implemented function `calculate_resampling_weights()` located in `scripts/AR_resampling_weight_calculator.py`.

Simply pass the training sample projections in latent space (generated by your own pipeline) as input to the function.
It will return a list of resampling weights corresponding to all training samples, which can then be used to perform resampling in the next training iteration.

The AR module can also be applied once prior to training by computing the resampling weights from embeddings generated by large-scale foundational models.
These weights can be used to modify the training data distribution at the start of model training for any desired downstream task.



## Citation

If our work assists your research, please use the following to cite our work:

```bash

```

## License

This project is available under the MIT License.

## Transparency Documentation

Microsoft's Responsible AI Transparency Documentation for Research is provided on this repository, please see [scAR_Transparency](https://github.com/microsoft/sc-AR/blob/main/scAR_Transparency.pdf).

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.
