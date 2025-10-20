"""evaluation_utils.py contains a set of utility functions for use
in the model evaluation scripts."""
import pandas as pd
import sklearn.metrics
import anndata as ad

# from geneformer import TranscriptomeTokenizer


# creates an empty anndata object that has the genes we need
# directory should be a directory containing adata_var.csv (the gene names) and
# idx_1pct_seed0_TEST.h5ad (any small h5ad file the the scTab genes)
def get_empty_anndata(formatted_h5ad_file, var_file):
    """Creates an AnnData object with no cells that contains the genes from the scTab dataset.

    Args:
        formatted_h5ad_file: An h5ad file for an AnnData that contains the gene from the
        scTab dataset.
        var_file: A file containing the variable names for the scTab AnnDatas.

    Returns:
        An AnnData with no cells that contains the gene from the scTab dataset.
    """
    adata = ad.read_h5ad(formatted_h5ad_file)

    var_df = pd.read_csv(var_file, index_col=0)
    var_df.index = var_df.index.map(str)

    empty_adata = adata[0:0, :]
    empty_adata.var = var_df
    empty_adata.var_names = empty_adata.var.feature_name

    return empty_adata


# https://discourse.scverse.org/t/help-with-concat/676/2
# https://discourse.scverse.org/t/loosing-anndata-var-layer-when-using-sc-concat/1605
def prep_for_evaluation(adata, formatted_h5ad_file, var_file):
    """Creates an AnnData object with no cells that contains the genes from the scTab dataset.

    Args:
        adata: An AnnData whose variable names are gene symbols.
        formatted_h5ad_file: An h5ad file for an AnnData that contains the gene from the
        scTab dataset.
        var_file: A file containing the variable names for the scTab AnnDatas.

    Returns:
        The original AnnData object with only the genes from the scTab dataset.
    """
    empty_adata = get_empty_anndata(formatted_h5ad_file, var_file)
    return ad.concat([empty_adata, adata], join="outer")[:, empty_adata.var_names]

def eval_classification_metrics(y_true, y_pred):
    """Computes metrics for cell type classificaiton given the true labels and the
    predicted labels.

    Args:
        y_true: The true cell type labels.
        y_pred: The predicted cell type labels.

    Returns:
        A dictionary containing the accuracy, precision, recall, micro F1 score,
        and marco F1 score.
    """
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    precision = sklearn.metrics.precision_score(
        y_true, y_pred, average="macro")
    recall = sklearn.metrics.recall_score(y_true, y_pred, average="macro")
    micro_f1 = sklearn.metrics.f1_score(y_true, y_pred, average="micro")
    macro_f1 = sklearn.metrics.f1_score(y_true, y_pred, average="macro")

    classification_metrics_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }

    return classification_metrics_dict
