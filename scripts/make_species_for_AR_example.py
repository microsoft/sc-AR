"""Build the small, imbalanced cross-species subset used by the AR tutorial.

This script reproduces ``data/species_for_AR_example.h5ad`` from the full
cross-species dataset (Hagai et al., ``data/species.h5ad``). That prepared file
is the only data the ``AR_custom_model_tutorial_standard_vs_AR.ipynb`` notebook
needs, so it can be shared on its own (~16 MB) instead of the full 209 MB set.

What it does:
  1. Loads the full dataset. X is already library-size normalized + log1p
     transformed, so no re-normalization is applied.
  2. Selects the top ``N_TOP_GENES`` highly variable genes.
  3. Labels each cell by its Louvain cluster (the "cell type" analog),
     stored in ``obs["str_labels"]``.
  4. Subsamples to a small, strongly imbalanced subset: the most abundant
     population keeps the first entry of ``TARGET_PER_TYPE`` cells, down to the
     last entry for the rarest — a ~250x imbalance over ~5,000 cells total.
  5. Writes the result to the output path.

Usage
-----
    python scripts/make_species_for_AR_example.py \
        --input data/species.h5ad \
        --output data/species_for_AR_example.h5ad
"""

import argparse
import os

import numpy as np
import scanpy as sc

# Cells kept per population, applied from the most- to least-abundant Louvain
# cluster. Sums to ~5,000 cells with a ~250x largest/smallest imbalance.
TARGET_PER_TYPE = [2500, 1200, 600, 300, 150, 90, 60, 45, 35, 28, 22, 18, 15, 12, 10]


def build_subset(input_path, n_top_genes, label_col, seed):
    """Return the small, imbalanced AnnData subset (no file I/O)."""
    adata = sc.read_h5ad(input_path)

    # species.h5ad is already normalized + log1p transformed -> only pick HVGs.
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var.highly_variable].copy()

    # Louvain cluster as the cell-population label (switch to "species" if desired).
    adata.obs["str_labels"] = (
        "clust_" + adata.obs[label_col].astype(str)
    ).astype("category")

    rng = np.random.default_rng(seed)
    counts = adata.obs["str_labels"].value_counts()  # most -> least abundant
    keep_idx = []
    for rank, (ctype, n_avail) in enumerate(counts.items()):
        target = TARGET_PER_TYPE[rank] if rank < len(TARGET_PER_TYPE) else TARGET_PER_TYPE[-1]
        n_keep = min(target, int(n_avail))
        pos = np.where(adata.obs["str_labels"].to_numpy() == ctype)[0]
        keep_idx.append(rng.choice(pos, size=n_keep, replace=False))

    keep_idx = np.sort(np.concatenate(keep_idx))
    return adata[keep_idx].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/species.h5ad",
                        help="Path to the full cross-species dataset.")
    parser.add_argument("--output", default="data/species_for_AR_example.h5ad",
                        help="Where to write the prepared subset.")
    parser.add_argument("--n-top-genes", type=int, default=2000,
                        help="Number of highly variable genes to keep.")
    parser.add_argument("--label-col", default="louvain",
                        help="obs column used as the cell-population label.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the imbalanced subsampling.")
    args = parser.parse_args()

    adata = build_subset(args.input, args.n_top_genes, args.label_col, args.seed)
    adata.write_h5ad(args.output)

    counts = adata.obs["str_labels"].value_counts()
    print(f"Saved: {args.output}")
    print(f"Shape: {adata.shape}")
    print("\nCell population distribution:")
    print(counts.to_string())
    print(f"\nImbalance ratio (largest / smallest): {counts.max() / counts.min():.0f}x")
    print(f"File size (MB): {os.path.getsize(args.output) / 1e6:.2f}")


if __name__ == "__main__":
    main()
