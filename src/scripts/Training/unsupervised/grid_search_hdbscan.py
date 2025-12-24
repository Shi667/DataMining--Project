import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from tqdm import tqdm
import os
import hdbscan


def stratified_sample(X, labels, total_samples, random_state=42):
    rng = np.random.default_rng(random_state)
    unique_labels, counts = np.unique(labels, return_counts=True)
    N = len(labels)

    indices = []

    for k, count in zip(unique_labels, counts):
        cluster_idx = np.where(labels == k)[0]
        # proportional sampling: fraction of total_samples
        n_samples = max(int(total_samples * count / N), 1)

        if len(cluster_idx) <= n_samples:
            indices.extend(cluster_idx)
        else:
            indices.extend(rng.choice(cluster_idx, n_samples, replace=False))

    # Convert X to NumPy if it's a DataFrame
    if isinstance(X, pd.DataFrame):
        X = X.values

    return X[indices], labels[indices]


def run_hdbscan_grid(
    X,
    min_cluster_sizes,
    min_samples_values,
    output_path: str,
    metrics=("ch", "dbi", "silhouette"),
    metric="euclidean",
    cluster_selection_method="eom",
    silhouette_sample_size=50_000,
    silhouette_n_repeats=5,
    random_state=42,
):
    """
    HDBSCAN grid search with:
    - number of clusters
    - number & percentage of outliers
    - internal clustering metrics
    - incremental CSV saving (safe for long runs)
    """

    # --------------------------------------------------
    # 🔑 Ignore lat/lon ONLY for the algorithm
    # --------------------------------------------------
    id_cols = ["latitude", "longitude"]
    X_algo = X.drop(columns=[c for c in id_cols if c in X.columns])

    # Create CSV with header if it doesn't exist
    if not os.path.exists(output_path):
        pd.DataFrame().to_csv(output_path, index=False)

    grid = list(product(min_cluster_sizes, min_samples_values))

    for min_cs, min_s in tqdm(grid, desc="HDBSCAN Grid Search"):

        # -----------------------
        # Fit HDBSCAN
        # -----------------------
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cs,
            min_samples=min_s,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
        )

        labels = clusterer.fit_predict(X_algo)

        N = len(labels)
        n_noise = np.sum(labels == -1)
        noise_ratio = n_noise / N

        unique_clusters = set(labels)
        n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)

        row = {
            "algorithm": "hdbscan",
            "min_cluster_size": min_cs,
            "min_samples": min_s,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_ratio": noise_ratio,
        }

        # Remove noise for metric computation
        mask = labels != -1
        X_clean = X_algo[mask]
        labels_clean = labels[mask]

        # -----------------------
        # Metrics (NO lat/lon)
        # -----------------------
        if len(np.unique(labels_clean)) >= 2:

            if "ch" in metrics:
                row["CH"] = calinski_harabasz_score(X_clean, labels_clean)

            if "dbi" in metrics:
                row["DBI"] = davies_bouldin_score(X_clean, labels_clean)

            if "silhouette" in metrics:
                silhouette_scores = []

                for i in range(silhouette_n_repeats):
                    Xs, ls = stratified_sample(
                        X_algo,
                        labels,
                        total_samples=silhouette_sample_size,
                        random_state=random_state + i,
                    )
                    silhouette_scores.append(silhouette_score(Xs, ls))

                row["Silhouette"] = np.mean(silhouette_scores)
                row["Silhouette_std"] = np.std(silhouette_scores)

        else:
            row["CH"] = np.nan
            row["DBI"] = np.nan
            row["Silhouette"] = np.nan
            row["Silhouette_std"] = np.nan

        # -----------------------
        # Save immediately
        # -----------------------
        df_row = pd.DataFrame([row])
        df_row.to_csv(
            output_path,
            mode="a",
            header=not os.path.getsize(output_path),
            index=False,
        )

        print(row)

    return pd.read_csv(output_path)
