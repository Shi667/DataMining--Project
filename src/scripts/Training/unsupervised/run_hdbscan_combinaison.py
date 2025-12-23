import os
import numpy as np
import pandas as pd
import hdbscan
from tqdm import tqdm
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def _get_non_existing_path(path):
    """Add suffix (_1, _2, ...) if file already exists."""
    if not os.path.exists(path):
        return path

    base, ext = os.path.splitext(path)
    i = 1
    new_path = f"{base}_{i}{ext}"
    while os.path.exists(new_path):
        i += 1
        new_path = f"{base}_{i}{ext}"
    return new_path


def run_hdbscan_combinations(
    X,
    param_combinations,
    output_path: str,
    metrics=("ch", "dbi", "silhouette"),
    metric="euclidean",
    cluster_selection_method="eom",
    silhouette_sample_size=50_000,
    silhouette_n_repeats=5,
    random_state=42,
):
    """
    Run HDBSCAN on explicit parameter combinations and save results to CSV.
    """

    output_path = _get_non_existing_path(output_path)

    all_rows = []

    for params in tqdm(param_combinations, desc="HDBSCAN Runs"):

        min_cs = params["min_cluster_size"]
        min_s = params["min_samples"]

        # -----------------------
        # Fit HDBSCAN
        # -----------------------
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cs,
            min_samples=min_s,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
        )

        labels = clusterer.fit_predict(X)

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

        # -----------------------
        # Remove noise
        # -----------------------
        mask = labels != -1

        # Ensure NumPy arrays (IMPORTANT)
        X_clean = X[mask]
        if isinstance(X_clean, pd.DataFrame):
            X_clean = X_clean.to_numpy()

        labels_clean = labels[mask]

        # -----------------------
        # Metrics
        # -----------------------
        if len(np.unique(labels_clean)) >= 2:

            if "ch" in metrics:
                row["CH"] = calinski_harabasz_score(X_clean, labels_clean)

            if "dbi" in metrics:
                row["DBI"] = davies_bouldin_score(X_clean, labels_clean)

            if "silhouette" in metrics:
                sil_scores = []

                for i in range(silhouette_n_repeats):
                    rng = np.random.RandomState(random_state + i)

                    if X_clean.shape[0] > silhouette_sample_size:
                        idx = rng.choice(
                            X_clean.shape[0],
                            silhouette_sample_size,
                            replace=False,
                        )
                        sil_scores.append(
                            silhouette_score(X_clean[idx], labels_clean[idx])
                        )
                    else:
                        sil_scores.append(silhouette_score(X_clean, labels_clean))

                row["Silhouette"] = np.mean(sil_scores)
                row["Silhouette_std"] = np.std(sil_scores)

        else:
            row["CH"] = np.nan
            row["DBI"] = np.nan
            row["Silhouette"] = np.nan
            row["Silhouette_std"] = np.nan

        # -----------------------
        # Save incrementally
        # -----------------------
        df_row = pd.DataFrame([row])
        df_row.to_csv(
            output_path,
            mode="a",
            header=not os.path.exists(output_path) or os.path.getsize(output_path) == 0,
            index=False,
        )

        print(row)

        all_rows.append(row)

    return pd.DataFrame(all_rows)
