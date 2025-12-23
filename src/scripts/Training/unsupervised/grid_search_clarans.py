import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product

from pyclustering.cluster.clarans import clarans

from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
    pairwise_distances,
)


def assign_labels_from_medoids(X, medoid_indices):
    medoids = X[medoid_indices]
    distances = pairwise_distances(X, medoids)
    return np.argmin(distances, axis=1)


def run_clarans_grid(
    X,
    k_values,
    output_path: str,
    numlocal_values=(5,),
    maxneighbor_values=(50,),
    metrics=("ch", "dbi", "silhouette", "avg_medoid_dist"),
    sample_size=50_000,  # sampling for CLARANS itself
    eval_sample_size=50_000,  # sampling for metrics
    silhouette_n_repeats=5,
    random_state=42,
):
    np.random.seed(random_state)

    N = X.shape[0]
    results = []

    # -----------------------
    # Sample for CLARANS
    # -----------------------
    if sample_size < N:
        sample_idx = np.random.choice(N, sample_size, replace=False)
        X_sample = X[sample_idx]
    else:
        X_sample = X

    X_sample_list = X_sample.tolist()

    grid = list(product(k_values, numlocal_values, maxneighbor_values))

    for K, numlocal, maxneighbor in tqdm(grid, desc="CLARANS Grid Search"):

        clarans_instance = clarans(
            data=X_sample_list,
            number_clusters=K,
            numlocal=numlocal,
            maxneighbor=maxneighbor,
        )

        clarans_instance.process()
        medoids_sample = clarans_instance.get_medoids()

        # Map medoids back to full data indices
        medoid_indices = sample_idx[medoids_sample]

        # Assign labels to full dataset
        labels = assign_labels_from_medoids(X, medoid_indices)

        row = {
            "K": K,
            "numlocal": numlocal,
            "maxneighbor": maxneighbor,
            "algorithm": "clarans",
        }

        # -----------------------
        # Metrics
        # -----------------------
        if "ch" in metrics:
            row["CH"] = calinski_harabasz_score(X, labels)

        if "dbi" in metrics:
            row["DBI"] = davies_bouldin_score(X, labels)

        if "silhouette" in metrics:
            sil_scores = []
            for i in range(silhouette_n_repeats):
                idx = np.random.choice(N, eval_sample_size, replace=False)
                sil_scores.append(silhouette_score(X[idx], labels[idx]))

            row["Silhouette"] = np.mean(sil_scores)
            row["Silhouette_std"] = np.std(sil_scores)

        if "avg_medoid_dist" in metrics:
            distances = pairwise_distances(X, X[medoid_indices])
            row["AvgMedoidDist"] = np.mean(np.min(distances, axis=1))

        results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    return results_df
