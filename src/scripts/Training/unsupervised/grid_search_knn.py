import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from itertools import product
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans


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


def run_kmeans_grid(
    X,
    k_values,
    output_path: str,
    init_methods=("k-means++",),
    metrics=("ch", "dbi", "wcss", "silhouette"),
    algorithm="kmeans",  # "kmeans" | "minibatch_kmeans"
    max_iter=300,
    n_init=10,
    batch_size=1024,  # only for minibatch_kmeans
    random_state=42,
    silhouette_sample_size=50_000,
    silhouette_n_repeats=5,  # number of times to repeat silhouette evaluation
):
    if algorithm not in {"kmeans", "minibatch_kmeans"}:
        raise ValueError("algorithm must be 'kmeans' or 'minibatch_kmeans'")

    N = X.shape[0]
    results = []

    grid = list(product(k_values, init_methods))

    for K, init in tqdm(grid, desc=f"{algorithm} Grid Search"):

        # Fit KMeans / MiniBatchKMeans
        if algorithm == "kmeans":
            model = KMeans(
                n_clusters=K,
                init=init,
                max_iter=max_iter,
                n_init=n_init,
                random_state=random_state,
                algorithm="lloyd",
            )
        else:
            model = MiniBatchKMeans(
                n_clusters=K,
                init=init,
                max_iter=max_iter,
                batch_size=batch_size,
                n_init=n_init,
                random_state=random_state,
            )

        labels = model.fit_predict(X)

        row = {
            "K": K,
            "init": init,
            "algorithm": algorithm,
        }

        # -----------------------
        # Metrics
        # -----------------------
        if "ch" in metrics:
            row["CH"] = calinski_harabasz_score(X, labels)

        if "dbi" in metrics:
            row["DBI"] = davies_bouldin_score(X, labels)

        if "silhouette" in metrics:
            silhouette_scores = []
            for i in range(silhouette_n_repeats):
                Xs, ls = stratified_sample(
                    X,
                    labels,
                    total_samples=silhouette_sample_size,
                    random_state=random_state + i,  # different seed each time
                )
                silhouette_scores.append(silhouette_score(Xs, ls))

            row["Silhouette"] = np.mean(silhouette_scores)  # average over repeats
            row["Silhouette_std"] = np.std(
                silhouette_scores
            )  # optional: track variability

        if "wcss" in metrics:
            row["WCSS_per_point"] = model.inertia_ / N

        results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    return results_df
