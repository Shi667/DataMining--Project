import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product

from sklearn_extra.cluster import KMedoids

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


def run_clara_grid(
    X,
    k_values,
    output_path: str,
    n_sampling_values=(40,),
    n_sampling_iter_values=(5,),
    max_iter_values=(300,),
    metrics=("ch", "dbi", "silhouette", "avg_medoid_dist"),
    eval_sample_size=50_000,
    silhouette_n_repeats=5,
    random_state=42,
):
    """
    CLARA (Clustering LARge Applications) implementation using KMedoids.
    
    CLARA samples multiple subsets, runs K-Medoids on each, and keeps the best medoids.
    """
    np.random.seed(random_state)

    # Ignore lat/lon for clustering
    id_cols = ["latitude", "longitude"]
    X_algo = X.drop(columns=[c for c in id_cols if c in X.columns])

    N = X_algo.shape[0]
    results = []

    grid = list(product(k_values, n_sampling_values, n_sampling_iter_values, max_iter_values))

    for K, n_sampling, n_sampling_iter, max_iter in tqdm(grid, desc="CLARA Grid Search"):
        
        best_medoids = None
        best_cost = float('inf')
        
        # CLARA algorithm: Multiple sampling iterations
        for iteration in range(n_sampling_iter):
            # Calculate sample size (standard CLARA formula)
            actual_sample_size = min(n_sampling + 2 * K, N)
            
            # Random sample
            sample_idx = np.random.choice(N, actual_sample_size, replace=False)
            X_sample = X_algo.iloc[sample_idx]

            # Run K-Medoids on sample
            kmedoids = KMedoids(
                n_clusters=K,
                method="alternate",
                init="k-medoids++",
                max_iter=max_iter,
                random_state=random_state + iteration,
            )

            kmedoids.fit(X_sample.values)
            
            # Map medoids back to full dataset indices
            medoid_indices = sample_idx[kmedoids.medoid_indices_]
            
            # Evaluate quality on FULL dataset
            distances = pairwise_distances(X_algo.values, X_algo.values[medoid_indices])
            cost = np.sum(np.min(distances, axis=1))
            
            # Keep best medoids across all samples
            if cost < best_cost:
                best_cost = cost
                best_medoids = medoid_indices
        
        # Assign all points to best medoids found
        labels = assign_labels_from_medoids(X_algo.values, best_medoids)

        row = {
            "K": K,
            "n_sampling": n_sampling,
            "n_sampling_iter": n_sampling_iter,
            "max_iter": max_iter,
            "algorithm": "clara",
            "best_cost": best_cost,
        }

        # Calculate metrics
        if "ch" in metrics:
            row["CH"] = calinski_harabasz_score(X_algo, labels)

        if "dbi" in metrics:
            row["DBI"] = davies_bouldin_score(X_algo, labels)

        if "silhouette" in metrics:
            sil_scores = []
            for i in range(silhouette_n_repeats):
                idx = np.random.choice(N, eval_sample_size, replace=False)
                sil_scores.append(silhouette_score(X_algo.values[idx], labels[idx]))

            row["Silhouette"] = np.mean(sil_scores)
            row["Silhouette_std"] = np.std(sil_scores)

        if "avg_medoid_dist" in metrics:
            distances = pairwise_distances(X_algo.values, X_algo.values[best_medoids])
            row["AvgMedoidDist"] = np.mean(np.min(distances, axis=1))

        results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    return results_df
