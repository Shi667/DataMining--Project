import matplotlib.pyplot as plt
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


def train_and_evaluate(model_class, X, **model_params):
    """
    General function to train any sklearn clustering model and evaluate metrics.

    Parameters:
    - model_class: The class object (e.g., KMeans, AgglomerativeClustering)
    - X: The dataset (numpy array)
    - **model_params: Dictionary of parameters for the model

    Returns:
    - model: The trained model object
    - labels: The cluster labels
    - metrics: Dictionary containing calculated scores
    """

    # 1. Instantiate and Train
    model = model_class(**model_params)

    # specific handling because some models use fit_predict, others fit
    try:
        labels = model.fit_predict(X)
    except AttributeError:
        model.fit(X)
        labels = model.labels_

    # 2. Safety Check: Metrics require at least 2 clusters
    n_clusters = len(set(labels)) - (
        1 if -1 in labels else 0
    )  # Exclude noise (-1) if using DBSCAN
    unique_labels = set(labels)

    if len(unique_labels) < 2:
        print(
            f"⚠️ Error: Model found {len(unique_labels)} clusters. Metrics require at least 2."
        )
        return model, labels, None

    # 3. Calculate Metrics
    # (We calculate all three because they are fast and useful)
    sil_score = silhouette_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    db_score = davies_bouldin_score(X, labels)

    results = {
        "Silhouette": sil_score,
        "Calinski-Harabasz": ch_score,
        "Davies-Bouldin": db_score,
    }

    # Check if model has inertia (Specific to K-Means)
    if hasattr(model, "inertia_"):
        results["Inertia"] = model.inertia_

    # 4. Print Report
    print(f"\n{'='*40}")
    print(f"Results for: {model_class.__name__}")
    print(f"Parameters: {model_params}")
    print(f"{'-'*40}")
    print(f"🔹 Silhouette Score:      {sil_score:.4f} (Close to +1 is best)")
    print(f"🔹 Calinski-Harabasz:     {ch_score:.1f}  (Higher is better)")
    print(f"🔹 Davies-Bouldin:        {db_score:.4f} (Lower is better)")

    if "Inertia" in results:
        print(f"🔹 Inertia (SSD):         {results['Inertia']:.1f}")

    print(f"{'='*40}\n")

    # 5. Plotting (Only if data is 2D)
    if X.shape[1] == 2:
        plt.figure(figsize=(8, 5))
        scatter = plt.scatter(
            X[:, 0], X[:, 1], c=labels, cmap="viridis", s=50, alpha=0.7
        )

        # Plot centers if available (KMeans only)
        if hasattr(model, "cluster_centers_"):
            plt.scatter(
                model.cluster_centers_[:, 0],
                model.cluster_centers_[:, 1],
                s=200,
                c="red",
                marker="X",
                label="Centroids",
            )

        plt.title(f"{model_class.__name__} (Silhouette: {sil_score:.2f})")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend(*scatter.legend_elements(), title="Clusters")
        plt.show()

    return model, labels, results
