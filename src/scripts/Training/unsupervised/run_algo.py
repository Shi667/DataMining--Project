import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
import hdbscan
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def kmeans_with_map(
    df,
    shapefile_path,
    k_value,
    init_methods=("k-means++",),
    metrics=("ch", "dbi", "wcss", "silhouette"),
    algorithm="kmeans",  # "kmeans" | "minibatch_kmeans"
    max_iter=300,
    n_init=10,
    batch_size=1024,
    random_state=42,
    silhouette_sample_size=50_000,
    silhouette_n_repeats=5,
):
    import numpy as np

    # -----------------------------
    # Separate geometry & features
    # -----------------------------
    geo_cols = ["latitude", "longitude"]
    X_algo = df.drop(columns=geo_cols)
    coords = df[geo_cols]

    # -----------------------------
    # Fit clustering
    # -----------------------------
    if algorithm == "kmeans":
        model = KMeans(
            n_clusters=k_value,
            init=init_methods[0],
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
            algorithm="lloyd",
        )
    else:
        model = MiniBatchKMeans(
            n_clusters=k_value,
            init=init_methods[0],
            max_iter=max_iter,
            batch_size=batch_size,
            n_init=n_init,
            random_state=random_state,
        )

    labels = model.fit_predict(X_algo)

    # -----------------------------
    # Metrics
    # -----------------------------
    metrics_result = {}

    if "ch" in metrics:
        metrics_result["CH"] = calinski_harabasz_score(X_algo, labels)

    if "dbi" in metrics:
        metrics_result["DBI"] = davies_bouldin_score(X_algo, labels)

    if "wcss" in metrics:
        metrics_result["WCSS_per_point"] = model.inertia_ / len(X_algo)

    if "silhouette" in metrics:
        sil_scores = []
        for i in range(silhouette_n_repeats):
            idx = np.random.choice(
                len(X_algo),
                min(silhouette_sample_size, len(X_algo)),
                replace=False,
            )
            sil_scores.append(silhouette_score(X_algo.iloc[idx], labels[idx]))
        metrics_result["Silhouette"] = np.mean(sil_scores)
        metrics_result["Silhouette_std"] = np.std(sil_scores)

    # -----------------------------
    # Create GeoDataFrame
    # -----------------------------
    gdf_points = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(coords["longitude"], coords["latitude"]),
        crs="EPSG:4326",
    )
    gdf_points["cluster"] = labels

    # -----------------------------
    # Load shapefile
    # -----------------------------
    gdf_shape = gpd.read_file(shapefile_path)
    gdf_shape = gdf_shape.to_crs(gdf_points.crs)

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    gdf_shape.plot(ax=ax, color="white", edgecolor="black")
    gdf_points.plot(
        ax=ax,
        column="cluster",
        cmap="tab20",
        markersize=3,
        legend=True,
    )

    ax.set_title(
        f"KMeans ({algorithm}) | K={k_value}\n"
        + " | ".join(f"{k}: {v:.3f}" for k, v in metrics_result.items())
    )
    ax.axis("off")
    plt.show()

    return gdf_points, metrics_result


def hdbscan_with_map(
    df,
    shapefile_path,
    min_cluster_size,
    min_samples,
    metrics=("ch", "dbi", "silhouette"),
    metric="euclidean",
    cluster_selection_method="eom",
    silhouette_sample_size=50_000,
    silhouette_n_repeats=5,
    random_state=42,
):

    # -----------------------------
    # Separate geometry & features
    # -----------------------------
    geo_cols = ["latitude", "longitude"]
    X_algo = df.drop(columns=geo_cols)
    coords = df[geo_cols]

    # -----------------------------
    # Fit HDBSCAN
    # -----------------------------
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
    )

    labels = clusterer.fit_predict(X_algo)

    # -----------------------------
    # Metrics (ignore noise)
    # -----------------------------
    metrics_result = {}

    mask = labels != -1
    X_clean = X_algo[mask]
    labels_clean = labels[mask]

    if len(np.unique(labels_clean)) >= 2:
        if "ch" in metrics:
            metrics_result["CH"] = calinski_harabasz_score(X_clean, labels_clean)

        if "dbi" in metrics:
            metrics_result["DBI"] = davies_bouldin_score(X_clean, labels_clean)

        if "silhouette" in metrics:
            sil_scores = []
            for i in range(silhouette_n_repeats):
                idx = np.random.choice(
                    len(X_clean),
                    min(silhouette_sample_size, len(X_clean)),
                    replace=False,
                )
                sil_scores.append(
                    silhouette_score(X_clean.iloc[idx], labels_clean[idx])
                )
            metrics_result["Silhouette"] = np.mean(sil_scores)
            metrics_result["Silhouette_std"] = np.std(sil_scores)

    # -----------------------------
    # GeoDataFrame
    # -----------------------------
    gdf_points = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(coords["longitude"], coords["latitude"]),
        crs="EPSG:4326",
    )
    gdf_points["cluster"] = labels

    # -----------------------------
    # Load shapefile
    # -----------------------------
    gdf_shape = gpd.read_file(shapefile_path)
    gdf_shape = gdf_shape.to_crs(gdf_points.crs)

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    gdf_shape.plot(ax=ax, color="white", edgecolor="black")
    gdf_points.plot(
        ax=ax,
        column="cluster",
        cmap="tab20",
        markersize=3,
        legend=True,
    )

    ax.set_title(
        f"HDBSCAN | min_cs={min_cluster_size}, min_samples={min_samples}\n"
        + " | ".join(f"{k}: {v:.3f}" for k, v in metrics_result.items())
    )
    ax.axis("off")
    plt.show()

    return gdf_points, metrics_result
