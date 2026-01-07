import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from scripts.unsupervisedAlgofromScratch.kmeans.kmeans import KMeansFromScratch
from scripts.unsupervisedAlgofromScratch.dbscan.hdbscan import HDBSCANFromScratch, HDBSCANOptimized
from scripts.unsupervisedAlgofromScratch.clara.clara import CLARAFromScratch


def clustering_with_map(
    df,
    shapefile_path,
    algorithm="kmeans_scratch",
    # K-means parameters
    k_value=None,
    max_iter=300,
    n_init=10,
    batch_size=1024,
    # HDBSCAN parameters
    min_cluster_size=5,
    min_samples=None,
    # CLARA parameters
    n_sampling=40,
    n_sampling_iter=5,
    # Common parameters
    metrics=("ch", "dbi", "wcss", "silhouette"),
    random_state=42,
    silhouette_sample_size=50_000,
    silhouette_n_repeats=5,
):
    """
    Clustering avec visualisation sur carte
    
    Parameters:
    -----------
    df : DataFrame, données avec latitude/longitude
    shapefile_path : str, chemin vers le shapefile
    algorithm : str, 'kmeans_scratch' | 'minibatch_kmeans' | 'hdbscan_scratch' | 'clara_scratch'
    k_value : int, nombre de clusters (pour K-means et CLARA)
    max_iter : int, nombre maximum d'itérations (K-means)
    n_init : int, nombre d'initialisations (K-means)
    batch_size : int, taille de batch pour MiniBatchKMeans
    min_cluster_size : int, pour HDBSCAN - taille minimale des clusters
    min_samples : int, pour HDBSCAN - voisins pour core points (défaut: min_cluster_size)
    n_sampling : int, pour CLARA - taille de l'échantillon (typiquement 40 + 2*k)
    n_sampling_iter : int, pour CLARA - nombre d'itérations d'échantillonnage
    metrics : tuple, métriques à calculer
    random_state : int, seed
    silhouette_sample_size : int, taille échantillon pour silhouette
    silhouette_n_repeats : int, répétitions pour silhouette
    
    Returns:
    --------
    gdf_points : GeoDataFrame avec les clusters
    metrics_result : dict avec les métriques calculées
    """
    
    print("\n" + "="*70)
    if algorithm == "hdbscan_scratch":
        print(f"🗺️  HDBSCAN CLUSTERING AVEC CARTE (min_cluster_size={min_cluster_size}, min_samples={min_samples})")
    elif algorithm == "clara_scratch":
        print(f"🗺️  CLARA CLUSTERING AVEC CARTE (K={k_value})")
    else:
        print(f"🗺️  K-MEANS CLUSTERING AVEC CARTE (K={k_value})")
    print("="*70)
    
    # -----------------------------
    # Separate geometry & features
    # -----------------------------
    geo_cols = ["latitude", "longitude"]
    X_algo = df.drop(columns=geo_cols)
    coords = df[geo_cols]
    
    print(f"📊 Nombre de points: {len(df):,}")
    print(f"📈 Features utilisées: {X_algo.shape[1]}")
    print(f"🔧 Algorithme: {algorithm}")

    # -----------------------------
    # Fit clustering
    # -----------------------------
    if algorithm == "kmeans_scratch":
        if k_value is None:
            raise ValueError("k_value must be specified for K-means")
        print(f"⚙️  Entraînement K-means from scratch (n_init={n_init})...")
        model = KMeansFromScratch(
            n_clusters=k_value,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
        )
        labels = model.fit_predict(X_algo)
        print(f"✅ Convergence en {model.n_iter_} itérations")
        
    elif algorithm == "minibatch_kmeans":
        if k_value is None:
            raise ValueError("k_value must be specified for K-means")
        print(f"⚙️  Entraînement MiniBatchKMeans (batch_size={batch_size})...")
        model = MiniBatchKMeans(
            n_clusters=k_value,
            max_iter=max_iter,
            batch_size=batch_size,
            n_init=n_init,
            random_state=random_state,
        )
        labels = model.fit_predict(X_algo)
        
    elif algorithm == "hdbscan_scratch":
        print(f"⚙️  Entraînement HDBSCAN from scratch...")
        model = HDBSCANOptimized(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )
        labels = model.fit_predict(X_algo.values)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"✅ Clustering terminé: {n_clusters} clusters, {n_noise} points de bruit")
        
    elif algorithm == "clara_scratch":
        if k_value is None:
            raise ValueError("k_value must be specified for CLARA")
        print(f"⚙️  Entraînement CLARA from scratch (n_sampling={n_sampling}, n_sampling_iter={n_sampling_iter})...")
        model = CLARAFromScratch(
            n_clusters=k_value,
            n_sampling=n_sampling,
            n_sampling_iter=n_sampling_iter,
            random_state=random_state,
        )
        labels = model.fit_predict(X_algo.values)
        print(f"✅ Clustering terminé: coût = {model.best_cost_:.2f}")
        
    else:
        raise ValueError("algorithm must be 'kmeans_scratch', 'minibatch_kmeans', 'hdbscan_scratch', or 'clara_scratch'")

    # -----------------------------
    # Metrics
    # -----------------------------
    print("\n📊 Calcul des métriques...")
    metrics_result = {}
    
    # Filter out noise points for metrics (HDBSCAN)
    if algorithm == "hdbscan_scratch":
        mask = labels != -1
        X_for_metrics = X_algo.values[mask]
        labels_for_metrics = labels[mask]
        
        if len(set(labels_for_metrics)) < 2:
            print("⚠️  Moins de 2 clusters trouvés, métriques non calculables")
            metrics_result = {"n_clusters": len(set(labels_for_metrics)), "n_noise": n_noise}
        else:
            if "ch" in metrics:
                metrics_result["CH"] = calinski_harabasz_score(X_for_metrics, labels_for_metrics)
                print(f"   📈 Calinski-Harabasz: {metrics_result['CH']:.2f}")

            if "dbi" in metrics:
                metrics_result["DBI"] = davies_bouldin_score(X_for_metrics, labels_for_metrics)
                print(f"   📉 Davies-Bouldin: {metrics_result['DBI']:.3f}")

            if "silhouette" in metrics:
                print(f"   🎯 Calcul Silhouette ({silhouette_n_repeats} répétitions)...")
                sil_scores = []
                for i in range(silhouette_n_repeats):
                    sample_size = min(silhouette_sample_size, len(X_for_metrics))
                    idx = np.random.choice(len(X_for_metrics), sample_size, replace=False)
                    sil_scores.append(silhouette_score(X_for_metrics[idx], labels_for_metrics[idx]))
                metrics_result["Silhouette"] = np.mean(sil_scores)
                metrics_result["Silhouette_std"] = np.std(sil_scores)
                print(f"   🎯 Silhouette: {metrics_result['Silhouette']:.4f} (±{metrics_result['Silhouette_std']:.4f})")
            
            metrics_result["n_clusters"] = n_clusters
            metrics_result["n_noise"] = n_noise
    else:
        # K-means/CLARA metrics
        if "ch" in metrics:
            metrics_result["CH"] = calinski_harabasz_score(X_algo, labels)
            print(f"   📈 Calinski-Harabasz: {metrics_result['CH']:.2f}")

        if "dbi" in metrics:
            metrics_result["DBI"] = davies_bouldin_score(X_algo, labels)
            print(f"   📉 Davies-Bouldin: {metrics_result['DBI']:.3f}")

        if "wcss" in metrics and hasattr(model, 'inertia_'):
            metrics_result["WCSS_per_point"] = model.inertia_ / len(X_algo)
            print(f"   💾 WCSS/point: {metrics_result['WCSS_per_point']:.3f}")

        if "silhouette" in metrics:
            print(f"   🎯 Calcul Silhouette ({silhouette_n_repeats} répétitions)...")
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
            print(f"   🎯 Silhouette: {metrics_result['Silhouette']:.4f} (±{metrics_result['Silhouette_std']:.4f})")

    # Distribution des clusters
    unique, counts = np.unique(labels, return_counts=True)
    n_display_clusters = len(unique) - (1 if -1 in unique else 0)
    print(f"\n📊 Distribution des clusters:")
    for cluster_id, count in zip(unique, counts):
        if cluster_id == -1:
            print(f"   Bruit: {count:,} points ({100*count/len(labels):.1f}%)")
        else:
            print(f"   Cluster {cluster_id}: {count:,} points ({100*count/len(labels):.1f}%)")

    # -----------------------------
    # Create GeoDataFrame
    # -----------------------------
    print("\n🗺️  Création de la carte...")
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
    gdf_shape.plot(ax=ax, color="white", edgecolor="black", linewidth=0.5)
    
    # Ensure cluster column is categorical for proper legend
    gdf_points['cluster'] = gdf_points['cluster'].astype('category')
    
    gdf_points.plot(
        ax=ax,
        column="cluster",
        cmap="tab20",
        markersize=3,
        categorical=True,
        legend=True,
        legend_kwds={
            'loc': 'upper left', 
            'bbox_to_anchor': (1, 1),
            'title': 'Cluster'
        },
    )

    # Titre avec métriques
    if algorithm == "hdbscan_scratch":
        title = f"HDBSCAN from scratch | min_cluster_size={min_cluster_size}, min_samples={min_samples}"
        title += f"\nClusters: {n_display_clusters} | Noise: {n_noise}"
    elif algorithm == "clara_scratch":
        title = f"CLARA from scratch | K={k_value}, n_sampling={n_sampling}, n_sampling_iter={n_sampling_iter}"
    else:
        title = f"K-means ({algorithm}) | K={k_value}"
        if algorithm == "kmeans_scratch":
            title += f" | {model.n_iter_} iter"
    
    if metrics_result:
        metrics_str = " | ".join(
            f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics_result.items() 
            if k not in ['n_clusters', 'n_noise', 'Silhouette_std']
        )
        if metrics_str:
            title += "\n" + metrics_str
    
    ax.set_title(title, fontsize=12, pad=20)
    ax.axis("off")
    plt.tight_layout()
    plt.show()
    
    print("✅ Carte générée avec succès!")
    print("="*70 + "\n")

    return gdf_points, metrics_result


# Alias pour compatibilité avec votre code existant
def kmeans_with_map_s(*args, **kwargs):
    """Wrapper pour compatibilité arrière"""
    return clustering_with_map(*args, **kwargs)
