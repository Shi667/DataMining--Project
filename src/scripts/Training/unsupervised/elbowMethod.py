import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def plot_inertia_elbow(X, k_range, **kmeans_kwargs):
    """
    Trains KMeans for a range of k values, calculates inertia,
    and plots the Elbow curve for the user to analyze.

    Parameters:
    - X: The dataset (numpy array or dataframe)
    - k_range: A range or list of k values (e.g., range(2, 11))
    - **kmeans_kwargs: Any standard parameters for sklearn KMeans
                       (e.g., random_state=42, init='k-means++', max_iter=300)

    Returns:
    - inertias: A list of inertia values corresponding to k_range
    """
    inertias = []

    print(f"Running KMeans for k = {k_range[0]} to {k_range[-1]}...")

    for k in k_range:
        # We pass **kmeans_kwargs to unpack user parameters automatically
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    # --- Plotting the Elbow Curve ---
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, inertias, marker="o", linestyle="--", color="b")
    plt.title("Elbow Method (Inertia)")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.xticks(list(k_range))  # Force integers on x-axis
    plt.grid(True)
    plt.show()

    return inertias
