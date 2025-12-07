import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import AgglomerativeClustering


def analyze_correlation_variance(csv_path, target_col="fire", corr_threshold=0.9):
    """
    Loads dataset, computes correlation matrix, finds correlated pairs,
    and computes variance of each feature.
    """
    df = pd.read_csv(csv_path)

    # Remove target + coordinates if needed
    numeric_df = df.drop(columns=[target_col, "latitude", "longitude"], errors="ignore")

    # --- 1. Compute correlation matrix
    corr_matrix = numeric_df.corr().abs()

    # --- 2. Extract pairs above threshold
    correlated_pairs = []
    for col1 in corr_matrix.columns:
        for col2 in corr_matrix.columns:
            if col1 < col2:  # avoid repetition
                if corr_matrix.loc[col1, col2] >= corr_threshold:
                    correlated_pairs.append((col1, col2, corr_matrix.loc[col1, col2]))

    # --- 3. Variance of each feature
    variances = numeric_df.var().sort_values(ascending=True)

    return {
        "correlated_pairs": correlated_pairs,
        "variances": variances,
        "corr_matrix": corr_matrix,
    }


def reduce_features(
    csv_path,
    output_path="reduced_dataset.csv",
    target_col="fire",
    var_threshold=0.01,
    corr_threshold=0.9,
    importance_method="RF",  # "MI" or "RF"
    top_k=20,  # number of best features to keep
):
    df = pd.read_csv(csv_path)

    X = df.drop(columns=[target_col], errors="ignore")
    y = df[target_col]

    # Remove non-informative ID columns
    X = X.drop(columns=["latitude", "longitude"], errors="ignore")

    # --- 1. Remove low variance features
    variances = X.var()
    keep_var = variances[variances > var_threshold].index
    X = X[keep_var]

    # --- 2. Remove highly correlated features smartly
    corr_matrix = X.corr().abs()

    # correlation of each feature with the target
    target_corr = df[X.columns].corrwith(y).abs()

    to_drop = set()
    cols = list(corr_matrix.columns)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1, col2 = cols[i], cols[j]
            if corr_matrix.loc[col1, col2] > corr_threshold:
                # drop the one LESS correlated with the label
                if target_corr[col1] >= target_corr[col2]:
                    to_drop.add(col2)
                else:
                    to_drop.add(col1)

    X = X.drop(columns=list(to_drop), errors="ignore")

    # --- 3. Feature importance (MI or RandomForest)
    if importance_method == "MI":
        scores = mutual_info_classif(X, y)
    else:
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X, y)
        scores = rf.feature_importances_

    importance_df = pd.DataFrame({"feature": X.columns, "score": scores}).sort_values(
        by="score", ascending=False
    )

    # Keep top_k best features
    selected_features = importance_df.head(top_k)["feature"].tolist()

    # --- Output reduced dataset
    reduced_df = df[selected_features + [target_col]]
    reduced_df.to_csv(output_path, index=False)

    return {
        "selected_features": selected_features,
        "importance_table": importance_df,
        "output_path": output_path,
    }


def unsupervised_feature_reduction(
    csv_path,
    output_path="reduced_dataset.csv",
    var_threshold=0.01,
    corr_threshold=0.95,
    cluster_distance=0.2,
):
    """
    csv_path
    output_path
    var_threshold: threshold for low variance removal
    corr_threshold: threshold for correlation-based elimination
    cluster_distance: distance threshold for agglomerative clustering (0-1)
    """
    df = pd.read_csv(csv_path)
    print("Initial number of features:", df.shape[1])

    # ------------------------------------------------------------
    # 1️⃣ Remove Low-variance features
    # ------------------------------------------------------------
    selector = VarianceThreshold(threshold=var_threshold)
    selector.fit(df)

    low_var_mask = selector.get_support()
    df = df.loc[:, low_var_mask]

    print("After low-variance filter:", df.shape[1])

    # ------------------------------------------------------------
    # 2️⃣ Remove Highly Correlated Features
    #     Keep the one with highest variance
    # ------------------------------------------------------------
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop_corr = set()
    for col in upper.columns:
        high_corr = upper[col][upper[col] > corr_threshold].index

        for other in high_corr:
            # Compare variance of col vs other
            if df[col].var() >= df[other].var():
                to_drop_corr.add(other)
            else:
                to_drop_corr.add(col)

    df = df.drop(columns=list(to_drop_corr))
    print("After correlation filter:", df.shape[1])

    # ------------------------------------------------------------
    # 3️⃣ Feature Clustering (redundancy reduction)
    # ------------------------------------------------------------
    # Recompute correlation matrix after previous reductions
    corr_matrix = df.corr().abs()

    # Convert correlation to distance
    distance_matrix = 1 - corr_matrix

    # Clustering features based on similarity
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=cluster_distance,
        compute_distances=True,
    )

    clustering.fit(distance_matrix)

    cluster_labels = clustering.labels_

    # Select 1 representative feature per cluster (highest variance)
    selected_features = []
    for cluster_id in np.unique(cluster_labels):
        cluster_features = df.columns[cluster_labels == cluster_id]

        # choose feature with highest variance
        best_feature = df[cluster_features].var().idxmax()
        selected_features.append(best_feature)

    df = df[selected_features]
    print("After feature clustering:", df.shape[1])
    df.to_csv(output_path, index=False)
