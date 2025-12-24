import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import AgglomerativeClustering
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt


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


def supervised_feature_reduction(
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
    cluster_distance=0.3,
    use_autoencoder=False,
    n_ae_features=10,
    ae_epochs=50,
    percentage_data=1,
):
    df = pd.read_csv(csv_path)

    # ----------------------------------------
    # 🔑 ID columns (kept but NOT reduced)
    # ----------------------------------------
    id_cols = ["latitude", "longitude"]
    id_df = df[id_cols].copy()

    # Remove label if exists
    if "fire" in df.columns:
        df = df.drop(columns=["fire"])

    # Remove ID cols from feature reduction
    df = df.drop(columns=id_cols)

    df = df.sample(frac=percentage_data, random_state=42)
    id_df = id_df.loc[df.index]  # keep alignment

    print(f"Initial features (without lat/lon): {df.shape[1]}")

    # ------------------------------------------------------------
    # 1️⃣ Remove Low-variance features
    # ------------------------------------------------------------
    selector = VarianceThreshold(threshold=var_threshold)
    selector.fit(df)
    df = df.loc[:, selector.get_support()]
    print(f"After Variance Filter: {df.shape[1]}")

    # ------------------------------------------------------------
    # 2️⃣ Feature Clustering
    # ------------------------------------------------------------
    corr_matrix = df.corr().abs()
    distance_matrix = 1 - corr_matrix

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=cluster_distance,
        compute_distances=True,
    )
    clustering.fit(distance_matrix)
    cluster_labels = clustering.labels_

    selected_features = []
    for cluster_id in np.unique(cluster_labels):
        cluster_features = df.columns[cluster_labels == cluster_id]
        if len(cluster_features) == 1:
            selected_features.append(cluster_features[0])
        else:
            cluster_corr = corr_matrix.loc[cluster_features, cluster_features]
            best_feature = cluster_corr.sum().idxmax()
            selected_features.append(best_feature)

    df = df[selected_features]
    print(f"After Clustering: {df.shape[1]}")

    # ------------------------------------------------------------
    # 3️⃣ Optional: Autoencoder
    # ------------------------------------------------------------
    if use_autoencoder:
        print("Starting Autoencoder compression...")

        X_input = df.astype("float32").values
        input_dim = X_input.shape[1]

        input_layer = Input(shape=(input_dim,))
        e = Dense(128, activation="relu")(input_layer)
        e = Dense(64, activation="relu")(e)
        bottleneck = Dense(n_ae_features, activation="linear")(e)

        d = Dense(64, activation="relu")(bottleneck)
        d = Dense(128, activation="relu")(d)
        output_layer = Dense(input_dim, activation="linear")(d)

        autoencoder = Model(input_layer, output_layer)
        encoder = Model(input_layer, bottleneck)

        autoencoder.compile(optimizer="adam", loss="mse")

        history = autoencoder.fit(
            X_input,
            X_input,
            epochs=ae_epochs,
            batch_size=32,
            shuffle=True,
            validation_split=0.1,
            verbose=1,
        )

        encoded_data = encoder.predict(X_input)
        df = pd.DataFrame(
            encoded_data,
            columns=[f"ae_feature_{i}" for i in range(n_ae_features)],
            index=df.index,
        )

        print(f"After Autoencoder: {df.shape[1]} synthetic features")

    # ------------------------------------------------------------
    # 🔁 Reattach latitude & longitude
    # ------------------------------------------------------------
    final_df = pd.concat([id_df, df], axis=1)

    final_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
