import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier


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
    importance_method="MI",  # "MI" or "RF"
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
