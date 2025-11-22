import pandas as pd
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder


# =====================================================================
# 1) SCALING (numeric features only)
# =====================================================================
def scale_dataset(csv_path, output_path, exclude_cols=None):
    """
    Scales numeric columns except those specified in exclude_cols.
    """
    df = pd.read_csv(csv_path)

    if exclude_cols is None:
        exclude_cols = []

    # numeric columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    # columns to scale = numeric - excluded
    scale_cols = [c for c in numeric_cols if c not in exclude_cols]

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    df.to_csv(output_path, index=False)
    return df


# =====================================================================
# 2) ONE-HOT ENCODING
# =====================================================================
def one_hot_encode(csv_path, categorical_cols, label_col, output_path):
    """
    Reads a CSV and applies one-hot encoding.
    Ensures the new encoded features appear BEFORE the label column.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    # Keep label aside
    label_series = df[label_col]

    # Apply OHE only on selected columns
    df_no_label = df.drop(columns=[label_col])
    df_encoded = pd.get_dummies(df_no_label, columns=categorical_cols, drop_first=False)

    # Reassemble with label at the end
    df_encoded[label_col] = label_series

    df_encoded.to_csv(output_path, index=False)
    return df_encoded


# =====================================================================
# 3) TARGET ENCODING
# =====================================================================
def target_encode(csv_path, categorical_cols, target_col, output_path):
    """
    Reads a CSV, applies Target Encoding on the given categorical columns,
    using the specified target column.
    Saves result as a new CSV.
    """
    df = pd.read_csv(csv_path)

    te = TargetEncoder(cols=categorical_cols)
    df[categorical_cols] = te.fit_transform(df[categorical_cols], df[target_col])

    df.to_csv(output_path, index=False)
    return df
