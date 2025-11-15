import pandas as pd
from sklearn.preprocessing import RobustScaler





def scalingEncodingDataset(csv_path, output_path,categorical_col=[]):
    """
    Preprocess HWSD soil dataset:
    - Drop TEXTURE_SOTER
    - Fix negative values
    - Replace nodata values
    - Impute with median
    - One-hot encode TEXTURE_USDA
    - Scale continuous features with RobustScaler
    """

    # ============================
    # 1. Load dataset
    # ============================
    df = pd.read_csv(csv_path)

    # ============================
    # 3. Identify continuous columns
    # ============================
    continuous_cols = [
        c for c in df.columns 
            if( c not in["latitude", "longitude"]  and  c not in categorical_col)
    ]

    # ============================
    # 4. One-Hot Encode categorical attributs
    # ============================

    # Get unique classes mapping BEFORE encoding
    for col in categorical_col : 
        df[col] = df[col].astype(int)
        unique_col= sorted(df[col].unique())
        print(f"{col} classes found:", unique_col)

        # One-Hot encode
        df = pd.get_dummies(df, columns=[col], prefix=f"{col[:3]}")

    # ============================
    # 6. Scale continuous variables
    # ============================
    scaler = RobustScaler()
    df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

    # ============================
    # 7. Save processed output
    # ============================
    df.to_csv(output_path, index=False)
    print(f"Saved preprocessed dataset → {output_path}")


