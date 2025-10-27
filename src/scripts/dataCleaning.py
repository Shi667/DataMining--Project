import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def impute_categorical_rf(df, cat_col):
    """
    Impute missing categorical values using Random Forest Classifier.
    """
    df = df.copy()

    # Encode categorical values
    le = LabelEncoder()
    non_missing = df[cat_col].dropna().unique()
    le.fit(non_missing)
    df[cat_col + "_enc"] = df[cat_col].map(
        lambda x: le.transform([x])[0] if pd.notna(x) else None
    )

    # Separate data
    train = df[df[cat_col + "_enc"].notna()]
    test = df[df[cat_col + "_enc"].isna()]
    if test.empty:
        return df

    X_train = pd.get_dummies(
        train.drop(columns=[cat_col, cat_col + "_enc"]), drop_first=True
    )
    y_train = train[cat_col + "_enc"]
    X_test = pd.get_dummies(
        test.drop(columns=[cat_col, cat_col + "_enc"]), drop_first=True
    )
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Fill missing values
    df.loc[df[cat_col + "_enc"].isna(), cat_col + "_enc"] = y_pred
    df[cat_col] = le.inverse_transform(df[cat_col + "_enc"].astype(int))
    df.drop(columns=[cat_col + "_enc"], inplace=True)

    return df


from sklearn.impute import KNNImputer


def impute_numeric_knn(df, num_cols, n_neighbors=5):
    """
    Impute missing numeric values using KNN Imputer.
    """
    df = df.copy()
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[num_cols] = imputer.fit_transform(df[num_cols])
    return df
