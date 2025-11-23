import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


def stratified_split(df, target_col="fire", test_size=0.2, random_state=42):
    """
    Stratified train/test split preserving label distribution.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    train = pd.concat(
        [X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1
    )
    test = pd.concat(
        [X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1
    )
    return train, test
