# balanced_dt_pipeline.py
import pandas as pd


if __name__ == "__main__":
    # Example usage
    # -------------
    # Put path to your CSV here and the name of the target column.
    csv_path = "../data/preprocessed/preprocessed_reduced_data.csv"  # <-- replace with your file

    target_col = "fire"  # <-- replace if your label column has another name
    test_size = 0.2
    desired_minority_prop = (
        0.30  # user-chosen: 0.30 means 30% minority in balanced training set
    )
    balanced_train_savepath = "balanced_train.csv"

    # Read CSV
    df = pd.read_csv(csv_path)
    print("Loaded dataset with shape:", df.shape)
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in CSV columns: {df.columns.tolist()}"
        )

    # 1) stratified train/test split
    train_df, test_df = stratified_split(
        df, target_col=target_col, test_size=test_size, random_state=42
    )
    """
    # 2) balance training set (hybrid undersample + SMOTE)
    balanced_train_df = hybrid_balance(
        train_df,
        target_col=target_col,
        minority_target=1,
        desired_minority_prop=desired_minority_prop,
        random_state=42,
        save_path=balanced_train_savepath,
        verbose=True,
    )
    """
    balanced_train_df = pd.read_csv("./balanced_train.csv")
    # 3) tune decision tree with cross-validation on balanced training set and evaluate on untouched test set
    results = tune_and_evaluate(
        balanced_train_df,
        test_df,
        target_col=target_col,
        cv_folds=2,
        random_state=42,
        scoring="recall",  # tune for f1 by default to handle imbalance tradeoff
        verbose=True,
    )

    # The trained best model is accessible via results['best_model'] and grid via results['grid_search']
    """

from scripts.dataPreprocessing.dataCleaning import duplicate_analysis
from scripts.statistics.firePourcentage import fired_pourcentage

# print(duplicate_analysis("./balanced_train.csv"))
print(duplicate_analysis("balanced_train.csv"))
fired_pourcentage("balanced_train.csv", "fire")
"""
