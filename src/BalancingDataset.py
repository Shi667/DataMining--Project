# balanced_dt_pipeline.py
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


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


import pandas as pd
import numpy as np
import os
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE


def hybrid_balance(
    train_df,
    target_col="fire",
    minority_target=1,
    desired_minority_prop=0.30,
    random_state=42,
    save_path=None,
    verbose=True,
):
    """
    Hybrid balancing:
    1) Undersample majority (NearMiss) to reduce size.
    2) SMOTE minority to achieve EXACT desired proportion.

    Output ALWAYS reaches desired_minority_prop exactly.
    """

    # -------------------------------------------------------
    # Safety check
    # -------------------------------------------------------
    if not (0.0 < desired_minority_prop < 1.0):
        raise ValueError("desired_minority_prop must be between 0 and 1.")

    X = train_df.drop(columns=[target_col])
    y = train_df[target_col].astype(int)

    counts = y.value_counts().to_dict()
    majority_class = y.mode()[0]
    minority_class = minority_target

    if minority_class not in counts:
        raise ValueError(f"Minority class {minority_class} not found in labels.")

    n_minority = counts.get(minority_class, 0)
    n_majority = counts.get(majority_class, 0)

    if verbose:
        print(
            f"Initial dataset: minority={n_minority}, majority={n_majority} (total={len(y)})"
        )

    # -------------------------------------------------------
    # STEP 1 — Undersample majority to speed up
    # -------------------------------------------------------

    # Target majority count for fast processing
    # Example: if desired minority = 30%, max maj allowed ≈ 70%
    max_majority_after_undersample = int(
        (n_minority / desired_minority_prop) * (1 - desired_minority_prop)
    )

    # Always cap so we never increase majority
    max_majority_after_undersample = min(max_majority_after_undersample, n_majority)

    if verbose:
        print(f"Undersampling majority {n_majority} → {max_majority_after_undersample}")

    rus = NearMiss(
        sampling_strategy={majority_class: max_majority_after_undersample},
        n_jobs=-1,
    )
    X_res, y_res = rus.fit_resample(X, y)

    # After undersampling
    counts_res = pd.Series(y_res).value_counts().to_dict()
    n_minority_res = counts_res.get(minority_class, 0)
    n_majority_res = counts_res.get(majority_class, 0)

    if verbose:
        print(
            f"After undersampling: minority={n_minority_res}, majority={n_majority_res}"
        )

    # -------------------------------------------------------
    # STEP 2 — SMOTE to reach EXACT desired proportion
    # -------------------------------------------------------

    p = desired_minority_prop

    # Exact math:
    # min_final / (min_final + maj_res) = p
    # → min_final = p * maj_res / (1 - p)
    min_final = int(np.ceil((p * n_majority_res) / (1 - p)))

    # must be >= current minority
    min_final = max(min_final, n_minority_res)

    if verbose:
        print(f"Target minority for EXACT {p*100:.0f}%: {min_final}")

    # SMOTE expects absolute class counts when using dict
    sampling_strategy = {minority_class: min_final}

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_bal, y_bal = smote.fit_resample(X_res, y_res)

    # -------------------------------------------------------
    # BUILD FINAL DF
    # -------------------------------------------------------
    balanced_df = pd.concat(
        [
            pd.DataFrame(X_bal, columns=X.columns).reset_index(drop=True),
            pd.Series(y_bal, name=target_col).reset_index(drop=True),
        ],
        axis=1,
    )

    # -------------------------------------------------------
    # Final checks
    # -------------------------------------------------------
    if verbose:
        final_counts = balanced_df[target_col].value_counts().to_dict()
        minority_final = final_counts.get(minority_class, 0)
        total_final = len(balanced_df)
        print(f"Final counts: {final_counts} (total={total_final})")
        print(f"Final minority proportion: {minority_final / total_final:.4f}")

    # -------------------------------------------------------
    # SAVE
    # -------------------------------------------------------
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        balanced_df.to_csv(save_path, index=False)
        if verbose:
            print(f"Saved balanced data to {save_path}")

    return balanced_df


def tune_and_evaluate(
    balanced_train_df,
    test_df,
    target_col="fire",
    cv_folds=5,
    random_state=42,
    scoring="f1",
    verbose=True,
):
    """
    Tune DecisionTree with GridSearchCV on balanced_train_df and evaluate on test_df.
    - scoring: metric used for GridSearchCV (e.g. 'f1', 'recall', 'roc_auc')
    Returns: dict with best_model, cv_results, test_metrics
    """
    X_train = balanced_train_df.drop(columns=[target_col])
    y_train = balanced_train_df[target_col].astype(int)

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].astype(int)

    dt = DecisionTreeClassifier(random_state=random_state)

    param_grid = {
        "criterion": ["entropy"],
        "max_depth": [5, 8, 12, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [5, 10],
        "class_weight": [
            None
        ],  # training was balanced; class_weight usually not necessary
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        estimator=dt,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    if verbose:
        print("Starting GridSearchCV on balanced training set...")
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    if verbose:
        print("Best params:", grid.best_params_)
        print(f"Best CV {scoring}: {grid.best_score_:.4f}")

    # Evaluate on untouched test set (original distribution)
    y_pred = best.predict(X_test)
    y_proba = None
    try:
        y_proba = best.predict_proba(X_test)[:, 1]
    except Exception:
        # tree might not have predict_proba for certain scikit-learn versions (rare)
        y_proba = None

    test_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }

    if verbose:
        print("\n--- Test set evaluation (untouched test set) ---")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall: {test_metrics['recall']:.4f}")
        print(f"F1: {test_metrics['f1']:.4f}")
        if test_metrics["roc_auc"] is not None:
            print(f"ROC AUC: {test_metrics['roc_auc']:.4f}")
        print("Confusion matrix:\n", test_metrics["confusion_matrix"])
        print("\nClassification report:\n", test_metrics["classification_report"])

    return {"best_model": best, "grid_search": grid, "test_metrics": test_metrics}


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
