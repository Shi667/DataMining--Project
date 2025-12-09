import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def train_and_evaluate(
    balanced_train_df,
    test_df,
    estimator,
    algo_name: str,
    params: dict,
    target_col="fire",
    verbose=True,
):

    if verbose:
        print("\n============================")
        print(f"🚀 Training: {algo_name}")
        print("🛠️ Parameters:", params)
        print("============================\n")

    # Split features and target
    X_train = balanced_train_df.drop(columns=[target_col])
    y_train = balanced_train_df[target_col].astype(int)

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].astype(int)

    # Set estimator params and fit
    model = estimator.set_params(**params)
    if verbose:
        print("📌 Training on full training set...")
    model.fit(X_train, y_train)

    def compute_metrics(X, y, label=""):
        y_pred = model.predict(X)

        try:
            y_proba = model.predict_proba(X)[:, 1]
        except AttributeError:
            y_proba = None

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_proba) if y_proba is not None else None,
            "tn_fp_fn_tp": confusion_matrix(y, y_pred).ravel(),
        }

        if verbose:
            print(f"\n====== 📊 {label.upper()} SET EVALUATION ======\n")
            for k, v in metrics.items():
                if k == "tn_fp_fn_tp":
                    continue
                if v is not None:
                    print(f"{k}: {v:.4f}")
            if metrics["roc_auc"] is not None:
                print(f"roc_auc: {metrics['roc_auc']:.4f}")

            print("\nConfusion matrix (tn, fp, fn, tp):\n", metrics["tn_fp_fn_tp"])

        return metrics

    train_metrics = compute_metrics(X_train, y_train, "Training")
    test_metrics = compute_metrics(X_test, y_test, "Test")

    # -----------------------
    # SAVE RESULTS TO CSV
    # -----------------------

    # Prepare dataframe
    result_dict = {
        "metric": [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "tn",
            "fp",
            "fn",
            "tp",
        ],
        "train": [
            train_metrics["accuracy"],
            train_metrics["precision"],
            train_metrics["recall"],
            train_metrics["f1"],
            train_metrics["roc_auc"],
            train_metrics["tn_fp_fn_tp"][0],
            train_metrics["tn_fp_fn_tp"][1],
            train_metrics["tn_fp_fn_tp"][2],
            train_metrics["tn_fp_fn_tp"][3],
        ],
        "test": [
            test_metrics["accuracy"],
            test_metrics["precision"],
            test_metrics["recall"],
            test_metrics["f1"],
            test_metrics["roc_auc"],
            test_metrics["tn_fp_fn_tp"][0],
            test_metrics["tn_fp_fn_tp"][1],
            test_metrics["tn_fp_fn_tp"][2],
            test_metrics["tn_fp_fn_tp"][3],
        ],
    }

    df_results = pd.DataFrame(result_dict)

    # Base filename
    filename = f"{algo_name}_results.csv"

    # If exists → create numbered suffix
    if os.path.exists(filename):
        i = 2
        while os.path.exists(f"{algo_name}_results_{i}.csv"):
            i += 1
        filename = f"{algo_name}_results_{i}.csv"

    df_results.to_csv(filename, index=False)

    if verbose:
        print(f"\n📁 Results saved to: {filename}")

    return {
        "model": model,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "csv_file": filename,
    }
