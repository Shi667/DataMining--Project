from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    ParameterGrid,
    cross_val_score,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    make_scorer,
)
import numpy as np


# ============================================================
# Custom GridSearch with LIVE CV METRICS PRINTING
# ============================================================
class LiveVerboseGridSearch(GridSearchCV):

    def _run_search(self, evaluate_candidates):
        """Print each parameter combination + CV metrics before fitting."""
        param_sets = list(ParameterGrid(self.param_grid))
        print(f"\n=== Total combinations: {len(param_sets)} ===")

        for i, params in enumerate(param_sets, start=1):
            print(f"\n🔧 Running combination {i}/{len(param_sets)}")
            print("Params:", params)
            print("-----------------------------------")

            # ============================================================
            # Compute cross-validation metrics manually
            # ============================================================
            estimator = self.estimator.set_params(**params)
            cv = self.cv

            print("📌 Cross-validation metrics:")

            # Recall (main scoring)
            recall_scores = cross_val_score(
                estimator,
                self.X_train_cv,
                self.y_train_cv,
                cv=cv,
                scoring="recall",
                n_jobs=self.n_jobs,
            )
            print(
                f"   • Recall:    {np.mean(recall_scores):.4f}  ± {np.std(recall_scores):.4f}"
            )

            # Accuracy
            acc_scores = cross_val_score(
                estimator,
                self.X_train_cv,
                self.y_train_cv,
                cv=cv,
                scoring="accuracy",
                n_jobs=self.n_jobs,
            )
            print(
                f"   • Accuracy:  {np.mean(acc_scores):.4f}  ± {np.std(acc_scores):.4f}"
            )

            # Precision
            precision_scores = cross_val_score(
                estimator,
                self.X_train_cv,
                self.y_train_cv,
                cv=cv,
                scoring="precision",
                n_jobs=self.n_jobs,
            )
            print(
                f"   • Precision: {np.mean(precision_scores):.4f}  ± {np.std(precision_scores):.4f}"
            )

            # F1
            f1_scores = cross_val_score(
                estimator,
                self.X_train_cv,
                self.y_train_cv,
                cv=cv,
                scoring="f1",
                n_jobs=self.n_jobs,
            )
            print(
                f"   • F1-score:  {np.mean(f1_scores):.4f}  ± {np.std(f1_scores):.4f}"
            )

            # ROC AUC (if classifier supports predict_proba)
            try:
                roc_scores = cross_val_score(
                    estimator,
                    self.X_train_cv,
                    self.y_train_cv,
                    cv=cv,
                    scoring="roc_auc",
                    n_jobs=self.n_jobs,
                )
                print(
                    f"   • ROC AUC:   {np.mean(roc_scores):.4f}  ± {np.std(roc_scores):.4f}"
                )
            except:
                print("   • ROC AUC:   Not available (no predict_proba)")

            print("-----------------------------------")

            evaluate_candidates([params])


# ============================================================
# Tune and evaluate function
# ============================================================
def tune_and_evaluate(
    balanced_train_df,
    test_df,
    estimator,  # algorithm passed as parameter
    param_grid,  # param grid passed as parameter
    target_col="fire",
    cv_folds=5,
    random_state=42,
    scoring="recall",
):
    # ==========  Split data  ==========
    X_train = balanced_train_df.drop(columns=[target_col])
    y_train = balanced_train_df[target_col].astype(int)
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].astype(int)

    # Save train for manual CV metrics
    LiveVerboseGridSearch.X_train_cv = X_train
    LiveVerboseGridSearch.y_train_cv = y_train

    # ==========  CV strategy  ==========
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # ==========  Grid Search  ==========
    grid = LiveVerboseGridSearch(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0,
        refit=True,
        error_score="raise",
    )

    print("\n🚀 Starting GridSearchCV...\n")
    grid.fit(X_train, y_train)

    print("\n🏆 Best parameters:", grid.best_params_)
    print(f"🏆 Best CV {scoring}: {grid.best_score_:.4f}\n")

    # ==========  Test Set Evaluation  ==========
    y_pred = grid.best_estimator_.predict(X_test)

    try:
        y_proba = grid.best_estimator_.predict_proba(X_test)[:, 1]
    except:
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

    print("\n====== 📊 TEST SET EVALUATION ======\n")
    for k, v in test_metrics.items():
        if k not in ["confusion_matrix", "classification_report", "roc_auc"]:
            print(f"{k}: {v:.4f}")

    if test_metrics["roc_auc"] is not None:
        print(f"roc_auc: {test_metrics['roc_auc']:.4f}")

    print("\nConfusion matrix:\n", test_metrics["confusion_matrix"])
    print("\nClassification report:\n", test_metrics["classification_report"])

    return {
        "best_model": grid.best_estimator_,
        "grid_search": grid,
        "test_metrics": test_metrics,
    }
