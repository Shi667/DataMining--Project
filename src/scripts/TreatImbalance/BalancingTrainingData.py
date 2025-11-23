import os
import pandas as pd
import numpy as np
from imblearn.under_sampling import NearMiss, RandomUnderSampler
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
    Hybrid balancing: undersample majority quickly, then SMOTE to reach exact desired proportion.
    - train_df: dataframe containing training set (not touched test)
    - target_col: name of label column
    - desired_minority_prop: fraction of samples that should be minority class after balancing (0<prop<1)
    Returns balanced_df (pandas DataFrame).
    Saves balanced CSV if save_path provided.
    """
    if not (0.0 < desired_minority_prop < 1.0):
        raise ValueError("desired_minority_prop must be between 0 and 1 (exclusive).")

    X = train_df.drop(columns=[target_col])
    y = train_df[target_col].astype(int)

    # current counts
    counts = y.value_counts().to_dict()
    majority_class = y.mode()[0] if len(counts) > 0 else 0
    minority_class = minority_target
    if minority_class not in counts:
        raise ValueError(f"Minority class {minority_class} not in training labels.")

    n_total = len(y)
    n_minority = int(counts.get(minority_class, 0))
    n_majority = int(counts.get(majority_class, 0))

    if verbose:
        print(
            f"Training size: {n_total} (minority={n_minority}, majority={n_majority})"
        )
        print(f"Desired minority proportion: {desired_minority_prop:.2f}")

    # Step 1: Compute Neamiss undersampling
    approx_final_total = int(max(len(y), np.ceil(n_minority / desired_minority_prop)))
    # But if approx_final_total is huge because n_minority small, cap it to current_total to avoid explosion:
    approx_final_total = max(len(y), int(np.ceil(n_minority / desired_minority_prop)))
    max_majority_after_undersample = max(
        0, int(approx_final_total * (1 - desired_minority_prop))
    )
    max_majority_after_undersample = min(max_majority_after_undersample, n_majority)

    if max_majority_after_undersample < n_majority:
        if verbose:
            print(
                f"Undersampling majority from {n_majority} -> {max_majority_after_undersample} (fast reduction)."
            )
        rus = NearMiss(
            sampling_strategy={majority_class: max_majority_after_undersample},
            n_jobs=-1,
        )
        X_res, y_res = rus.fit_resample(X, y)
    else:
        if verbose:
            print("No undersampling applied (majority not reduced).")
        X_res, y_res = X.copy(), y.copy()

    # recompute counts after undersampling
    counts_res = pd.Series(y_res).value_counts().to_dict()
    n_minority_res = counts_res.get(minority_class, 0)
    n_majority_res = counts_res.get(majority_class, 0)
    if verbose:
        print(
            f"After undersampling: minority={n_minority_res}, majority={n_majority_res}"
        )

    # Step 2: Use SMOTE to increase minority to reach desired proportion exactly.

    final_total = int(np.ceil(n_minority_res / desired_minority_prop))
    final_total = max(final_total, n_minority_res + n_majority_res)
    m_final = int(np.ceil(desired_minority_prop * final_total))
    m_final = max(m_final, n_minority_res)
    if n_minority_res == 0:
        raise ValueError(
            "No minority samples present after undersampling; SMOTE cannot proceed."
        )
    smote_ratio = m_final / n_minority_res

    # imblearn.SMOTE sampling_strategy can be dict {minority: n_samples}
    sampling_strategy = {minority_class: int(m_final)}
    if verbose:
        print(
            f"SMOTE will generate minority to reach {m_final} samples (ratio={smote_ratio:.2f})."
        )

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_bal, y_bal = smote.fit_resample(X_res, y_res)

    balanced_df = pd.concat(
        [
            pd.DataFrame(X_bal, columns=X.columns).reset_index(drop=True),
            pd.Series(y_bal, name=target_col).reset_index(drop=True),
        ],
        axis=1,
    )

    if verbose:
        final_counts = balanced_df[target_col].value_counts().to_dict()
        print(f"Final balanced sizes: {final_counts} | total={len(balanced_df)}")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        balanced_df.to_csv(save_path, index=False)
        if verbose:
            print(f"Balanced training data saved to: {save_path}")

    return balanced_df
