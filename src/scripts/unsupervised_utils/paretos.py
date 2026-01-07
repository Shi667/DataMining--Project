import pandas as pd
import numpy as np  # Si vous l'utilisez aussi


def dominates(a, b, metrics, directions):
    """
    Returns True if solution a dominates solution b
    """
    better_or_equal = True
    strictly_better = False

    for m in metrics:
        if directions[m] == "max":
            if a[m] < b[m]:
                better_or_equal = False
                break
            elif a[m] > b[m]:
                strictly_better = True

        else:  # minimize
            if a[m] > b[m]:
                better_or_equal = False
                break
            elif a[m] < b[m]:
                strictly_better = True

    return better_or_equal and strictly_better


METRIC_DIRECTIONS = {
    "CH": "max",
    "DBI": "min",
    "Silhouette": "max",
    "WCSS_per_point": "min",
}


def extract_pareto_front(
    csv_path,
    metrics,
    directions=METRIC_DIRECTIONS,
):
    """
    Parameters
    ----------
    csv_path : str
        Path to CSV file containing grid search results
    metrics : list[str]
        Metrics to consider for Pareto dominance
    directions : dict
        Metric optimization directions ("min" or "max")

    Returns
    -------
    pareto_df : pd.DataFrame
        Non-dominated solutions
    """

    df = pd.read_csv(csv_path)

    # --- Safety checks ---
    for m in metrics:
        if m not in df.columns:
            raise ValueError(f"Metric '{m}' not found in CSV")
        if m not in directions:
            raise ValueError(f"No direction specified for metric '{m}'")

    pareto_mask = [True] * len(df)

    for i in range(len(df)):
        if not pareto_mask[i]:
            continue

        for j in range(len(df)):
            if i == j or not pareto_mask[j]:
                continue

            if dominates(df.iloc[j], df.iloc[i], metrics, directions):
                pareto_mask[i] = False
                break

    pareto_df = df[pareto_mask].reset_index(drop=True)
    return pareto_df
