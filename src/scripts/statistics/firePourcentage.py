import pandas as pd
import os


def fired_pourcentage(csv_filepath, label_column_name="fire"):
    """
    Loads a CSV dataset, calculates key statistics, and prints the results.

    Args:
        csv_filepath (str): The path to the input CSV file.
        label_column_name (str): The name of the column that serves as the label.
                                 Defaults to 'fire'.

    Returns:
        None: The function prints the statistics directly.
    """
    # 1. Check if the file exists
    if not os.path.exists(csv_filepath):
        print(f"❌ Error: File not found at path: {csv_filepath}")
        return

    # 2. Load the dataset
    try:
        df = pd.read_csv(csv_filepath)
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        return

    # --- Core Statistics Calculation ---

    # Total lines (rows)
    total_lines = len(df)

    # Total features (columns)
    # We subtract 1 because the label column is included in df.shape[1]
    total_features = df.shape[1] - 1

    # Check if the label column exists
    if label_column_name not in df.columns:
        print(
            f"⚠️ Warning: Label column '{label_column_name}' not found in the dataset."
        )
        print(f"Total lines: **{total_lines}**")
        print(f"Total columns (features + label): **{df.shape[1]}**")
        return

    # --- Label Specific Statistics ---

    # Check for binary labels (0 and 1)
    label_counts = df[label_column_name].value_counts()
    count_0 = label_counts.get(0, 0)
    count_1 = label_counts.get(1, 0)

    # Calculate percentages
    pct_0 = (count_0 / total_lines) * 100 if total_lines > 0 else 0
    pct_1 = (count_1 / total_lines) * 100 if total_lines > 0 else 0

    # --- Output Results ---

    print("\n## 📊 Dataset Statistics")
    print("---------------------------------")
    print(f"**Total Lines (Rows):** **{total_lines}**")
    print(f"**Total Features:** **{total_features}**")
    print(f"**Label Column Name:** `{label_column_name}`")
    print("---------------------------------")

    print("### Label Distribution (Classes 0 and 1)")
    print(f"**Label 0 Count:** **{count_0}** (Non-Fire)")
    print(f"**Label 0 Percentage:** **{pct_0:.2f}%**")
    print("---")
    print(f"**Label 1 Count:** **{count_1}** (Fire)")
    print(f"**Label 1 Percentage:** **{pct_1:.2f}%**")
    print("---------------------------------")
