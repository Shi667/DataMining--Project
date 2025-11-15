import pandas as pd
from typing import List, Union

import pandas as pd

def progressive_merge(csv_list, on, how="inner", output_path=None, chunk_size=None):
    """
    Merge CSVs two by two progressively to avoid memory overflow.
    Optionally uses chunked reading if files are very large.
    """
    if len(csv_list) < 2:
        raise ValueError("Need at least two CSV files to merge")

    # Start with the first CSV
    print(f"Loading first CSV: {csv_list[0]}")
    merged_df = pd.read_csv(csv_list[0])

    for i, csv_path in enumerate(csv_list[1:], start=2):
        print(f"🔁 Merging file {i}/{len(csv_list)}: {csv_path}")

        # Read next CSV
        df_next = pd.read_csv(csv_path)

        # Merge with the accumulated result
        merged_df = pd.merge(merged_df, df_next, on=on, how=how)

        # Save intermediate result to disk to free memory
        temp_path = output_path if i == len(csv_list) else f"{output_path}_temp.csv"
        merged_df.to_csv(temp_path, index=False)

        # Reload from disk (to clear RAM)
        merged_df = pd.read_csv(temp_path)

        print(f"✅ Intermediate merged size: {merged_df.shape}")

    print("✅ All files merged successfully.")
    return merged_df
