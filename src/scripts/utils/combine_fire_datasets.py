import pandas as pd
import glob
from typing import List


def merge_viirs_data(file_patterns: List[str], output_filename: str) -> pd.DataFrame:
    """
    Finds CSV files matching specified patterns, reads them into pandas DataFrames,
    concatenates them, and saves the result to a new CSV file.

    Args:
        file_patterns: A list of glob patterns (strings) to search for CSV files.
                       E.g., ["viirs-jpss1_*_Algeria.csv", "viirs-jpss1_*_Tunisia.csv"]
        output_filename: The name of the CSV file to save the merged data to.

    Returns:
        The merged pandas DataFrame.
    """

    # 1. Find and list all files matching the patterns
    files = []
    for pattern in file_patterns:
        files.extend(glob.glob(pattern))

    if not files:
        print(f"⚠️ Warning: No files found matching patterns: {file_patterns}")
        return pd.DataFrame()

    print(f"📂 Found {len(files)} files:")
    for f in files:
        print(" →", f)

    # 2. Read and concatenate the files
    try:
        df_list = [pd.read_csv(f) for f in files]
        merged_df = pd.concat(df_list, ignore_index=True)
    except Exception as e:
        print(f"❌ Error during file reading or concatenation: {e}")
        return pd.DataFrame()

    # 3. Save the result
    try:
        merged_df.to_csv(output_filename, index=False)
    except Exception as e:
        print(f"❌ Error saving to {output_filename}: {e}")
        return merged_df  # Return the DF even if saving fails

    print(f"\n✅ Done! Saved as **{output_filename}**")
    print(f"📈 Total rows: **{len(merged_df)}**")

    return merged_df


# --- Example Usage ---
if __name__ == "__main__":
    # The original file matching logic is passed as a list of patterns
    patterns = ["viirs-jpss1_*_Algeria.csv", "viirs-jpss1_*_Tunisia.csv"]
    output_file = "viirs-jpss1_alg_Tun.csv"

    merged_data = merge_viirs_data(patterns, output_file)

    # You can now work with the 'merged_data' DataFrame
    if not merged_data.empty:
        print("\nFirst 5 rows of the merged data:")
        print(merged_data.head())


files = glob.glob("../../../data/fire_dataset/viirs-jpss1_*_Algeria.csv") + glob.glob(
    "../../../data/fire_dataset/viirs-jpss1_*_Tunisia.csv"
)


merge_viirs_data(
    file_patterns=files,
    output_filename="../../../data/fire_dataset/viirs-jpss1_alg_Tun.csv",
)
