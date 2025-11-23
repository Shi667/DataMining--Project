import pandas as pd
import glob
import re
from typing import List


def merge_viirs_data(file_patterns: List[str], output_filename: str) -> pd.DataFrame:
    """
    Finds CSV files matching specified patterns, reads them into pandas DataFrames,
    adds a 'Year' column based on the filename, concatenates them, and saves
    the result to a new CSV file.

    Args:
        file_patterns: A list of glob patterns (strings) to search for CSV files.
                       E.g., ["viirs-jpss1_*_Algeria.csv", "viirs-jpss1_*_Tunisia.csv"]
        output_filename: The name of the CSV file to save the merged data to.

    Returns:
        The merged pandas DataFrame.
    """

    # 1. Find and list all files matching the patterns
    files = []
    # If the input is already a list of specific file paths (not glob patterns),
    # use them directly. Otherwise, use glob.
    if all("*" not in p for p in file_patterns):
        files = file_patterns
    else:
        for pattern in file_patterns:
            files.extend(glob.glob(pattern))

    if not files:
        print(f"⚠️ Warning: No files found matching patterns: {file_patterns}")
        return pd.DataFrame()

    print(f"📂 Found {len(files)} files:")
    for f in files:
        print(" →", f)

    # 2. Read, extract year, and concatenate the files
    df_list = []
    year_pattern = re.compile(r"(\d{4})")  # Regex to find a 4-digit number (the year)

    for f in files:
        try:
            # Extract the year from the filename
            match = year_pattern.search(f)
            if match:
                year = int(match.group(1))
            else:
                # Fallback if no year is found
                print(
                    f"   Note: Could not find year in filename: {f}. Setting 'Year' to NaN."
                )
                year = None

            # Read the file
            df = pd.read_csv(f)

            # Add the new 'Year' column
            df["Year"] = year

            df_list.append(df)

        except Exception as e:
            print(f"❌ Error reading file {f} or adding 'Year' column: {e}")
            # Continue to the next file
            continue

    if not df_list:
        print("❌ Error: No DataFrames were successfully loaded.")
        return pd.DataFrame()

    try:
        merged_df = pd.concat(df_list, ignore_index=True)
    except Exception as e:
        print(f"❌ Error during concatenation: {e}")
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
    # --- Example 1: Using glob patterns (like in the original function) ---
    print("## Running Example 1 (Using Glob Patterns) ##")
    # This assumes there are files in the local directory matching the patterns
    patterns = ["viirs-jpss1_*_Algeria.csv", "viirs-jpss1_*_Tunisia.csv"]
    output_file = "viirs-jpss1_alg_Tun_Example1.csv"

    # NOTE: This part might fail if you don't have matching files in the current directory.
    merged_data_1 = merge_viirs_data(patterns, output_file)

    if not merged_data_1.empty:
        print("\nFirst 5 rows of the merged data (Example 1):")
        print(merged_data_1[["Year"]].head())  # Show just the new column and index

    # --- Example 2: Using the specific file paths from the end of your original script ---
    print("\n" + "=" * 50 + "\n")
    print("## Running Example 2 (Using Explicit File Paths) ##")

    # NOTE: These paths are *relative to the calling script* and might need adjustment.
    # The function now handles a list of explicit paths as input.
    explicit_files = glob.glob(
        "../../../data/fire_dataset/viirs-jpss1_*_Algeria.csv"
    ) + glob.glob("../../../data/fire_dataset/viirs-jpss1_*_Tunisia.csv")
    output_file_2 = "../../../data/fire_dataset/viirs-jpss1_alg_Tun.csv"

    # NOTE: This part might fail if the file paths don't exist.
    merged_data_2 = merge_viirs_data(
        file_patterns=explicit_files,
        output_filename=output_file_2,
    )

    if not merged_data_2.empty:
        print("\nFirst 5 rows of the merged data (Example 2):")
        print(merged_data_2[["Year"]].head())  # Show just the new column and index
