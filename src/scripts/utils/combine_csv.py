import pandas as pd
import os

# --- Configuration ---
# Define the paths for your input files
file1_path = "../../../data/fire_dataset/viirs-jpss1_2024_Algeria.csv"
file2_path = "../../../data/fire_dataset/viirs-jpss1_2024_Tunisia.csv"

# Define the folder where you want to save the output
output_folder = "../../../data/fire_dataset/"

# Define the name for the combined CSV file
output_filename = "viirs-jpss1_2024_alg_Tun.csv"

# --- Execution ---

try:
    # 1. Read the two CSV files into pandas DataFrames
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # 2. Combine the DataFrames (vertically, appending df2 rows to df1)
    # The 'ignore_index=True' part resets the row indices in the final DataFrame
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # 3. Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # 4. Construct the full path for the output file
    output_path = os.path.join(output_folder, output_filename)

    # 5. Save the combined DataFrame to a new CSV file
    # 'index=False' prevents pandas from writing the DataFrame index as a column
    combined_df.to_csv(output_path, index=False)

    print(
        f"✅ Successfully combined '{os.path.basename(file1_path)}' and '{os.path.basename(file2_path)}'"
    )
    print(f"   Saved the result to: **{output_path}**")

except FileNotFoundError as e:
    print(f"❌ Error: One of the files was not found. Please check the paths.")
    print(f"   Details: {e}")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
