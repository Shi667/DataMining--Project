import pandas as pd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import numpy as np


def process_fire_data(
    grid_path: str,
    fire_path: str,
    target_type: int,  # Assuming common fire sensor types, adjust if needed
    output_file: str,
):
    """
    Processes fire detection data: filters by a specified type, snaps detections
    to the nearest grid point, creates a binary fire indicator, handles duplicates,
    and merges the result with the full grid.

    Args:
        grid_path: Path to the CSV file containing grid points (columns: latitude, longitude).
        fire_path: Path to the CSV file containing fire detections
                   (must contain 'latitude', 'longitude', and the fire type column).
        fire_type_col: The name of the column in the fire data that specifies the type/sensor.
        target_type: The specific value in the fire_type_col to filter for (e.g., 'VIIRS').

    Returns:
        A DataFrame containing all grid points (latitude, longitude) with a
        'fire' column (1 if fire detected, 0 otherwise).
    """

    # --- 1. Load Data ---
    grid = pd.read_csv(grid_path)
    fire = pd.read_csv(fire_path)

    # --- 2. Filter Fire Data by Type ---
    # Delete rows where the type is not the target type
    fire = fire[fire["type"] == target_type].copy() # <-- FIX IS HERE
    # Check if any fire data remains after filtering
    if fire.empty:
        print(f"⚠️ Warning: No fire detections found for type '{target_type}'.")
        # Proceed with an empty 'fire' column which will become 0s after merging.

    # --- 3. KDTree Matching (Snap to Nearest Grid Cell) ---
    grid_coords = np.vstack((grid["latitude"], grid["longitude"])).T
    fire_coords = np.vstack((fire["latitude"], fire["longitude"])).T

    tree = cKDTree(grid_coords)
    # The tree.query operation can be slow if 'fire' is very large, but is necessary for snapping.
    distances, indices = tree.query(fire_coords, k=1)

    # Snap each fire point to its nearest grid cell's coordinates
    fire["grid_lat"] = grid.iloc[indices]["latitude"].values
    fire["grid_lon"] = grid.iloc[indices]["longitude"].values

    # --- 4. Prepare Snapped Data (Binary Fire Indicator & Deduplication) ---

    # Create a binary fire indicator (1 for every snapped detection)
    fire["fire"] = 1

    # Keep only the snapped coordinates and the new 'fire' indicator
    snapped = fire[["grid_lat", "grid_lon", "fire"]].copy()

    # Delete duplicated grid (lat, lon) pairs, keeping the first (which all have fire=1)
    # This ensures each unique grid cell has at most one fire entry.
    snapped.drop_duplicates(subset=["grid_lat", "grid_lon"], keep="first", inplace=True)

    # --- 5. Merge with Grid and Finalize ---

    # Merge with grid to include all grid cells.
    # 'left' ensures all grid cells are present.
    merged = pd.merge(
        grid,
        snapped,
        left_on=["latitude", "longitude"],
        right_on=["grid_lat", "grid_lon"],
        how="left",
    )

    # Fill NaN values in the 'fire' column with 0 (no fire detected at this grid cell)
    # The 'fire' column now contains 1 or 0, as requested.
    merged["fire"] = merged["fire"].fillna(0).astype(int)

    # Keep only the desired final columns: (latitude, longitude, fire)
    merged = merged[["latitude", "longitude", "fire"]]

    merged.to_csv(output_file, index=False)
    print(
        f"✅ Saved {merged.shape} grid points with binary fire data (1/0) to {output_file}"
    )




def treat_sensor_errors_soil(csv_path, output_path):
    """
    Clean soil dataset by:
      1. Replacing "-" strings with NaN.
      2. If a row contains ANY negative numeric values:
            * delete it if another row exists with same (latitude, longitude)
            * else replace negative values with NaN
    Overwrites the original CSV.
    """

    # Load dataset
    df = pd.read_csv(csv_path)
    if "TEXTURE_SOTER" in df.columns:
        df.drop(columns=["TEXTURE_SOTER"], inplace=True)
    # Replace "-" with NaN everywhere
    df = df.replace("-", np.nan)

    # Numeric columns to check for negatives
    numeric_cols = [
        "COARSE",
        "SAND",
        "SILT",
        "CLAY",
        "BULK",
        "REF_BULK",
        "ORG_CARBON",
        "PH_WATER",
        "TOTAL_N",
        "CN_RATIO",
        "CEC_SOIL",
        "CEC_CLAY",
        "CEC_EFF",
        "TEB",
        "BSAT",
        "ALUM_SAT",
        "ESP",
        "TCARBON_EQ",
        "GYPSUM",
        "ELEC_COND",
    ]

    # Convert numeric columns to float (important!)
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Detect rows with at least one negative value
    mask_negative = df[numeric_cols].lt(0).any(axis=1)
    bad_rows = df[mask_negative].copy()

    # Count rows per coordinate
    counts = df.groupby(["latitude", "longitude"]).size()

    rows_to_delete = []
    rows_to_fix = []

    # Process each bad row
    for idx, row in bad_rows.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]

        if counts.loc[(lat, lon)] > 1:
            rows_to_delete.append(idx)
        else:
            rows_to_fix.append(idx)

    # Delete rows entirely
    df_clean = df.drop(rows_to_delete)

    # Replace negative values with NaN in rows to fix
    df_clean.loc[rows_to_fix, numeric_cols] = df_clean.loc[
        rows_to_fix, numeric_cols
    ].where(df_clean.loc[rows_to_fix, numeric_cols] >= 0, np.nan)

    # Overwrite original CSV
    df_clean.to_csv(output_path, index=False)

    print("✔ Cleaning complete!")
    print(f"  Deleted rows : {len(rows_to_delete)}")
    print(f"  Fixed rows   : {len(rows_to_fix)}")


def impute_with_geo_zones(
    input_csv,
    num_cols=None,  # list of numeric columns
    cat_cols=None,  # list of categorical columns
    lat_col="latitude",
    lon_col="longitude",
    base_res=0.1,
    min_points=10,
    max_res=5.0,
    output_path="./dataCleaned.csv",
):
    df = pd.read_csv(input_csv)

    # Default empty lists if not provided
    num_cols = num_cols or []
    cat_cols = cat_cols or []

    # --- 1. Missing percentage ---
    missing_percent = df.isnull().mean() * 100
    print("Missing values (percent) per column :")
    print(missing_percent[missing_percent > 0])

    def get_zone_df(lat, lon, res):
        lat_min, lat_max = lat - res, lat + res
        lon_min, lon_max = lon - res, lon + res
        return df[
            (df[lat_col] >= lat_min)
            & (df[lat_col] <= lat_max)
            & (df[lon_col] >= lon_min)
            & (df[lon_col] <= lon_max)
        ]

    # --- 2. Process each column ---
    for col in df.columns:

        # Ignore latitude/longitude and columns with no missing
        if col in [lat_col, lon_col]:
            continue
        if missing_percent.get(col, 0) == 0:
            continue

        print(f"\n=== Imputing column: {col} ===")

        # --- Decide column type ---
        if col in num_cols:
            is_num = True
        elif col in cat_cols:
            is_num = False
        else:
            # fallback: detect automatically
            is_num = pd.api.types.is_numeric_dtype(df[col])

        for idx, row in df[df[col].isnull()].iterrows():
            lat, lon = row[lat_col], row[lon_col]
            resolution = base_res

            # Try expanding zone
            while resolution <= max_res:
                zone_df = get_zone_df(lat, lon, resolution)
                if len(zone_df) >= min_points:
                    break
                resolution *= 1.5

            # If zone empty → fallback global
            if len(zone_df) == 0:
                if is_num:
                    df.at[idx, col] = df[col].median()
                else:
                    df.at[idx, col] = df[col].mode(dropna=True).iat[0]
                continue

            # Numerical
            if is_num:
                df.at[idx, col] = zone_df[col].median()
            # Categorical
            else:
                mode_val = zone_df[col].mode(dropna=True)
                if not mode_val.empty:
                    df.at[idx, col] = mode_val.iat[0]
                else:
                    df.at[idx, col] = df[col].mode(dropna=True).iat[0]

        print(f"{col}: imputation done using geo-zones.")

        if output_path:
            df.to_csv(output_path, index=False)
            print(f"💾 Saved imputation to {output_path}")
