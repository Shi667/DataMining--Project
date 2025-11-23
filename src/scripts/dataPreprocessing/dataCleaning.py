import pandas as pd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import numpy as np


def process_fire_data(
    grid_path: str,
    fire_path: str,
    target_type: int,
    output_file: str,
    no_fire_year: int = 2024,  # default year when no fire
):
    """
    Processes fire detection data and attaches the fire year to the grid.
    Non-fire grid cells are assigned a default year (2024).
    """

    # --- 1. Load Data ---
    grid = pd.read_csv(grid_path)
    fire = pd.read_csv(fire_path)

    # --- 2. Filter Fire Data by Sensor Type ---
    fire = fire[fire["type"] == target_type].copy()
    if fire.empty:
        print(f"⚠️ Warning: No fire detections found for type '{target_type}'.")

    # --- 3. KDTree Matching (Snap Fire Points to Nearest Grid Point) ---
    grid_coords = np.vstack((grid["latitude"], grid["longitude"])).T
    fire_coords = np.vstack((fire["latitude"], fire["longitude"])).T

    tree = cKDTree(grid_coords)
    distances, indices = tree.query(fire_coords, k=1)

    # Assign snapped coordinates
    fire["grid_lat"] = grid.iloc[indices]["latitude"].values
    fire["grid_lon"] = grid.iloc[indices]["longitude"].values

    # --- 4. Keep Year + Deduplicate ---
    fire["fire"] = 1  # binary indicator

    # Keep the year of the fire!
    snapped = fire[["grid_lat", "grid_lon", "fire", "year"]].copy()

    # If multiple fires snap to the same grid → keep the earliest fire year
    snapped.sort_values(by="year", inplace=True)
    snapped.drop_duplicates(subset=["grid_lat", "grid_lon"], keep="first", inplace=True)

    # --- 5. Merge With Grid ---
    merged = pd.merge(
        grid,
        snapped,
        left_on=["latitude", "longitude"],
        right_on=["grid_lat", "grid_lon"],
        how="left",
    )

    # Binary fire indicator: NaN → 0
    merged["fire"] = merged["fire"].fillna(0).astype(int)

    # Assign year
    merged["year"] = merged["year"].fillna(no_fire_year).astype(int)

    # Final clean dataset
    merged = merged[["latitude", "longitude", "fire", "year"]]

    merged.to_csv(output_file, index=False)

    print(
        f"✅ Saved {merged.shape[0]} grid points with fire + year info to {output_file}"
    )


def treat_sensor_errors_soil(csv_path, output_path):
    df = pd.read_csv(csv_path)

    # Remove TEXTURE_SOTER if present
    if "TEXTURE_SOTER" in df.columns:
        df.drop(columns=["TEXTURE_SOTER"], inplace=True)

    # Replace '-' with NaN
    df = df.replace("-", np.nan)

    # Numeric columns
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

    # Convert to numeric
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Detect negative-value rows
    mask_negative = df[numeric_cols].lt(0).any(axis=1)

    # Compute duplicates per coordinate pair
    coord_cols = ["latitude", "longitude"]
    counts = df.groupby(coord_cols)[coord_cols[0]].transform("count")

    # Rows to delete and fix
    mask_delete = mask_negative & (counts > 1)
    mask_fix = mask_negative & (counts <= 1)

    rows_to_delete_count = mask_delete.sum()
    rows_to_fix_count = mask_fix.sum()

    # Delete rows with duplicates
    df_clean = df.loc[~mask_delete].copy()

    # -------- FIX rows (replace only the negative values with NaN) --------
    # Only apply fixes to rows that originally had negative values
    idx_fix = df_clean.index.intersection(df.index[mask_fix])

    # Replace negatives with NaN in those rows
    df_clean.loc[idx_fix, numeric_cols] = df_clean.loc[idx_fix, numeric_cols].mask(
        df_clean.loc[idx_fix, numeric_cols] < 0
    )
    # ----------------------------------------------------------------------

    # Save cleaned version
    df_clean.to_csv(output_path, index=False)

    print("✔ Cleaning complete!")
    print(f"  Deleted rows : {rows_to_delete_count}")
    print(f"  Fixed rows   : {rows_to_fix_count}")


def impute_with_geo_zones(
    input_csv,
    num_cols=None,
    cat_cols=None,
    lat_col="latitude",
    lon_col="longitude",
    base_res=0.1,
    min_points=10,
    max_res=5.0,
    output_path="./dataCleaned.csv",
):
    df = pd.read_csv(input_csv)
    num_cols = num_cols or []
    cat_cols = cat_cols or []

    # --- 1. Missing percentage ---
    missing_percent = df.isnull().mean() * 100
    print("Missing values (percent) per column :")
    print(missing_percent[missing_percent > 0])

    def impute_value(row, col, is_num, max_res):
        lat, lon = row[lat_col], row[lon_col]
        resolution = base_res

        while resolution <= max_res:
            lat_min, lat_max = lat - resolution, lat + resolution
            lon_min, lon_max = lon - resolution, lon + resolution

            # Vectorized filtering for the zone
            zone_df = df[
                (df[lat_col] >= lat_min)
                & (df[lat_col] <= lat_max)
                & (df[lon_col] >= lon_min)
                & (df[lon_col] <= lon_max)
            ]

            # Check the number of non-missing values in the target column
            valid_count = zone_df[col].count()

            if valid_count >= min_points:
                break
            resolution *= 1.5

        if valid_count == 0:  # Fallback to global
            print("Fallback to global , valid_count=0")
            return df[col].median() if is_num else df[col].mode(dropna=True).iat[0]

        if is_num:
            return zone_df[col].median()
        else:
            mode_val = zone_df[col].mode(dropna=True)
            return (
                mode_val.iat[0]
                if not mode_val.empty
                else df[col].mode(dropna=True).iat[0]
            )

    # --- 2. Process each column ---
    for col in df.columns:
        if col in [lat_col, lon_col] or missing_percent.get(col, 0) == 0:
            continue

        print(f"\n=== Imputing column: {col} ===")

        is_num = col in num_cols or (
            col not in cat_cols and pd.api.types.is_numeric_dtype(df[col])
        )

        # Select only the rows where the column is missing
        missing_rows = df[df[col].isnull()].copy()

        # Apply the imputation function to the missing rows
        imputed_values = missing_rows.apply(
            lambda row: impute_value(row, col, is_num, max_res), axis=1
        )

        # Assign the calculated values back to the original DataFrame
        df.loc[imputed_values.index, col] = imputed_values

        print(f"{col}: imputation done using geo-zones.")

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"💾 Saved imputation to {output_path}")


def duplicate_analysis(
    csv_path: str,
    ignore_columns=None,  # list of columns NOT to consider when checking duplicates
    delete_duplicates=False,  # whether to drop duplicates
    output_clean_path=None,  # where to save cleaned CSV
):
    """
    Analyze duplicate rows in a CSV with optional column exclusion and deletion.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    ignore_columns : list[str], optional
        Columns to ignore when detecting duplicates.
    delete_duplicates : bool
        If True, removes duplicates and saves cleaned CSV.
    output_clean_path : str
        Path to save cleaned CSV when delete_duplicates=True.

    Returns
    -------
    dict
        {
            "total_rows": int,
            "duplicate_rows": int,
            "duplicate_percentage": float,
            "duplicated_sample": pd.DataFrame
        }
    """

    df = pd.read_csv(csv_path)

    # Handle ignore_columns
    if ignore_columns:
        cols_to_check = [col for col in df.columns if col not in ignore_columns]
    else:
        cols_to_check = df.columns.tolist()

    # Detect duplicates
    duplicated_mask = df.duplicated(subset=cols_to_check, keep=False)
    duplicate_rows = df[duplicated_mask]

    stats = {
        "total_rows": len(df),
        "duplicate_rows": len(duplicate_rows),
        "duplicate_percentage": round((len(duplicate_rows) / len(df)) * 100, 3),
        "duplicated_sample": duplicate_rows.head(10),
    }

    # Optionally delete duplicates
    if delete_duplicates:
        if output_clean_path is None:
            raise ValueError(
                "output_clean_path must be provided when delete_duplicates=True"
            )

        df_clean = df.drop_duplicates(subset=cols_to_check, keep="first")
        df_clean.to_csv(output_clean_path, index=False)

    return stats
