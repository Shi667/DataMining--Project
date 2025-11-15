import geopandas as gpd
import rasterio
import pandas as pd
import glob
import os
from tqdm import tqdm
import numpy as np


def extract_features_elevation(
    raster_path: str,
    fire_csv_path: str,
    output_csv: str,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    value_name: str = "elevation",
) -> pd.DataFrame:
    df = pd.read_csv(fire_csv_path)
    print(f"Loaded {len(df)} points from {fire_csv_path}")

    # --- Open raster and extract values ---
    with rasterio.open(raster_path) as src:
        # Create coords generator directly from DataFrame columns
        coords = zip(df[lon_col], df[lat_col])

        # Use rasterio.sample generator, converting to a list of values
        # The .sample() function is already optimized and returns (value,) tuples

        # Optimization: Use list comprehension to flatten and handle None/NoData in one step
        values = [
            val[0] if val[0] is not None and val[0] != src.nodata else None
            for val in tqdm(
                src.sample(coords), total=len(df), desc=f"Extracting {value_name}"
            )
        ]

    # --- Create new DataFrame with only lat, lon, and extracted values ---
    # Resulting column order is determined by the dictionary insertion order (Python 3.7+ guarantee)
    result_df = pd.DataFrame(
        {lat_col: df[lat_col], lon_col: df[lon_col], value_name: values}
    )

    # --- Save result ---
    result_df.to_csv(output_csv, index=False)
    print(f"✅ Saved extracted {value_name} to {output_csv}")

    return result_df


def extract_features_landcover(
    csv_path: str,
    shapefile_path: str,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    keep_cols: list = None,
    output_path: str = None,
    crs: str = "EPSG:4326",
):
    """
    Extracts attributes from a shapefile at given point locations (from a CSV).

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing point coordinates.
    shapefile_path : str
        Path to the shapefile containing polygons or other geometries.
    lat_col : str, optional
        Name of the latitude column in the CSV (default: 'latitude').
    lon_col : str, optional
        Name of the longitude column in the CSV (default: 'longitude').
    keep_cols : list, optional
        Columns from the shapefile to include in the output (default: all non-geometry columns).
    output_path : str, optional
        Path to save the resulting CSV (default: no save, only return).
    crs : str, optional
        CRS to assume for both datasets (default: 'EPSG:4326').

    Returns
    -------
    pd.DataFrame
        A DataFrame containing original CSV columns + extracted attributes.
    """

    # --- Load points ---
    df_points = pd.read_csv(csv_path)
    gdf_points = gpd.GeoDataFrame(
        df_points,
        geometry=gpd.points_from_xy(df_points[lon_col], df_points[lat_col]),
        crs=crs,
    )

    # --- Load shapefile ---
    gdf_shapes = gpd.read_file(shapefile_path).to_crs(crs)

    # --- Select columns to keep ---
    if keep_cols is None:
        keep_cols = [c for c in gdf_shapes.columns if c != "geometry"]
    gdf_shapes = gdf_shapes[keep_cols + ["geometry"]]

    # --- Spatial join ---
    joined = gpd.sjoin(gdf_points, gdf_shapes, how="left")

    # --- Clean and keep only desired columns ---
    joined = joined[[lat_col, lon_col] + keep_cols]

    # --- Save if needed ---
    if output_path:
        joined.to_csv(output_path, index=False)



def extract_features_soil(
    csv_path,
    raster_path,
    soil_attributes_csv,
    lat_col="latitude",
    lon_col="longitude",
    output_soil_ids: str = None,
    output_soil_feature: str = None,
):
    # Load input points
    df = pd.read_csv(csv_path)

    # --- Optimization: Sample raster without creating Shapely Points ---
    coords = zip(df[lon_col], df[lat_col])  # Generator of (lon, lat) tuples

    with rasterio.open(raster_path) as src:
        # Sample raster directly, then flatten the result
        values = [x[0] for x in src.sample(coords)]

    df["HWSD2_SMU_ID"] = values

    # Output only latitude, longitude, and HWSD2_SMU_ID
    output_ids = df[[lat_col, lon_col, "HWSD2_SMU_ID"]]
    if output_soil_ids:
        output_ids.to_csv(output_soil_ids, index=False)

    # Load soil attributes
    df_soil = pd.read_csv(soil_attributes_csv)

    # Merge on HWSD2_SMU_ID
    merged = output_ids.merge(df_soil, on="HWSD2_SMU_ID", how="left")

    # --- Optimization: Vectorized check for repeated values ---
    # Identify soil feature columns (all columns after the first three)
    soil_feature_cols = merged.columns[3:]

    if not soil_feature_cols.empty:
        # Select soil feature data and convert to NumPy array
        soil_data = merged[soil_feature_cols].values

        # Fill NaN for comparison purposes (e.g., fill with a unique large number)
        soil_data_filled = np.nan_to_num(soil_data, nan=-9999999)

        # Check if all values in each row are equal to the first value of that row
        # This is the vectorized equivalent of the original is_repeated logic
        is_repeated_mask = (
            soil_data_filled == soil_data_filled[:, 0][:, np.newaxis]
        ).all(axis=1)

        # Check if there are any non-NaN values at all in the soil feature columns
        has_non_nan = ~merged[soil_feature_cols].isnull().all(axis=1)

        # The rows to be flagged as NaN are those that are repeated AND have at least one non-NaN value.
        mask_to_nan = is_repeated_mask & has_non_nan

        # Set all values in the merged row (starting from the 4th column) to NaN
        merged.loc[mask_to_nan, soil_feature_cols] = np.nan

    # Drop the HWSD2_SMU_ID column from the final merged CSV
    if "HWSD2_SMU_ID" in merged.columns:
        merged = merged.drop(columns=["HWSD2_SMU_ID"])

    # Save final merged soil features
    if output_soil_feature:
        merged.to_csv(output_soil_feature, index=False)



def extract_features_monthly_clim(
    point_csv,
    raster_dict,
    lat_col="latitude",
    lon_col="longitude",
    output_path=None,
    col_name="clim",
    agg_mode: str = "mean",
):
    df = pd.read_csv(point_csv)
    coords = list(zip(df[lon_col], df[lat_col]))

    # --- Sample ALL monthly rasters for each point ---
    month_values = {}

    for month, raster_path in raster_dict.items():
        with rasterio.open(raster_path) as src:
            nodata = src.nodata

            # Optimization: Use list comprehension to sample, flatten, and handle NoData
            # This is more compact and potentially faster than the inner loop structure
            vals = [
                v[0] if v[0] is not None and v[0] != nodata else np.nan
                for v in tqdm(
                    src.sample(coords), total=len(coords), desc=f"Month {month}"
                )
            ]

        month_values[month] = vals
        df[f"m{month}"] = (
            vals  # Add column inside the loop for memory efficiency (no need for month_values dict)
        )

    print("✅ Finished sampling all monthly rasters.")

    # --- Define seasons ---
    winter_cols = [f"m{m}" for m in ["12", "01", "02"]]
    spring_cols = [f"m{m}" for m in ["03", "04", "05"]]
    summer_cols = [f"m{m}" for m in ["06", "07", "08"]]
    autumn_cols = [f"m{m}" for m in ["09", "10", "11"]]

    # --- Compute seasonal aggregation (Vectorized) ---
    if agg_mode == "median":
        agg_func = "median"
    else:
        agg_func = "mean"

    df[f"winter_{col_name}"] = df[winter_cols].agg(agg_func, axis=1)
    df[f"spring_{col_name}"] = df[spring_cols].agg(agg_func, axis=1)
    df[f"summer_{col_name}"] = df[summer_cols].agg(agg_func, axis=1)
    df[f"autumn_{col_name}"] = df[autumn_cols].agg(agg_func, axis=1)

    # --- Drop monthly columns and select final columns ---
    final_cols = [
        lat_col,
        lon_col,
        f"winter_{col_name}",
        f"spring_{col_name}",
        f"summer_{col_name}",
        f"autumn_{col_name}",
    ]
    df = df[final_cols]

    # --- Save result ---
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"💾 Saved seasonal climatology to {output_path}")


def organize_monthly_climat_files(data_folder_path):
    tmax_paths = glob.glob(data_folder_path)
    monthly_files = {}

    for file_path in tmax_paths:
        filename = os.path.basename(file_path)

        # Optimization: Use rsplit and split for cleaner parsing
        try:
            # Assumes format: prefix_month-year.tif or prefix_date_month-year.tif
            base_name = filename.rsplit(".", 1)[0]
            date_part = base_name.split("_")[-1]
            month = date_part.split("-")[
                0
            ]  # Assuming month is the first part of date_part (e.g., 01-2000)

            # Re-read original logic: if date_part is 'YYYY-MM', then month is 'MM'
            if len(date_part.split("-")) == 2:
                month = date_part.split("-")[1]
            elif len(date_part.split("-")) == 1 and len(date_part) == 2:
                # Handle case where the month is directly the last part
                month = date_part
            else:
                # Fallback to original logic if naming is complex: base_name ends with YYYY-MM
                month = base_name.split("-")[-1]

            # Ensure month is two characters (e.g., '1' becomes '01')
            month = month.zfill(2)

            if month.isdigit() and 1 <= int(month) <= 12:
                monthly_files[month] = file_path

        except IndexError:
            # Skip files that don't match the expected naming convention
            continue

    return monthly_files
