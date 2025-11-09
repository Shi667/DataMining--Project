import geopandas as gpd
import rasterio
import pandas as pd
import glob
import os
from tqdm import tqdm


def extract_features_elevation(
    raster_path: str,
    fire_csv_path: str,
    output_csv: str,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    value_name: str = "elevation",
) -> pd.DataFrame:
    """
    Extract raster values (e.g., elevation) at given latitude/longitude points
    from a fire dataset CSV and save the result to a new CSV containing only
    latitude, longitude, and the extracted value.
    """

    # --- Load fire dataset ---
    df = pd.read_csv(fire_csv_path)
    print(f"Loaded {len(df)} points from {fire_csv_path}")

    # --- Open raster and extract values ---
    with rasterio.open(raster_path) as src:
        coords = [(x, y) for x, y in zip(df[lon_col], df[lat_col])]
        values = [
            val[0] if val[0] is not None else None
            for val in tqdm(
                src.sample(coords), total=len(coords), desc=f"Extracting {value_name}"
            )
        ]

    # --- Create new DataFrame with only lat, lon, and extracted values ---
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

    return joined


def extract_features_soil(
    csv_path,
    raster_path,
    soil_attributes_csv,
    lat_col="latitude",
    lon_col="longitude",
    output_soil_ids: str = None,
    output_soil_feature: str = None,
):
    import pandas as pd
    import rasterio
    from shapely.geometry import Point
    import numpy as np

    # Load input points
    df = pd.read_csv(csv_path)
    points = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]

    # Sample raster for soil IDs
    with rasterio.open(raster_path) as src:
        values = [x[0] for x in src.sample([(p.x, p.y) for p in points])]

    df["HWSD2_SMU_ID"] = values

    # Output only latitude, longitude, and HWSD2_SMU_ID
    output_ids = df[[lat_col, lon_col, "HWSD2_SMU_ID"]]
    if output_soil_ids:
        output_ids.to_csv(output_soil_ids, index=False)

    # Load soil attributes
    df_soil = pd.read_csv(soil_attributes_csv)

    # Merge on HWSD2_SMU_ID
    merged = output_ids.merge(df_soil, on="HWSD2_SMU_ID", how="left")

    # Remove rows where all soil attributes after 3rd column are repeated
    def is_repeated(row):
        values = row[3:]  # skip latitude, longitude, HWSD2_SMU_ID
        # keep only non-empty/non-NaN values
        values = [v for v in values if pd.notna(v) and v != ""]
        return len(values) > 0 and all(v == values[0] for v in values)

    merged.loc[merged.apply(is_repeated, axis=1), :] = np.nan

    # Drop the HWSD2_SMU_ID column from the final merged CSV
    if "HWSD2_SMU_ID" in merged.columns:
        merged = merged.drop(columns=["HWSD2_SMU_ID"])

    # Save final merged soil features
    if output_soil_feature:
        merged.to_csv(output_soil_feature, index=False)

    return output_ids, merged


import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm


def extract_features_monthly_clim(
    fire_csv,
    raster_dict,
    lat_col="latitude",
    lon_col="longitude",
    date_col="acq_date",
    output_path=None,
    value_name: str = "clim",
    agg_mode: str = "median",  # or "mean"
):
    """
    Extracts monthly climatology values for each point based on acquisition date.
    If acq_date is missing, averages (or takes median) of all raster values at that location.
    """
    df = pd.read_csv(fire_csv)
    df["month"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%m")
    df[value_name] = np.nan

    # ✅ For points with known month
    for month, raster_path in raster_dict.items():
        mask = df["month"] == month
        if not mask.any():
            continue

        coords = list(zip(df.loc[mask, lon_col], df.loc[mask, lat_col]))
        with rasterio.open(raster_path) as src:
            nodata = src.nodata
            values = []
            for val in tqdm(
                src.sample(coords), total=len(coords), desc=f"Month {month}"
            ):
                v = val[0]
                if v is None or (nodata is not None and v == nodata):
                    v = np.nan
                values.append(v)
        df.loc[mask, value_name] = values
        print(f"✅ Extracted {value_name} for month {month} ({mask.sum()} points)")

    # ✅ For missing months → average or median over all rasters (pixel-wise per point)
    missing_mask = df["month"].isna()
    if missing_mask.any():
        print(
            f"ℹ️ Handling {missing_mask.sum()} points with no month — using {agg_mode} of all rasters."
        )
        sub_df = df.loc[missing_mask, [lon_col, lat_col]]
        coords = list(zip(sub_df[lon_col], sub_df[lat_col]))

        all_values = []

        for path in raster_dict.values():
            with rasterio.open(path) as src:
                nodata = src.nodata
                vals = []
                for val in src.sample(coords):
                    v = val[0]
                    if v is None or (nodata is not None and v == nodata):
                        v = np.nan
                    vals.append(v)
                all_values.append(vals)

        stacked = np.stack(all_values, axis=1)
        if agg_mode == "median":
            agg_vals = np.nanmedian(stacked, axis=1)
        else:
            agg_vals = np.nanmean(stacked, axis=1)

        df.loc[missing_mask, value_name] = agg_vals

    # ✅ Keep only relevant columns
    df = df[[lat_col, lon_col, date_col, value_name]]

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"💾 Saved to {output_path}")

    return df


def organize_monthly_climat_files(data_folder_path):
    """
    Finds .tif files with a specific naming convention, extracts the month,
    and stores the file path in a dictionary with the month number as the key.

    Args:
        data_folder_path (str): The path pattern to your data folder,
                                e.g., "../data/climate_dataset/5min/max/*.tif"

    Returns:
        dict: A dictionary where keys are month numbers (str) and values are
              the full file paths (str).
    """
    tmax_paths = glob.glob(data_folder_path)

    monthly_files = {}

    for file_path in tmax_paths:
        # Extract just the filename from the full path
        filename = os.path.basename(file_path)
        base_name = filename.replace(".tif", "")
        date_part = base_name.split("_")[-1]
        month = date_part.split("-")[1]
        monthly_files[month] = file_path

    return monthly_files
