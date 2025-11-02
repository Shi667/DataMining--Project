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
    from a fire dataset CSV and save the result to a new CSV.

    Parameters
    ----------
    raster_path : str
        Path to the .tif raster file.
    fire_csv_path : str
        Path to the fire dataset CSV file.
    output_csv : str
        Path where to save the output CSV file.
    lat_col : str
        Column name for latitude in the fire CSV.
    lon_col : str
        Column name for longitude in the fire CSV.
    value_name : str
        Name for the new raster value column (e.g., "elevation", "tmax").

    Returns
    -------
    pd.DataFrame
        DataFrame with added raster value column.
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

    # --- Add column and save ---
    df[value_name] = values
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved extracted values to {output_csv}")

    return df


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

    df = pd.read_csv(csv_path)
    points = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]

    with rasterio.open(raster_path) as src:
        # Sample the raster at the coordinates and get the first band value (index 0)
        values = [x[0] for x in src.sample([(p.x, p.y) for p in points])]

    df["HWSD2_SMU_ID"] = values

    # Select only the required columns: latitude, longitude, and HWSD2_SMU_ID
    output_ids = df[[lat_col, lon_col, "HWSD2_SMU_ID"]]

    if output_soil_ids:
        output_ids.to_csv(output_soil_ids, index=False)

    df_fires = output_ids
    df_soil = pd.read_csv(soil_attributes_csv)

    merged = df_fires.merge(df_soil, on="HWSD2_SMU_ID", how="left")

    if output_soil_feature:
        merged.to_csv(output_soil_feature, index=False)

    return (output_ids, merged)


def extract_features_monthly_clim(
    fire_csv,
    raster_dict,
    lat_col="latitude",
    lon_col="longitude",
    date_col="acq_date",
    output_path=None,
    value_name: str = "clim",
):
    """
    Extracts monthly Tmax values for each fire point based on its acquisition date.

    Parameters
    ----------
    fire_csv : str
        Path to the fire dataset CSV (must contain latitude, longitude, acq_date).
    raster_dict : dict
        Dictionary mapping 'MM' (month string) -> raster path (.tif file).
    lat_col, lon_col, date_col : str
        Column names in fire CSV.
    output_path : str, optional
        Path to save the resulting CSV.

    Returns
    -------
    pd.DataFrame
        Fire dataset with an extra 'value_name' column.
    """
    df = pd.read_csv(fire_csv)

    # --- Extract month from acq_date ---
    df["month"] = pd.to_datetime(df[date_col]).dt.strftime("%m")

    # --- Initialize result column ---
    df[value_name] = None

    # --- Process month by month ---
    for month, raster_path in raster_dict.items():
        mask = df["month"] == month
        if not mask.any():
            continue  # skip months with no fires

        sub_df = df[mask]
        coords = list(zip(sub_df[lon_col], sub_df[lat_col]))

        with rasterio.open(raster_path) as src:
            values = [
                x[0] if x[0] is not None else float("nan") for x in src.sample(coords)
            ]
            df.loc[mask, value_name] = values

        print(f"✅ Extracted {value_name} for month {month} ({mask.sum()} points)")

    # --- Clean up ---
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
