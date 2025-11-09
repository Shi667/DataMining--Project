import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
import os
import glob
from rasterio.transform import xy


def extract_raster_points_in_shape(
    raster_path: str,
    shapefile_path: str,
    output_csv: str,
    resolution: float,
    value_name: str = "value",
) -> pd.DataFrame:
    """
    Extract raster values (e.g., elevation) within a shapefile region at a given resolution.

    Parameters
    ----------
    raster_path : str
        Path to the input raster (.tif) file.
    shapefile_path : str
        Path to the shapefile (.shp) defining the region of interest.
    output_csv : str
        Path where the output CSV will be saved.
    resolution : float
        Step size for sampling (e.g., 0.01 for ~1km grid).
    value_name : str
        Column name for the raster value (e.g., 'elevation').

    Returns
    -------
    pd.DataFrame
        DataFrame with [latitude, longitude, value_name] inside the shapefile.
    """

    # --- Load shapefile ---
    shape_gdf = gpd.read_file(shapefile_path)
    print(f"Loaded shapefile with {len(shape_gdf)} geometries")

    # --- Open raster ---
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        bounds = src.bounds

        # Reproject shapefile if needed
        if shape_gdf.crs != raster_crs:
            print("Reprojecting shapefile to match raster CRS...")
            shape_gdf = shape_gdf.to_crs(raster_crs)

        # Create coordinate grid within raster bounds
        xs = np.arange(bounds.left, bounds.right, resolution)
        ys = np.arange(bounds.bottom, bounds.top, resolution)
        coords = [(x, y) for y in ys for x in xs]

        # Convert to GeoDataFrame for spatial filtering
        points_gdf = gpd.GeoDataFrame(
            geometry=[Point(x, y) for x, y in coords], crs=raster_crs
        )

        # Keep only points within the shapefile polygons
        points_in_shape = gpd.sjoin(
            points_gdf, shape_gdf, how="inner", predicate="intersects"
        )
        coords_filtered = [(pt.x, pt.y) for pt in points_in_shape.geometry]

        print(f"Filtered to {len(coords_filtered)} points inside shapefile")

        # --- Sample raster values ---
        values = [
            val[0] if val is not None else np.nan
            for val in tqdm(
                src.sample(coords_filtered),
                total=len(coords_filtered),
                desc=f"Extracting {value_name}",
            )
        ]

    # --- Build output DataFrame ---
    result_df = pd.DataFrame(coords_filtered, columns=["longitude", "latitude"])
    result_df[value_name] = values

    # --- Save to CSV ---
    result_df.to_csv(output_csv, index=False)
    print(f"✅ Saved {len(result_df)} points to {output_csv}")

    return result_df


def extract_monthly_clim_in_shape(
    shapefile_path: str,
    raster_dict: dict,
    output_folder: str,
    resolution: float,
    value_name: str = "clim",
) -> dict:
    """
    Extracts monthly climate raster values (e.g., Tmax) within a shapefile region
    for all months in raster_dict, at a given spatial resolution.

    Parameters
    ----------
    shapefile_path : str
        Path to shapefile (.shp) defining the region of interest.
    raster_dict : dict
        Dictionary mapping month strings ('01', '02', ..., '12') → raster file paths (.tif).
    output_folder : str
        Folder where the monthly CSV files will be saved.
    resolution : float
        Step size for sampling (e.g., 0.05 for ~5km grid in degrees).
    value_name : str
        Name for the extracted raster value column (e.g., 'tmax').

    Returns
    -------
    dict
        Mapping {month: DataFrame} with extracted points for each month.
    """
    os.makedirs(output_folder, exist_ok=True)

    # --- Load shapefile ---
    shape_gdf = gpd.read_file(shapefile_path)
    print(f"Loaded shapefile with {len(shape_gdf)} geometries")

    month_results = {}

    # --- Process each month ---
    for month, raster_path in raster_dict.items():
        print(f"\n📆 Processing month {month} from {os.path.basename(raster_path)}")

        with rasterio.open(raster_path) as src:
            raster_crs = src.crs
            bounds = src.bounds

            # Reproject shapefile if needed
            if shape_gdf.crs != raster_crs:
                shape_gdf = shape_gdf.to_crs(raster_crs)

            # Create grid of points within raster bounds
            xs = np.arange(bounds.left, bounds.right, resolution)
            ys = np.arange(bounds.bottom, bounds.top, resolution)
            coords = [(x, y) for y in ys for x in xs]

            # Convert to GeoDataFrame for spatial filtering
            points_gdf = gpd.GeoDataFrame(
                geometry=[Point(x, y) for x, y in coords], crs=raster_crs
            )

            # Keep only points inside the shapefile
            points_in_shape = gpd.sjoin(
                points_gdf, shape_gdf, how="inner", predicate="intersects"
            )
            coords_filtered = [(pt.x, pt.y) for pt in points_in_shape.geometry]
            print(f"  → {len(coords_filtered)} points inside region")

            # Extract raster values
            values = [
                v[0] if v is not None else np.nan
                for v in tqdm(
                    src.sample(coords_filtered),
                    total=len(coords_filtered),
                    desc=f"Extracting {value_name}",
                )
            ]

        # Build DataFrame
        result_df = pd.DataFrame(coords_filtered, columns=["longitude", "latitude"])
        result_df[value_name] = values
        result_df["month"] = month

        # Save to CSV
        out_csv = os.path.join(output_folder, f"{value_name}_month_{month}.csv")
        result_df.to_csv(out_csv, index=False)
        print(f"✅ Saved {len(result_df)} points to {out_csv}")

        month_results[month] = result_df

    print("\n🎉 All months processed successfully.")
    return month_results


def organize_monthly_climat_files(data_folder_path: str) -> dict:
    """
    Organize monthly raster files into a dictionary {month: path}.
    Example expected filename: 'tmax_2020-03.tif' → month = '03'
    """
    tmax_paths = glob.glob(data_folder_path)
    monthly_files = {}

    for file_path in tmax_paths:
        filename = os.path.basename(file_path)
        base_name = filename.replace(".tif", "")
        date_part = base_name.split("_")[-1]
        month = date_part.split("-")[1]
        monthly_files[month] = file_path

    return monthly_files







"""
# Folder with monthly .tif files
data_folder_tmin = "../../../data/climate_dataset/5min/min/*.tif"
data_folder_tmax = "../../../data/climate_dataset/5min/max/*.tif"
data_folder_prec = "../../../data/climate_dataset/5min/prec/*.tif"

raster_path_elevation = (
    "../../../data/elevation_dataset/simplified/elevation_clipped.tif"
)
output_csv_elevation = "../../../data/features/elevation_grid.csv"
output_csv_tmin = "../../../data/features/tmin_grid.csv"
output_csv_tmax = "../../../data/features/tmax_grid.csv"
output_csv_prec = "../../../data/features/tprec_grid.csv"

shapefile_path = "../../../data/shapefiles/combined/alg_tun.shp"
resolution = 0.03  # degrees (~3 km)
df_elev = extract_raster_points_in_shape(
    raster_path_elevation,
    shapefile_path,
    output_csv_elevation,
    resolution,
    value_name="elevation",
)


tmin_raster_dict = organize_monthly_climat_files(data_folder_tmin)

# Extract values
df_tmin = extract_monthly_clim_in_shape(
    shapefile_path,
    raster_dict=tmin_raster_dict,
    output_folder=output_csv_tmin,
    resolution=resolution,
    value_name="tmin",
)


tmax_raster_dict = organize_monthly_climat_files(data_folder_tmax)

# Extract values
df_tmax = extract_monthly_clim_in_shape(
    shapefile_path,
    raster_dict=tmax_raster_dict,
    output_folder=output_csv_tmax,
    resolution=resolution,
    value_name="tmax",
)


prec_raster_dict = organize_monthly_climat_files(data_folder_prec)

# Extract values
df_prec = extract_monthly_clim_in_shape(
    shapefile_path,
    raster_dict=prec_raster_dict,
    output_folder=output_csv_prec,
    resolution=resolution,
    value_name="prec",
)
"""
