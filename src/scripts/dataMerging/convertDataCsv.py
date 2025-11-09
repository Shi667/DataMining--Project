import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
import os
import glob
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.mask import mask


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


def extract_landcover(
    landcover_path: str,
    output_csv: str,
    resolution: float,
    keep_cols: list = None,
):
    """
    Fast version: rasterizes land cover shapefile instead of heavy spatial join.
    Produces (longitude, latitude, attributes) grid limited to land polygons.
    """
    print("📂 Loading land cover shapefile...")
    landcover = gpd.read_file(landcover_path).to_crs("EPSG:4326")

    # --- Keep selected columns ---
    if keep_cols is None:
        keep_cols = [c for c in landcover.columns if c != "geometry"]
    main_attr = keep_cols[0]  # assume first column is main landcover class

    # --- Prepare rasterization parameters ---
    bounds = landcover.total_bounds  # [minx, miny, maxx, maxy]
    width = int(np.ceil((bounds[2] - bounds[0]) / resolution))
    height = int(np.ceil((bounds[3] - bounds[1]) / resolution))
    transform = from_origin(bounds[0], bounds[3], resolution, resolution)

    print(f"🗺️ Rasterizing {len(landcover)} polygons ({width}×{height} grid)...")

    # --- Rasterize main landcover attribute ---
    shapes = zip(landcover.geometry, landcover[main_attr])
    raster = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=np.nan,
        dtype="float32",
    )

    # --- Convert raster to points ---
    xs = np.arange(width)
    ys = np.arange(height)
    longs, lats, values = [], [], []

    for y in tqdm(ys, desc="Converting raster to points"):
        for x in xs:
            val = raster[y, x]
            if not np.isnan(val):  # keep only valid land pixels
                lon, lat = transform * (x + 0.5, y + 0.5)
                longs.append(lon)
                lats.append(lat)
                values.append(val)

    # --- Create DataFrame ---
    df = pd.DataFrame({"longitude": longs, "latitude": lats, main_attr: values})

    # --- Save ---
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved {len(df):,} landcover points to {output_csv}")

    return df


def extract_soil_in_shape(
    shapefile_path: str,
    raster_path: str,
    soil_attributes_csv: str,
    output_csv: str,
    resolution: float,
    value_name: str = "HWSD2_SMU_ID",
):
    """
    Fast soil extraction limited to ROI.
    Reads only inside the shapefile region, samples at given resolution.
    """
    print("📂 Loading region and raster...")
    roi = gpd.read_file(shapefile_path)
    soil_attrs = pd.read_csv(soil_attributes_csv)

    with rasterio.open(raster_path) as src:
        # Reproject ROI if needed
        if roi.crs != src.crs:
            roi = roi.to_crs(src.crs)

        # Mask raster to ROI
        print("🪣 Clipping raster to region...")
        out_image, out_transform = mask(src, roi.geometry, crop=True)
        out_image = out_image[0]

        # Generate coordinate grid inside ROI
        bounds = roi.total_bounds
        xs = np.arange(bounds[0], bounds[2], resolution)
        ys = np.arange(bounds[1], bounds[3], resolution)

        coords = [(x, y) for y in ys for x in xs]
        values, valid_coords = [], []

        for x, y in tqdm(coords, desc="Sampling soil values"):
            try:
                row, col = src.index(x, y)
                val = out_image[row, col]
                if not np.isnan(val) and val != src.nodata:
                    valid_coords.append((x, y))
                    values.append(val)
            except Exception:
                continue

    result_df = pd.DataFrame(valid_coords, columns=["longitude", "latitude"])
    result_df[value_name] = values

    # Merge with soil attributes
    merged = result_df.merge(soil_attrs, on=value_name, how="left")

    merged.to_csv(output_csv, index=False)
    print(f"✅ Saved {len(merged):,} soil points to {output_csv}")

    return merged


shapefile_path = "../../../data/shapefiles/combined/alg_tun.shp"

# --- Land cover ---
extract_landcover(
    landcover_path="../../../data/land_dataset/combined/alg_tun_landcvr.shp",
    output_csv="../../../data/features/landcover_grid.csv",
    resolution=0.03,
    keep_cols=["GRIDCODE"],
)

# --- Soil ---
extract_soil_in_shape(
    shapefile_path=shapefile_path,
    raster_path="../../../data/soil_dataset/original/HWSD2_RASTER/HWSD2.bil",
    soil_attributes_csv="../../../data/soil_dataset/simplified/D1_soil_features_alg_tun.csv",
    output_csv="../../../data/features/soil_grid.csv",
    resolution=0.03,
    value_name="HWSD2_SMU_ID",
)

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
