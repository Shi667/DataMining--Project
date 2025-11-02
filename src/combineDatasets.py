import geopandas as gpd
import rasterio
import pandas as pd
from shapely.geometry import Point


def extract_features_at_points(
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


def extract_raster_values(
    csv_path, raster_path, lat_col="latitude", lon_col="longitude", output_path=None
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
    output_df = df[[lat_col, lon_col, "HWSD2_SMU_ID"]]

    if output_path:
        output_df.to_csv(output_path, index=False)

    return output_df


def join_soil_attributes(fire_soil_csv, soil_attributes_csv, output_path=None):
    df_fires = pd.read_csv(fire_soil_csv)
    df_soil = pd.read_csv(soil_attributes_csv)

    merged = df_fires.merge(df_soil, on="HWSD2_SMU_ID", how="left")

    if output_path:
        merged.to_csv(output_path, index=False)

    return merged


"""

fires_with_landcover = extract_features_at_points(
    csv_path="../data/fire_dataset/viirs-jpss1_2024_Algeria.csv",
    shapefile_path="../data/land_dataset/algeria/dza_gc_adg.shp",
    lat_col="latitude",
    lon_col="longitude",
    keep_cols=["GRIDCODE"],  # can be ["GRIDCODE", "CLASS", "AREA", ...]
    output_path="../data/features/landcover_at_fires.csv",
)"""

fires_with_soil_id = extract_raster_values(
    csv_path="../data/fire_dataset/viirs-jpss1_2024_Algeria.csv",
    raster_path="../data/soil_dataset/HWSD2_RASTER/HWSD2.bil",
    output_path="../data/features/fire_soil_ids.csv",
)


fires_with_soil = join_soil_attributes(
    fire_soil_csv="../data/features/fire_soil_ids.csv",
    soil_attributes_csv="./D1_soil_features_alg_tun.csv",
    output_path="../data/features/soil_features_at_fires.csv",
)




