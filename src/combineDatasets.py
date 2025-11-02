import geopandas as gpd
import pandas as pd


import geopandas as gpd
import pandas as pd


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


fires_with_landcover = extract_features_at_points(
    csv_path="../data/fire_dataset/viirs-jpss1_2024_Algeria.csv",
    shapefile_path="../data/land_dataset/algeria/dza_gc_adg.shp",
    lat_col="latitude",
    lon_col="longitude",
    keep_cols=["GRIDCODE"],  # can be ["GRIDCODE", "CLASS", "AREA", ...]
    output_path="../data/features/landcover_at_fires.csv",
)

