import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


def generate_grid_in_shape(
    shapefile_path: str,
    resolution: float,
    output_csv: str,
    min_latitude: float = None,
    max_latitude: float = None,
) -> pd.DataFrame:
    """
    Generate a regular lon/lat grid over the bounding box of the input shapefile,
    then keep only points inside the shapefile. Optionally filter by latitude
    BEFORE creating the GeoDataFrame / running the spatial join.

    Parameters
    ----------
    shapefile_path : str
        Path to the polygon shapefile (will be reprojected to EPSG:4326).
    resolution : float
        Grid cell size in degrees (e.g. 0.01).
    output_csv : str
        Path where resulting points CSV will be saved.
    min_latitude : float, optional
        Minimum latitude (inclusive). If set, only keep points with latitude >= min_latitude.
    max_latitude : float, optional
        Maximum latitude (inclusive). If set, only keep points with latitude <= max_latitude.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['longitude', 'latitude'] for points inside the shapefile
        and satisfying the latitude constraints.
    """
    print("📂 Loading shapefile and reprojecting to EPSG:4326...")
    gdf = gpd.read_file(shapefile_path).to_crs("EPSG:4326")

    # --- Compute bounds ---
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    print(f"🗺️ Bounding box (lon/lat): {bounds}")

    # --- Create full grid using vectorized ranges ---
    xs = np.arange(bounds[0], bounds[2] + 1e-12, resolution)  # include end
    ys = np.arange(bounds[1], bounds[3] + 1e-12, resolution)
    print(
        f"📏 Grid candidate size: {len(xs)} × {len(ys)} = {len(xs) * len(ys):,} points"
    )

    lon_grid, lat_grid = np.meshgrid(xs, ys)
    points_df = pd.DataFrame(
        {"longitude": lon_grid.ravel(), "latitude": lat_grid.ravel()}
    )

    # --- Filter by latitude band BEFORE building the GeoDataFrame / sjoin ---
    if min_latitude is not None:
        before_count = len(points_df)
        points_df = points_df[points_df["latitude"] >= min_latitude]
        print(
            f"⬆️ Applied min_latitude={min_latitude}: {before_count:,} -> {len(points_df):,}"
        )

    if max_latitude is not None:
        before_count = len(points_df)
        points_df = points_df[points_df["latitude"] <= max_latitude]
        print(
            f"⬇️ Applied max_latitude={max_latitude}: {before_count:,} -> {len(points_df):,}"
        )

    if points_df.empty:
        print(
            "⚠️ No points remain after latitude filtering. Adjust bounds/resolution/min_latitude."
        )
        return points_df

    # --- Convert to GeoDataFrame (faster creation using points_from_xy) ---
    points_gdf = gpd.GeoDataFrame(
        points_df,
        geometry=gpd.points_from_xy(points_df.longitude, points_df.latitude),
        crs="EPSG:4326",
    )

    # --- Keep only points inside the shapefile using spatial join (intersects) ---
    print("🔍 Filtering points inside region using spatial join...")
    # Ensure spatial index is available for speed
    points_in_shape = gpd.sjoin(points_gdf, gdf, how="inner", predicate="intersects")

    df = points_in_shape[["longitude", "latitude"]].reset_index(drop=True)
    print(f"✅ {len(df):,} points inside shapefile after spatial join")

    # --- Save ---
    df.to_csv(output_csv, index=False)
    print(f"💾 Saved grid to {output_csv}")

    return df
