import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm
import rasterio


def generate_grid_in_shape(
    shapefile_path: str, resolution: float, output_csv: str
) -> pd.DataFrame:
    """
    Generate a regular (longitude, latitude) grid inside a shapefile region.
    """
    print("📂 Loading shapefile...")
    gdf = gpd.read_file(shapefile_path).to_crs("EPSG:4326")

    # --- Compute bounds ---
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    print(f"🗺️ Bounding box: {bounds}")

    # --- Create full grid ---
    xs = np.arange(bounds[0], bounds[2], resolution)
    ys = np.arange(bounds[1], bounds[3], resolution)
    print(f"📏 Grid: {len(xs)} × {len(ys)} = {len(xs) * len(ys):,} total points")

    # --- Convert to points ---
    points = [Point(x, y) for y in ys for x in xs]
    points_gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")

    # --- Keep only points inside the shapefile ---
    print("🔍 Filtering points inside region...")
    points_in_shape = gpd.sjoin(points_gdf, gdf, how="inner", predicate="intersects")
    coords_filtered = [(pt.x, pt.y) for pt in points_in_shape.geometry]
    print(f"✅ {len(coords_filtered):,} points inside shapefile")

    # --- Save ---
    df = pd.DataFrame(coords_filtered, columns=["longitude", "latitude"])
    df.to_csv(output_csv, index=False)
    print(f"💾 Saved grid to {output_csv}")

    return df
