import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm
import rasterio


import pandas as pd
import geopandas as gpd
import numpy as np


def generate_grid_in_shape(
    shapefile_path: str, resolution: float, output_csv: str
) -> pd.DataFrame:
    print("📂 Loading shapefile...")
    gdf = gpd.read_file(shapefile_path).to_crs("EPSG:4326")

    # --- Compute bounds ---
    bounds = gdf.total_bounds
    print(f"🗺️ Bounding box: {bounds}")

    # --- Create full grid using vectorization ---
    # Create coordinate arrays
    xs = np.arange(bounds[0], bounds[2], resolution)
    ys = np.arange(bounds[1], bounds[3], resolution)
    print(
        f"📏 Grid: {len(xs)} × {len(ys)} = {len(xs) * len(ys):,} total potential points"
    )

    # Use meshgrid to generate all combinations (X, Y)
    lon_grid, lat_grid = np.meshgrid(xs, ys)

    # Flatten the arrays into two columns (much faster than Point creation loop)
    # The columns are already in (x=longitude, y=latitude) order
    points_df = pd.DataFrame(
        {
            "longitude": lon_grid.ravel(),
            "latitude": lat_grid.ravel(),
        }
    )

    # Convert to GeoDataFrame using `points_from_xy` for speed
    points_gdf = gpd.GeoDataFrame(
        points_df,
        geometry=gpd.points_from_xy(points_df.longitude, points_df.latitude),
        crs="EPSG:4326",
    )

    # --- Keep only points inside the shapefile ---
    print("🔍 Filtering points inside region...")
    # Use sjoin to find intersections
    points_in_shape = gpd.sjoin(points_gdf, gdf, how="inner", predicate="intersects")

    # Extract coordinates directly from the columns that were already prepared
    df = points_in_shape[["longitude", "latitude"]].reset_index(drop=True)

    print(f"✅ {len(df):,} points inside shapefile")

    # --- Save ---
    df.to_csv(output_csv, index=False)
    print(f"💾 Saved grid to {output_csv}")

    return df
