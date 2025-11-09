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


def sample_raster_at_points(
    raster_path: str, points_csv: str, output_csv: str, value_name: str
):
    """
    Samples a raster file at specific (longitude, latitude) points.
    Handles CRS mismatches and missing values robustly.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    print(f"📂 Sampling {raster_path}...")

    # --- Load points ---
    df = pd.read_csv(points_csv)
    gdf_points = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
    )

    # --- Open raster ---
    with rasterio.open(raster_path) as src:
        # If raster CRS is not EPSG:4326, reproject points
        if src.crs.to_string() != "EPSG:4326":
            gdf_points = gdf_points.to_crs(src.crs)

        coords = [(geom.x, geom.y) for geom in gdf_points.geometry]

        # --- Clamp points to raster bounds to avoid edge precision errors ---
        minx, miny, maxx, maxy = src.bounds
        coords_clamped = [
            (
                min(max(x, minx + 1e-6), maxx - 1e-6),
                min(max(y, miny + 1e-6), maxy - 1e-6),
            )
            for x, y in coords
        ]

        # --- Sample raster ---
        values = []
        nodata = src.nodata if src.nodata is not None else np.nan
        for val in tqdm(
            src.sample(coords_clamped),
            total=len(coords),
            desc=f"Extracting {value_name}",
        ):
            v = val[0]
            if np.isnan(v) or v == nodata:
                values.append(np.nan)
            else:
                values.append(v)

    df[value_name] = values
    valid_ratio = np.sum(~np.isnan(values)) / len(values) * 100
    print(
        f"✅ Saved {len(df)} samples to {output_csv} ({valid_ratio:.2f}% valid values)"
    )

    df.to_csv(output_csv, index=False)
    return df


# ======================================================
# Example Usage
# ======================================================

# Step 1: Generate grid (only once)
grid_df = generate_grid_in_shape(
    "../../../data/shapefiles/combined/alg_tun.shp",
    resolution=0.05,
    output_csv="grid_points.csv",
)

# Step 2: Sample raster on that grid
sample_raster_at_points(
    "../../../data/climate_dataset/5min/max/wc2.1_cruts4.09_5m_tmax_2024-01.tif",
    "grid_points.csv",
    "tmax_jan.csv",
    "tmax",
)
