import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from affine import Affine
from shapely.ops import unary_union
from rasterio.features import geometry_mask
from src.scripts.clip_raster_with_shape import clip_raster_with_shape


def raster_to_dataframe_filter_background(
    out_image: np.ndarray,
    out_transform: Affine,
    shape_input,
    value_name: str = "value",
) -> pd.DataFrame:
    """
    Convert clipped raster to DataFrame while removing background NaNs and
    preserving internal NaNs. Uses rasterio.features.geometry_mask for robust
    inside/outside detection.

    Returns DataFrame with columns: [value_name, x, y]
    """
    import geopandas as gpd

    # --- 1. Read/prepare geometry ---
    if isinstance(shape_input, str):
        clip_gdf = gpd.read_file(shape_input)
    elif isinstance(shape_input, gpd.GeoDataFrame):
        clip_gdf = shape_input.copy()
    else:
        raise TypeError("shape_input must be a file path (str) or GeoDataFrame")

    # combine geometries
    clip_geometry = clip_gdf.geometry.unary_union

    # --- 2. Build inside mask (True for pixels INSIDE the geometry) ---
    # geometry_mask returns True for masked pixels (i.e., outside) by default.
    mask_outside = geometry_mask(
        [clip_geometry],
        out_shape=out_image.shape,
        transform=out_transform,
        invert=False,
    )
    inside_mask = ~mask_outside  # True for pixels inside the polygon

    # --- 3. Detect whether there are NaNs INSIDE the polygon ---
    nan_inside = np.isnan(out_image) & inside_mask
    inside_nan_exists = np.any(nan_inside)

    # --- 4. Build DataFrame using vectorized indexing (memory-efficient) ---
    # Keep only pixels that are inside the polygon (we always want inside pixels)
    inside_row_idx, inside_col_idx = np.where(inside_mask)
    if len(inside_row_idx) == 0:
        # nothing inside
        print("Warning: geometry contains no pixels in this raster extent.")
        return pd.DataFrame(columns=[value_name, "x", "y"])

    # Extract values for inside pixels
    inside_values = out_image[inside_row_idx, inside_col_idx]

    # Compute coords (cell centers)
    xs, ys = out_transform * (inside_col_idx + 0.5, inside_row_idx + 0.5)

    df = pd.DataFrame({value_name: inside_values, "x": xs, "y": ys})

    if not inside_nan_exists:
        # Fast path: there are NO NaNs inside polygon, so drop any NaNs if present
        # (should normally be none because inside_mask already restricts to polygon)
        df = df.dropna(subset=[value_name]).reset_index(drop=True)
        print("Fast path: no internal NaNs detected — background removed quickly.")
    else:
        # Slow path: there are NaNs inside the polygon, so keep them (we already only have inside pixels)
        # but we still return only inside pixels (some of which will be NaN).
        df = df.reset_index(drop=True)
        print("Slow path: internal NaNs detected — preserved internal missing values.")

    return df


# -------------------------
# Full test with elevation (memory-safe + hole test)
# -------------------------
if __name__ == "__main__":
    import geopandas as gpd
    import numpy as np
    import matplotlib.pyplot as plt
    from shapely.ops import unary_union
    from rasterio.features import geometry_mask

    # 1️⃣ Load shapefile and build unified geometry
    alg_tun_shp = gpd.read_file("./data/shapefiles/combined/alg_tun.shp")
    alg_tun_gpd = gpd.GeoDataFrame(
        geometry=[alg_tun_shp.geometry.union_all()], crs=alg_tun_shp.crs
    )

    # 2️⃣ Clip elevation raster
    elevation_file = "./data/elevation_dataset/be15_grd/w001001.adf"
    elevation_image, elevation_transform, elevation_meta = clip_raster_with_shape(
        elevation_file, alg_tun_gpd
    )

    # 3️⃣ Build inside mask (to ensure hole is within polygon)
    mask_outside = geometry_mask(
        [alg_tun_gpd.geometry.union_all()],
        out_shape=elevation_image.shape,
        transform=elevation_transform,
        invert=False,
    )
    inside_mask = ~mask_outside
    inside_coords = np.argwhere(inside_mask)
    if inside_coords.size == 0:
        raise RuntimeError("No pixels fall inside the polygon for this raster extent.")

    center_idx = inside_coords[len(inside_coords) // 2]
    center_row, center_col = int(center_idx[0]), int(center_idx[1])

    # 4️⃣ Introduce artificial hole
    hole_size = 30  # half-size in pixels
    r0 = max(0, center_row - hole_size)
    r1 = min(elevation_image.shape[0], center_row + hole_size)
    c0 = max(0, center_col - hole_size)
    c1 = min(elevation_image.shape[1], center_col + hole_size)

    elevation_image[r0:r1, c0:c1] = np.nan
    print(
        f"🕳️ Introduced artificial hole (inside polygon) at rows {r0}:{r1}, cols {c0}:{c1}"
    )

    # 5️⃣ Apply filtering function
    elev_df = raster_to_dataframe_filter_background(
        out_image=elevation_image,
        out_transform=elevation_transform,
        shape_input=alg_tun_shp,
        value_name="elevation",
    )

    # 6️⃣ Diagnostics
    total_inside_pixels = np.count_nonzero(inside_mask)
    missing_inside_after = elev_df["elevation"].isna().sum()
    missing_pct_inside = missing_inside_after / total_inside_pixels * 100.0
    print(f"Total pixels inside polygon: {total_inside_pixels:,}")
    print(
        f"Missing pixels inside polygon after filtering (preserved NaNs): {missing_inside_after:,} ({missing_pct_inside:.4f}%)"
    )
    print(f"Returned DataFrame rows: {len(elev_df):,}")

    # 7️⃣ Memory-safe imshow
    step = 4  # downsampling factor for raster
    elev_small = elevation_image[::step, ::step]

    extent = [
        elevation_transform.c,
        elevation_transform.c + elevation_transform.a * elevation_image.shape[1],
        elevation_transform.f + elevation_transform.e * elevation_image.shape[0],
        elevation_transform.f,
    ]

    plt.figure(figsize=(8, 6))
    plt.imshow(elev_small, cmap="terrain", extent=extent, origin="upper")
    plt.colorbar(label="Elevation (m)")
    plt.title("Elevation raster (downsampled, with artificial hole)")
    plt.tight_layout()
    plt.show()

    # 8️⃣ Scatter plot from DataFrame (downsample if large)
    if len(elev_df) == 0:
        print("No points to plot from DataFrame.")
    else:
        nplot = min(1_000_000, len(elev_df))
        if len(elev_df) > nplot:
            print(f"Downsampling DataFrame to {nplot:,} points for plotting.")
            plot_df = elev_df.sample(nplot, random_state=42)
        else:
            plot_df = elev_df

        plt.figure(figsize=(8, 6))
        plt.scatter(
            plot_df["x"],
            plot_df["y"],
            c=plot_df["elevation"],
            s=0.6,
            cmap="terrain",
            marker=".",
        )
        plt.title(
            f"Elevation DataFrame (preserved internal NaNs) - {missing_pct_inside:.4f}% missing inside"
        )
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.colorbar(label="Elevation (m)")
        plt.tight_layout()
        plt.show()
