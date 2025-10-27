import numpy as np
import pandas as pd
import geopandas as gpd
from affine import Affine
from shapely.geometry import Point


def raster_to_dataframe_filter_background(
    out_image: np.ndarray, out_transform: Affine, shape_input
) -> pd.DataFrame:
    """
    Transforms a clipped raster NumPy array (2D) into a pandas DataFrame,
    eliminating only the background NaN values and keeping internal NaN values.

    Parameters
    ----------
    out_image : np.ndarray
        Clipped raster data (2D) with all NoData/background set to np.nan.
    out_transform : affine.Affine
        Transform for the clipped raster.
    shape_input : str or GeoDataFrame
        The original shapefile path or GeoDataFrame used for clipping.

    Returns
    -------
    pd.DataFrame
        A DataFrame with 'value', 'x', and 'y', retaining internal missing data (NaNs).
    """

    # --- 1. Get the geometry(ies) for masking ---
    if isinstance(shape_input, str):
        # Read shapefile/GeoJSON using geopandas (more convenient for spatial checks)
        clip_gdf = gpd.read_file(shape_input)
    elif isinstance(shape_input, gpd.GeoDataFrame):
        clip_gdf = shape_input.copy()
    else:
        raise TypeError("Shape input must be a file path (str) or GeoDataFrame.")

    # Ensure the GeoDataFrame uses a geometry column compatible with Shapely's contains
    # FIX IS HERE: Add parentheses to CALL the method
    clip_geometry = clip_gdf.geometry.union_all()  # <-- CORRECTED LINE

    # --- 2. Convert ALL pixels (including all NaNs) to DataFrame ---
    rows, cols = out_image.shape
    col_indices, row_indices = np.meshgrid(np.arange(cols), np.arange(rows))

    # Flatten indices and calculate coordinates
    col_indices_flat = col_indices.flatten()
    row_indices_flat = row_indices.flatten()
    x_coords, y_coords = out_transform * (
        col_indices_flat + 0.5,
        row_indices_flat + 0.5,
    )  # Adding 0.5 for cell center

    df = pd.DataFrame({"value": out_image.flatten(), "x": x_coords, "y": y_coords})

    # --- 3. Create a spatial mask to identify background pixels ---
    print("Creating spatial mask to isolate background...")

    # Create shapely Point objects from coordinates
    # Note: Creating points in a list comprehension can be slow for large rasters.
    # Consider using GeoPandas' `sjoin` or `within` for a potentially faster vectorization later if performance is an issue.
    points = [Point(x, y) for x, y in zip(df["x"], df["y"])]

    # Check which points are INSIDE the clip geometry
    # The 'clip_geometry' is the combined shape boundary (unary_union)
    is_inside = [clip_geometry.contains(point) for point in points]

    # --- 4. Apply the mask to keep only pixels INSIDE the geometry ---
    # We filter the DataFrame to keep only rows where the pixel is 'inside'
    # This automatically removes the background NaNs, but keeps the internal NaNs.
    df_filtered = df[is_inside].reset_index(drop=True)

    print(f"Original pixels in bounding box: {len(df)}")
    print(f"Final pixels inside geometry: {len(df_filtered)}")

    return df_filtered
