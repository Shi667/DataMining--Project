import numpy as np
import pandas as pd
import geopandas as gpd
from affine import Affine
from shapely.geometry import Point
from typing import Union  # For type hinting the shape_input
from rasterio.transform import xy


def raster_to_dataframe_filter_background(
    out_image: np.ndarray,
    out_transform: Affine,
    shape_input: Union[str, gpd.GeoDataFrame],
    value_name: str = "value",
) -> pd.DataFrame:
    """
    Transforms a clipped raster NumPy array (2D) into a pandas DataFrame,
    eliminating only the background NaN values and keeping internal NaN values.
    Includes an optimization: if no internal NaNs are detected, the slow
    spatial check is skipped, and all NaNs (background) are dropped.

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
        A DataFrame with 'value_name', 'x', and 'y', retaining internal missing data (NaNs).
    """

    # --- 1. Get the geometry(ies) for masking ---
    if isinstance(shape_input, str):
        # Read shapefile/GeoJSON using geopandas
        clip_gdf = gpd.read_file(shape_input)
    elif isinstance(shape_input, gpd.GeoDataFrame):
        clip_gdf = shape_input.copy()
    else:
        raise TypeError("Shape input must be a file path (str) or GeoDataFrame.")

    # Get the combined geometry boundary (unary_union)
    clip_geometry = clip_gdf.geometry.union_all()

    # --- 2. Convert ALL pixels (including all NaNs) to DataFrame ---
    rows, cols = out_image.shape
    col_indices, row_indices = np.meshgrid(np.arange(cols), np.arange(rows))

    # Flatten indices and calculate coordinates (cell center)
    col_indices_flat = col_indices.flatten()
    row_indices_flat = row_indices.flatten()
    x_coords, y_coords = out_transform * (
        col_indices_flat + 0.5,
        row_indices_flat + 0.5,
    )

    df = pd.DataFrame({value_name: out_image.flatten(), "x": x_coords, "y": y_coords})
    df_nan = df[df[value_name].isna()]

    # --- OPTIMIZATION CHECK: Are there any NaNs inside the geometry? ---
    # The image is clipped, so NaNs outside the boundary box are background NaNs.
    # If the number of NaNs in the *clipped* raster is equal to the number of
    # NaNs outside the *shape geometry*, then there are no internal NaNs.
    # However, an easier check is to just look for *non-NaN* values *inside* the geometry.
    # The fastest way to check for internal NaNs is still the spatial check, but
    # we can try to skip the *DataFrame* filtering step if it's not needed.

    # A simpler and safe optimization (The EDIT you requested):
    # If the raster has *any* value that is NOT NaN, then there *might* be internal NaNs.
    # We will ONLY skip the expensive spatial check if *all* NaN values are guaranteed to be background.

    # --- 3. Check for internal NaNs to decide on the filtering method ---
    # To definitively know if *any* NaN is internal, we must do a spatial check.
    # A faster alternative is to assume *all* NaNs are background if all *non-NaN* pixels are inside.

    # Optimization: Filter *out* all NaNs, then check if any remaining non-NaN pixel is *outside*.
    # If no non-NaN pixel is outside, then all *present* data is inside, and we must check for NaNs.

    # Check if there are ANY NaNs in the DataFrame. If not, the function is done.
    if df_nan.empty:
        print("No missing data found in the bounding box. Returning all pixels.")
        return df

    print(f"Total NaN pixels in bounding box: {len(df_nan)}")

    # Check 1: Are there any non-NaN values?
    df_non_nan = df[df[value_name].notna()]

    if not df_non_nan.empty:
        # Check 2 (Expensive): Do any non-NaN values fall *outside* the geometry?
        # This confirms if the clipping was tight to the geometry or just the bounding box.

        # Create shapely Point objects for non-NaN coordinates
        points_non_nan = [Point(x, y) for x, y in zip(df_non_nan["x"], df_non_nan["y"])]

        # Check which *non-NaN* points are INSIDE the clip geometry
        is_inside_non_nan = np.array(
            [clip_geometry.contains(point) for point in points_non_nan]
        )

        # Count non-NaN pixels OUTSIDE the geometry
        non_nan_outside_count = (~is_inside_non_nan).sum()

        if non_nan_outside_count > 0:
            # Case 1: Non-NaN pixels are OUTSIDE the geometry.
            # This is unexpected for a well-clipped raster but necessitates the full spatial check,
            # as it implies the geometry is smaller than the clipped extent.
            print(
                "Non-NaN pixels found outside the geometry. Performing full spatial check."
            )
            perform_full_filter = True
        else:
            # Case 2: All non-NaN pixels are INSIDE the geometry.
            # Now we must check the NaNs: are they background or internal?
            # Create shapely Point objects for NaN coordinates
            points_nan = [Point(x, y) for x, y in zip(df_nan["x"], df_nan["y"])]

            # Check which *NaN* points are INSIDE the clip geometry
            is_inside_nan = np.array(
                [clip_geometry.contains(point) for point in points_nan]
            )

            # Count NaN pixels INSIDE the geometry (These are the internal NaNs)
            internal_nan_count = is_inside_nan.sum()

            if internal_nan_count > 0:
                # Case 2a: Internal NaNs exist. We must perform the full filter to keep them.
                print(
                    f"Found {internal_nan_count} internal NaN pixels. Performing full spatial check."
                )
                perform_full_filter = True
            else:
                # Case 2b: NO internal NaNs exist (All NaNs are background).
                # We can skip the slow spatial check and just drop all NaNs.
                print(
                    "No internal NaN pixels found. Skipping spatial check and dropping all NaNs."
                )
                perform_full_filter = False

    else:
        # Case 3: The entire clipped area is NaN (e.g., clipping an area with no data).
        # Since all non-NaN pixels are inside (vacuously true), we check for internal NaNs:
        points_nan = [Point(x, y) for x, y in zip(df_nan["x"], df_nan["y"])]
        is_inside_nan = np.array(
            [clip_geometry.contains(point) for point in points_nan]
        )
        internal_nan_count = is_inside_nan.sum()

        if internal_nan_count > 0:
            print(
                f"Entire image is NaN, but {internal_nan_count} internal NaNs exist. Performing full spatial check."
            )
            perform_full_filter = True
        else:
            print(
                "Entire image is NaN and all NaNs are outside the geometry. Returning empty DataFrame."
            )
            return pd.DataFrame({value_name: [], "x": [], "y": []})

    # --- 4. Apply the required filtering method ---
    if perform_full_filter:
        # Full spatial filter (original slow method)
        # Combine non-NaN points and NaN points for the full check if needed

        # Note: The most efficient way when full filter is needed is to compute *all* points once.
        # However, to reuse the checks above, we'll re-do the inside check if Case 1 or 2a was true.

        # If Case 2b was true, we've already set perform_full_filter=False.

        # To avoid re-checking, let's just go with the original logic if the flag is True,
        # but using the pieces we've already computed.

        # df_filtered will be the union of (df_non_nan where inside) and (df_nan where inside).

        # Combine and filter
        df_inside_non_nan = df_non_nan[is_inside_non_nan]
        df_inside_nan = df_nan[is_inside_nan]

        # Concatenate the data inside the geometry
        df_filtered = pd.concat([df_inside_non_nan, df_inside_nan]).reset_index(
            drop=True
        )

    else:  # perform_full_filter is False (Case 2b - Only background NaNs exist)
        # Optimized filter: Drop ALL NaNs. This is much faster.
        df_filtered = df[df[value_name].notna()].reset_index(drop=True)
        # Note: If Case 3 (All NaN, no internal), this branch is not hit, and an empty DF is returned above.

    print(f"Original pixels in bounding box: {len(df)}")
    print(f"Final pixels inside geometry: {len(df_filtered)}")

    return df_filtered


def raster_to_dataframe(image, transform):
    """
    Convert a raster array to a DataFrame with x, y coordinates and value.

    Parameters
    ----------
    image : np.ndarray
        2D raster array (single band).
    transform : affine.Affine
        Affine transform of the raster.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: x, y, elevation
    """
    # Flatten the array
    values = image.flatten()

    # Get row, col indices
    rows, cols = np.indices(image.shape)
    rows = rows.flatten()
    cols = cols.flatten()

    # Convert to coordinates
    xs, ys = xy(transform, rows, cols)

    # Build DataFrame
    df = pd.DataFrame({"x": xs, "y": ys, "elevation": values})

    # Remove missing values if any
    df = df[~np.isnan(df["elevation"])]

    return df
