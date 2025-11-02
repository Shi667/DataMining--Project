import numpy as np
import pandas as pd
import geopandas as gpd
from affine import Affine
from shapely.geometry import Point
from typing import Union  # For type hinting the shape_input
from rasterio.transform import xy
from rasterio.warp import transform as transform_coords

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from rasterio.warp import transform as transform_coords
from affine import Affine


def raster_to_dataframe_filter_background(
    out_image: np.ndarray,
    out_transform: Affine,
    shape_input,
    src_crs,
    value_name: str = "value",
    dst_crs: str = "EPSG:4326",
) -> pd.DataFrame:
    """
    Convert a clipped raster (2D array) to a DataFrame of lon, lat, and value,
    keeping internal NaNs and filtering out background pixels outside the mask geometry.

    Parameters
    ----------
    out_image : np.ndarray
        2D clipped raster with background set to np.nan.
    out_transform : affine.Affine
        Affine transform of the raster.
    shape_input : str or GeoDataFrame
        Path to the shapefile or the GeoDataFrame used for clipping.
    src_crs : str or dict
        CRS of the raster.
    value_name : str, optional
        Column name for pixel values.
    dst_crs : str, optional
        Destination CRS (default: EPSG:4326).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [lon, lat, value], filtered to the shape area.
    """

    # 1️⃣ Read or copy the shape input
    if isinstance(shape_input, str):
        clip_gdf = gpd.read_file(shape_input)
    elif isinstance(shape_input, gpd.GeoDataFrame):
        clip_gdf = shape_input.copy()
    else:
        raise TypeError("Shape input must be a file path or a GeoDataFrame.")

    clip_geometry = clip_gdf.geometry.union_all()

    # 2️⃣ Generate all pixel centers
    rows, cols = out_image.shape
    col_indices, row_indices = np.meshgrid(np.arange(cols), np.arange(rows))
    x_coords, y_coords = out_transform * (
        col_indices.flatten() + 0.5,
        row_indices.flatten() + 0.5,
    )

    # 3️⃣ Reproject to lon/lat
    lon, lat = transform_coords(src_crs, dst_crs, x_coords, y_coords)

    df = pd.DataFrame({"lon": lon, "lat": lat, value_name: out_image.flatten()})

    # 4️⃣ Filter background using the shape geometry (in lon/lat)
    clip_gdf = clip_gdf.to_crs(dst_crs)
    clip_geometry = clip_gdf.geometry.union_all()

    points = gpd.GeoSeries([Point(xy) for xy in zip(df["lon"], df["lat"])], crs=dst_crs)
    mask = points.within(clip_geometry)

    df_filtered = df[mask].reset_index(drop=True)

    return df_filtered


def raster_to_dataframe(image, transform, src_crs, dst_crs="EPSG:4326"):
    """
    Convert a raster array to a DataFrame with lon, lat, and data values.

    Parameters
    ----------
    image : np.ndarray
        2D raster array (single band).
    transform : affine.Affine
        Affine transform of the raster.
    src_crs : str or dict
        CRS of the raster (e.g. "EPSG:32632").
    dst_crs : str, optional
        Destination CRS, default is "EPSG:4326" (lon/lat).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: lon, lat, value
    """
    # Flatten raster values
    values = image.flatten()

    # Get row/col indices
    rows, cols = np.indices(image.shape)
    rows = rows.flatten()
    cols = cols.flatten()

    # Compute x, y in source CRS
    xs, ys = xy(transform, rows, cols)
    xs = np.array(xs)
    ys = np.array(ys)

    # Convert to lon/lat
    lon, lat = transform_coords(src_crs, dst_crs, xs, ys)

    # Build DataFrame
    df = pd.DataFrame({"lon": lon, "lat": lat, "value": values})

    # Remove NaN / nodata
    df = df[~np.isnan(df["value"])]

    return df
