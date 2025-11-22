import geopandas as gpd
import rasterio
from rasterio.mask import mask
import fiona
import numpy as np


def clip_raster_with_shape(raster_input, shape_input):
    """
    Clip a raster using a shapefile or GeoDataFrame.

    Parameters
    ----------
    raster_input : str
        Path to the raster file (.tif, .adf, etc.).
    shape_input : str or GeoDataFrame
        Path to the shapefile or a GeoDataFrame geometry.

    Returns
    -------
    out_image : np.ndarray
        Clipped raster data as a NumPy array (2D).
    out_transform : affine.Affine
        Transform for the clipped raster.
    out_meta : dict
        Updated raster metadata (CRS, transform, width, height, etc.).
    """

    # --- Handle raster input ---
    if not isinstance(raster_input, str):
        raise TypeError("Raster input must be a file path (str).")

    # --- Handle shape input ---
    if isinstance(shape_input, str):
        with fiona.open(shape_input) as shape:
            geoms = [feature["geometry"] for feature in shape]
    elif isinstance(shape_input, gpd.GeoDataFrame):
        geoms = [feature.__geo_interface__ for feature in shape_input.geometry]
    else:
        raise TypeError("Shape input must be a file path (str) or GeoDataFrame.")

    # --- Open and clip raster ---
    with rasterio.open(raster_input) as src:
        out_image, out_transform = mask(src, geoms, crop=True)
        out_image = out_image[0].astype(float)

        # Handle NoData values
        nodata = src.nodata
        if nodata is not None:
            out_image[out_image == nodata] = np.nan

        # Copy and update metadata
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "height": out_image.shape[0],
                "width": out_image.shape[1],
                "transform": out_transform,
            }
        )

    return out_image, out_transform, out_meta


