import numpy as np
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
import rasterio
import os
import geopandas as gpd


def show_raster(
    source,
    band=1,
    cmap="terrain",
    title=None,
    scale=None,
    colorbar_label=None,
    transform=None,
    meta=None,
):
    """
    Display a raster map with optional downsampling.

    Parameters
    ----------
    source : str or np.ndarray
        Path to a raster file (.tif, .adf, etc.) or a NumPy array (from clip_raster_with_shape()).
    band : int, default=1
        Raster band to display.
    cmap : str, default="terrain"
        Colormap for visualization.
    title : str, optional
        Plot title.
    scale : float, optional
        Downsampling factor between 0 and 1.
    colorbar_label : str, optional
        Label for colorbar.
    transform : affine.Affine, optional
        Spatial transform (required if source is a NumPy array).
    meta : dict, optional
        Metadata dictionary (as returned by clip_raster_with_shape).

    Notes
    -----
    GeoDataFrames are not supported here. Use `.plot(ax=ax)` to overlay vector data later.
    """

    # --- Case 1: GeoDataFrame (invalid) ---
    if isinstance(source, gpd.GeoDataFrame):
        raise TypeError(
            "GeoDataFrame provided. Use a raster file path or NumPy array instead. "
            "You can overlay it later using GeoDataFrame.plot(ax=ax)."
        )

    # --- Case 2: File path ---
    elif isinstance(source, str):
        with rasterio.open(source) as src:
            if scale is not None and 0 < scale < 1:
                new_height = int(src.height * scale)
                new_width = int(src.width * scale)
                data = src.read(
                    band,
                    out_shape=(1, new_height, new_width),
                    resampling=Resampling.average,
                )[0]
            else:
                data = src.read(band)[0]

            if src.nodata is not None:
                data = np.ma.masked_equal(data, src.nodata)

            extent = [
                src.bounds.left,
                src.bounds.right,
                src.bounds.bottom,
                src.bounds.top,
            ]

    # --- Case 3: NumPy array (from clip_raster_with_shape) ---
    elif isinstance(source, np.ndarray):
        data = source
        if data.ndim > 2:
            raise ValueError("NumPy raster must be 2D (single band).")

        if transform is not None and meta is not None:
            # Compute spatial extent from transform
            height, width = data.shape
            from rasterio.transform import array_bounds

            extent = array_bounds(height, width, transform)
        else:
            # Plot in pixel space if no georeferencing
            extent = None

    else:
        raise TypeError("source must be a file path (str) or NumPy array")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(data, cmap=cmap, extent=extent)
    plt.colorbar(im, ax=ax, label=colorbar_label or f"Band {band}")
    ax.set_title(title or "Raster Preview")
    ax.axis("off")
    plt.show()


def describe_raster(source, meta=None, transform=None):
    """
    Print key metadata about a raster file, GeoDataFrame, or clipped raster.

    Parameters
    ----------
    source : str, GeoDataFrame, or np.ndarray
        - Raster file path (.tif, .adf, etc.)
        - GeoDataFrame
        - NumPy array (from clip_raster_with_shape)
    meta : dict, optional
        Metadata dictionary returned by clip_raster_with_shape.
    transform : affine.Affine, optional
        Transform returned by clip_raster_with_shape.
    """

    # --- Case 1: GeoDataFrame ---
    if isinstance(source, gpd.GeoDataFrame):
        gdf = source
        print("\n🗺️ Exploring GeoDataFrame")
        print(f"CRS: {gdf.crs}")
        print(f"Number of features: {len(gdf)}")
        print(f"Geometry type(s): {gdf.geom_type.unique()}")
        print(f"Bounds: {gdf.total_bounds}")

    # --- Case 2: Raster file path ---
    elif isinstance(source, str):
        with rasterio.open(source) as src:
            print(f"\n🗺️ Exploring raster: {os.path.basename(source)}")
            print(f"CRS: {src.crs}")
            print(f"Dimensions: {src.width} x {src.height}")
            print(f"Number of bands: {src.count}")
            print(f"Resolution: {src.res}")
            print(f"Bounds: {src.bounds}")
            print(f"Data type: {src.dtypes[0]}")
            print(f"NoData value: {src.nodata}")

    # --- Case 3: NumPy array (clipped raster) ---
    elif isinstance(source, np.ndarray):
        print("\n🗺️ Exploring clipped raster (NumPy array)")

        if source.ndim == 2:
            print(f"Dimensions: {source.shape[1]} x {source.shape[0]}")
        elif source.ndim == 3:
            print(
                f"Bands: {source.shape[0]}, Dimensions: {source.shape[2]} x {source.shape[1]}"
            )

        # If metadata available, show more info
        if meta is not None:
            print(f"CRS: {meta.get('crs', 'Unknown')}")
            print(
                f"Resolution: {meta.get('transform', transform)[0]:.4f}, {abs(meta.get('transform', transform)[4]):.4f}"
            )
            print(f"Data type: {meta.get('dtype', 'Unknown')}")
            print(
                f"Bounds: {rasterio.transform.array_bounds(meta['height'], meta['width'], transform)}"
                if transform
                else "Bounds: Unknown"
            )
        else:
            print("⚠️ No metadata provided — spatial reference unknown.")

        print(f"Contains NaN values: {np.isnan(source).any()}")

    else:
        raise TypeError(
            "source must be a GeoDataFrame, raster file path (str), or NumPy array"
        )
