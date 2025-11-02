import rasterio
from rasterio import shutil as rio_shutil
from rasterio.crs import CRS
import os
import geopandas as gpd
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


def convert_and_clip_esri_grid(src_folder, shape_input, dst_tif):
    """
    Converts an ESRI GRID (.adf) dataset folder to a clipped GeoTIFF file.
    Automatically fixes CRS and clips using shapefile.
    """
    if not os.path.isdir(src_folder):
        raise FileNotFoundError(f"Source folder not found: {src_folder}")

    print("🛰️ Reading ESRI GRID from:", src_folder)

    with rasterio.open(src_folder) as src:
        meta = src.meta.copy()

        # --- Fix CRS if missing ---
        if src.crs is None:
            prj_path = os.path.join(src_folder, "prj.adf")
            if os.path.exists(prj_path):
                print("📐 Found 'prj.adf' — attempting CRS fix...")
                try:
                    with open(prj_path, "r") as f:
                        wkt = f.read().strip()
                        crs = CRS.from_wkt(wkt)
                        meta.update(crs=crs)
                        print("✅ CRS successfully fixed from prj.adf")
                except Exception as e:
                    print("⚠️ Could not read CRS from prj.adf:", e)
            else:
                print("⚠️ No CRS info found — saving without projection.")
        else:
            print("✅ CRS already present:", src.crs)

    # --- Save temporary GeoTIFF (for clipping) ---
    temp_tif = os.path.join(os.path.dirname(dst_tif), "temp_elevation.tif")
    print("💾 Creating temporary GeoTIFF...")
    rio_shutil.copy(src_folder, temp_tif, driver="GTiff")

    # --- Clip raster ---
    print("✂️ Clipping raster with shapefile...")
    out_image, out_transform, out_meta = clip_raster_with_shape(temp_tif, shape_input)

    # --- Save clipped GeoTIFF ---
    print("💾 Saving clipped GeoTIFF to:", dst_tif)
    out_meta.update(count=1)

    if out_image.ndim == 2:
        out_image = np.expand_dims(out_image, axis=0)

    with rasterio.open(dst_tif, "w", **out_meta) as dest:
        dest.write(out_image)

    os.remove(temp_tif)
    print("🧹 Temporary file removed.")
    print("✅ Done! Saved clipped elevation raster at:", dst_tif)


# === Example usage ===
if __name__ == "__main__":
    src_folder = r"../../../data/elevation_dataset/be15_grd/"
    shape_path = r"../../../data/shapefiles/combined/alg_tun.shp"
    dst_tif = r"../../../data/elevation_dataset/selevation_clipped.tif"

    convert_and_clip_esri_grid(src_folder, shape_path, dst_tif)
