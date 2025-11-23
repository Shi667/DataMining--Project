import os
import glob
import numpy as np
import rasterio
from clip_raster_with_shape import clip_raster_with_shape


def clip_folder_rasters(folder_path, shape_input_path):
    """
    Clips all raster (.tif) files inside a folder using a shapefile,
    and overwrites each original file with the clipped version.
    """

    # --- 1. Validate Paths ---
    if not os.path.isdir(folder_path):
        print(f"🛑 Error: Folder not found at {folder_path}")
        return
    if not os.path.exists(shape_input_path):
        print(f"🛑 Error: Shapefile not found at {shape_input_path}")
        return

    # --- 2. Find raster files ---
    raster_files = glob.glob(os.path.join(folder_path, "*.tif"))
    if not raster_files:
        print(f"⚠️ No .tif files found in {folder_path}")
        return

    print(f"🚀 Starting clipping for {len(raster_files)} raster(s)...")

    # --- 3. Process rasters ---
    for raster_input_path in raster_files:
        filename = os.path.basename(raster_input_path)
        print(f"\n🗂 Processing: {filename}")

        try:
            # Clip raster (your function)
            out_image, out_transform, out_meta = clip_raster_with_shape(
                raster_input=raster_input_path, shape_input=shape_input_path
            )

            # --- Validate output ---
            if out_image is None or out_image.size == 0:
                raise ValueError("Clipped raster is empty (no overlap with shape).")

            # --- Ensure 3D array (bands, H, W) ---
            if out_image.ndim == 2:
                out_image = out_image[np.newaxis, :, :]  # (1, H, W)

            # --- Update metadata ---
            out_meta.update(
                {
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "count": out_image.shape[0],
                    "transform": out_transform,
                    "compress": "lzw",
                }
            )

            # --- Overwrite original raster ---
            with rasterio.open(raster_input_path, "w", **out_meta) as dst:
                dst.write(out_image)

            print(f"✅ Successfully clipped and saved: {filename}")

        except Exception as e:
            print(f"❌ Error clipping {filename}: {e}")

    print("\n🎉 Clipping process finished!")


rasters_directory_tmin = r"../../../data/climate_dataset/5min/max/"
rasters_directory_tmax = r"../../../data/climate_dataset/5min/min/"
rasters_directory_prec = r"../../../data/climate_dataset/5min/prec/"

clipping_boundary = r"../../../data/shapefiles/combined/full/alg_tun.shp"
clip_folder_rasters(rasters_directory_tmin, clipping_boundary)
clip_folder_rasters(rasters_directory_tmax, clipping_boundary)
clip_folder_rasters(rasters_directory_prec, clipping_boundary)
