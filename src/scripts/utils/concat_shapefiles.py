import geopandas as gpd
import pandas as pd


def concat_shapefiles(path1, path2):
    """
    Concatenate two shapefiles into a single GeoDataFrame.

    Parameters
    ----------
    path1 : str
        Path to the first shapefile.
    path2 : str
        Path to the second shapefile.
    output_path : str, optional
        Path to save the combined shapefile. If None, file is not saved.
    plot : bool, optional
        Whether to display a plot of the combined shapefile.

    Returns
    -------
    GeoDataFrame
        The combined GeoDataFrame.
    """
    # Load shapefiles
    gdf1 = gpd.read_file(path1)
    gdf2 = gpd.read_file(path2)

    # Ensure same CRS
    if gdf1.crs != gdf2.crs:
        gdf2 = gdf2.to_crs(gdf1.crs)

    # Concatenate
    combined = pd.concat([gdf1, gdf2], ignore_index=True)

    return combined


# === Combine Algeria and Tunisia landcover shapefiles ===
alg_tun_shp_landcvr = concat_shapefiles(
    "../data/land_dataset/algeria/dza_gc_adg.shp",
    "../data/land_dataset/tunisia/tun_gc_adg.shp",
)

# === Describe the combined shapefile ===
print("✅ Combined Algeria–Tunisia landcover shapefile created successfully!\n")
# 1. Specify the output folder and filename
output_folder = "../../data/land_dataset/combined/"
output_filename = "alg_tun_landcvr.shp"
full_output_path = output_folder + output_filename

# 2. (Optional but recommended) Ensure the output directory exists
import os

os.makedirs(output_folder, exist_ok=True)

# 3. Use the .to_file() method to save the GeoDataFrame
alg_tun_shp_landcvr.to_file(
    full_output_path,
    driver="ESRI Shapefile",  # Explicitly set the driver for clarity (optional, as it's often the default)
)

print(f"Shapefile successfully saved to: {full_output_path}")
