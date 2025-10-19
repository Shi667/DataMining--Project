import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt


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
