import geopandas as gpd
from shapely.geometry import box
import os


def save_shapefile_north_of_latitude(
    input_shapefile_path,
    output_shapefile_path,
    latitude_cut,
):
    """
    Save the part of a shapefile that is NORTH of a given latitude.

    Parameters
    ----------
    input_shapefile_path : str
        Path to original shapefile
    output_shapefile_path : str
        Path where the clipped shapefile will be saved
    latitude_cut : float
        Latitude threshold (keep geometry >= this latitude)
    """

    # -----------------------------
    # Load shapefile
    # -----------------------------
    gdf = gpd.read_file(input_shapefile_path)

    # Ensure geographic CRS
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)

    # -----------------------------
    # Create clipping box
    # -----------------------------
    minx, miny, maxx, maxy = gdf.total_bounds

    north_box = box(
        minx,
        latitude_cut,  # south boundary
        maxx,
        maxy,
    )

    north_gdf = gdf.clip(north_box)

    # -----------------------------
    # Create output directory
    # -----------------------------
    os.makedirs(os.path.dirname(output_shapefile_path), exist_ok=True)

    # -----------------------------
    # Save shapefile
    # -----------------------------
    north_gdf.to_file(output_shapefile_path)

    print(
        f"✅ Shapefile saved:\n"
        f"   {output_shapefile_path}\n"
        f"   (North of latitude {latitude_cut})"
    )

    return north_gdf
