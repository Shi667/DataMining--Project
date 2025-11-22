import geopandas as gpd
import os


from shapely.geometry import box


def clip_by_latitude(input_shp, output_folder, min_latitude, output_name="cropped.shp"):
    gdf = gpd.read_file(input_shp)

    # Ensure correct CRS
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    # Create a clipping rectangle (west=-180, east=180, south=min_latitude, north=90)
    clip_rect = box(-180, min_latitude, 180, 90)

    clipped = gpd.clip(gdf, clip_rect)

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_name)
    clipped.to_file(output_path)

    print("✔ Exact clipping done:", output_path)


clip_by_latitude(
    input_shp="../../../data/shapefiles/combined/full/alg_tun.shp",
    output_folder="../data/shapefiles/combined/north/",
    min_latitude=30.0,  # cut Sahara
    output_name="alg_tun_north.shp",
)


import geopandas as gpd
import matplotlib.pyplot as plt

# 1. Specify the path to your Shapefile
# NOTE: Replace 'path/to/your/shapefile.shp' with the actual file path.
shapefile_path = "../../../data/shapefiles/combined/north/alg_tun_north.shp"

try:
    # 2. Load the Shapefile into a GeoDataFrame
    gdf = gpd.read_file(shapefile_path)

    # 3. Create a plot (figure and axis)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # 4. Plot the GeoDataFrame
    # 'edgecolor' sets the color of the boundary lines.
    # 'facecolor' sets the fill color.
    # 'ax=ax' tells GeoPandas to plot on the axis we created.
    gdf.plot(ax=ax, edgecolor="black", facecolor="lightblue")

    # 5. Add a title and remove axis for a cleaner map view
    ax.set_title(f"Displaying Shapefile: {shapefile_path.split('/')[-1]}", fontsize=16)
    ax.set_axis_off()  # Hides the x and y coordinate axes

    # 6. Display the plot
    plt.show()

except FileNotFoundError:
    print(f"Error: The Shapefile at '{shapefile_path}' was not found.")
except Exception as e:
    print(f"An error occurred while reading or plotting the Shapefile: {e}")
