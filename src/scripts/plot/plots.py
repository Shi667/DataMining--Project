import matplotlib.pyplot as plt


def plot_fire_map(base_map, fire_points, title="Fire Detections", color="red", size=8):
    """
    Plot fire detection points over a base map.

    Parameters:
        base_map (GeoDataFrame): Shapefile of countries or regions.
        fire_points (GeoDataFrame): Fire detections as points.
        title (str): Title for the map.
        color (str): Color for fire points.
        size (int): Marker size.
    """
    fig, ax = plt.subplots(figsize=(9, 9))
    base_map.plot(ax=ax, color="whitesmoke", edgecolor="black")
    fire_points.plot(
        ax=ax, markersize=size, color=color, alpha=0.6, label="Fire Detections"
    )

    plt.title(f"{title}", fontsize=14)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
