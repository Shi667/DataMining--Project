import geopandas as gpd
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import pandas as pd


def show_landcover(
    source,
    legend_path,
    code_column="GRIDCODE",
    title="Land Cover Map",
    figsize=(10, 8),
    alpha=0.9,
    edgecolor="none",
    legend_position="bottom",  # "right" also supported
):
    """
    Display a land cover shapefile using colors and labels from a GlobCover legend.

    Parameters
    ----------
    source : str or GeoDataFrame
        Path to the shapefile (.shp) or a loaded GeoDataFrame.
    legend_path : str
        Path to the legend Excel file (e.g., globcover_legend.xls).
    code_column : str, default="GRIDCODE"
        Column in the shapefile that corresponds to the 'Value' column in the legend.
    title : str, optional
        Title of the plot.
    figsize : tuple, default=(10, 8)
        Figure size.
    alpha : float, default=0.9
        Transparency for the polygons.
    edgecolor : str, default="none"
        Border color.
    legend_position : str, default="bottom"
        Where to place the legend — "bottom" or "right".
    """

    # --- Load shapefile ---
    if isinstance(source, str):
        gdf = gpd.read_file(source)
        name = os.path.basename(source)
    elif isinstance(source, gpd.GeoDataFrame):
        gdf = source
        name = "GeoDataFrame"
    else:
        raise TypeError("source must be a file path (str) or a GeoDataFrame")

    if code_column not in gdf.columns:
        raise ValueError(
            f"'{code_column}' not found in shapefile columns: {gdf.columns}"
        )

    # --- Load legend ---
    legend = pd.read_excel(legend_path)
    legend.columns = legend.columns.str.strip()

    required_cols = {"Value", "Label", "Red", "Green", "Blue"}
    if not required_cols.issubset(legend.columns):
        raise ValueError(
            f"Legend file missing required columns. Expected at least: {required_cols}"
        )

    # --- Build color mapping ---
    legend["color"] = legend.apply(
        lambda row: f"#{int(row['Red']):02x}{int(row['Green']):02x}{int(row['Blue']):02x}",
        axis=1,
    )

    value_to_color = dict(zip(legend["Value"], legend["color"]))
    value_to_label = dict(zip(legend["Value"], legend["Label"]))

    # --- Assign colors to GeoDataFrame ---
    gdf["_color"] = gdf[code_column].map(value_to_color)
    gdf["_label"] = gdf[code_column].map(value_to_label)

    # For any missing values → gray color
    gdf["_color"] = gdf["_color"].fillna("#808080")
    gdf["_label"] = gdf["_label"].fillna("Unknown")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    gdf.plot(ax=ax, color=gdf["_color"], edgecolor=edgecolor, alpha=alpha)
    ax.set_title(title, fontsize=14)
    ax.axis("off")

    # --- Legend ---
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label,
            markerfacecolor=color,
            markersize=8,
        )
        for label, color in zip(legend["Label"], legend["color"])
    ]

    if legend_position == "right":
        ax.legend(
            handles=handles,
            title="Land Cover Classes",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8,
        )
    elif legend_position == "bottom":
        ax.legend(
            handles=handles,
            title="Land Cover Classes",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=3,
            fontsize=8,
        )
    else:
        raise ValueError("legend_position must be 'bottom' or 'right'")

    plt.tight_layout()
    plt.show()


def describe_shapefile(source):
    """
    Print key metadata and attributes of a shapefile or GeoDataFrame.

    Parameters
    ----------
    source : str or GeoDataFrame
        Path to a shapefile (.shp) or a GeoDataFrame.
    """

    # --- Case 1: File path ---
    if isinstance(source, str):
        gdf = gpd.read_file(source)
        name = os.path.basename(source)
    # --- Case 2: GeoDataFrame ---
    elif isinstance(source, gpd.GeoDataFrame):
        gdf = source
        name = "GeoDataFrame"
    else:
        raise TypeError("source must be a file path (str) or a GeoDataFrame")

    print(f"\n🗺️ Exploring shapefile: {name}")
    print(f"CRS: {gdf.crs}")
    print(f"Number of features: {len(gdf)}")
    print(f"Geometry types: {gdf.geom_type.unique()}")
    print(f"Bounds: {gdf.total_bounds}")
    print(f"Columns: {list(gdf.columns)}")

    # Summary of attribute columns (non-geometry)
    non_geom_cols = [c for c in gdf.columns if c != "geometry"]
    if non_geom_cols:
        print("\n📋 Attribute summary:")
        print(gdf[non_geom_cols].describe(include="all").transpose())
    else:
        print("\n⚠️ No attribute columns (geometry only).")
