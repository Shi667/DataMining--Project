# ==============================================================
# 📊 eda_utils.py — Utility functions for Exploratory Data Analysis (EDA)
# ==============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
from IPython.display import display

# ==============================================================
# 1. Basic Data Overview
# ==============================================================

def describe_dataset(df):
    print("=== Dataset Preview ===")
    display(df.head())
    print("\n=== General Information ===")
    print(df.info())
    print("\n=== Descriptive Statistics ===")
    display(df.describe(include='all'))

def check_missing_values(df):
    df_clean = df.replace(["", " ", "NaN", "nan", "NULL", "null"], np.nan)
    missing = df_clean.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) == 0:
        print("✅ No missing values detected.")
    else:
        print("⚠️ Missing values detected:")
        display(missing)
    return missing

def plot_missing_values(df):
    """
    Visualize missing values across the dataset as a heatmap.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.show()

# ==============================================================
# 2. Summary Statistics
# ==============================================================

def summary_stats(df, columns):
    return df[columns].agg(['mean', 'median', 'std', 'min', 'max', 'count'])

def report_skew_kurtosis(df, columns):
    for col in columns:
        print(f"{col}: skew={df[col].skew():.2f}, kurtosis={df[col].kurtosis():.2f}")

# ==============================================================
# 3. Visualization Functions
# ==============================================================

def plot_histogram(df, column, bins=30):
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column], kde=True, bins=bins)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

def plot_boxplot(df, column, by=None):
    plt.figure(figsize=(8, 5))
    if by:
        sns.boxplot(data=df, x=by, y=column)
        plt.title(f"Boxplot of {column} by {by}")
    else:
        sns.boxplot(data=df, y=column)
        plt.title(f"Boxplot of {column}")
    plt.show()

def plot_scatter(df, x, y, hue=None):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x, y=y, hue=hue)
    plt.title(f"Relationship between {x} and {y}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

def plot_correlation_matrix(df):
    plt.figure(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Matrix")
    plt.show()

def plot_pairplot(df, columns=None, hue=None):
    """
    Create a seaborn pairplot for multivariate relationships.
    """
    sns.pairplot(df if columns is None else df[columns + ([hue] if hue else [])], hue=hue)
    plt.show()

def compare_distributions(df, column, by):
    """
    Histograms of one variable split by a categorical column.
    """
    plt.figure(figsize=(10,6))
    for group in df[by].unique():
        sns.histplot(df[df[by]==group][column], kde=True, alpha=0.6, label=str(group))
    plt.legend(title=by)
    plt.title(f"Distribution of {column} by {by}")
    plt.show()

# ==============================================================
# 4. Outlier Detection
# ==============================================================

def detect_outliers_iqr(df, column, factor=1.5):
    

    # Compute quartiles and IQR
    Q1 = df[column].quantile(0.25)
    Q2 = df[column].quantile(0.50)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR

    # Find outliers
    outliers = df[(df[column] < lower) | (df[column] > upper)]

    # Print detailed summary
    print(f"=== Outlier Detection Report for '{column}' ===")
    print(f"Q1 (25th percentile): {Q1:.4f}")
    print(f"Q2 (Median):          {Q2:.4f}")
    print(f"Q3 (75th percentile): {Q3:.4f}")
    print(f"IQR (Q3 - Q1):        {IQR:.4f}")
    print(f"Lower bound:          {lower:.4f}")
    print(f"Upper bound:          {upper:.4f}")
    print(f"Number of outliers:   {len(outliers)} / {len(df)} rows")
    print("=" * 60)

    return outliers


# ==============================================================
# 5. Geospatial (GeoDataFrame) Functions
# ==============================================================

def plot_geodata(gdf, column=None, title="Map"):
    gdf.plot(column=column, legend=True, figsize=(10, 8))
    plt.title(title)
    plt.axis('off')
    plt.show()

def summarize_geodata(gdf):
    print("=== GeoDataFrame Info ===")
    print(gdf.info())
    print("\n=== Coordinate Reference System (CRS) ===")
    print(gdf.crs)
    print("\n=== Geometry Type Counts ===")
    print(gdf.geom_type.value_counts())
    print("\n=== Descriptive Statistics for Numeric Columns ===")
    display(gdf.describe())

def plot_spatial_heatmap(gdf, title="Spatial Heatmap", bins=100):
    x = gdf.geometry.x
    y = gdf.geometry.y
    plt.figure(figsize=(10, 7))
    plt.hexbin(x, y, gridsize=bins, cmap='YlOrRd', mincnt=1)
    plt.colorbar(label="Counts")
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

# ==============================================================
# 6. Data Quality & Integrity Checks
# ==============================================================

def data_quality_report(df):
    """
    Generate a quick data quality summary for each column.
    """
    report = pd.DataFrame({
        "Data Type": df.dtypes,
        "Missing (%)": df.isnull().mean() * 100,
        "Unique Values": df.nunique(),
        "Zeros (%)": (df == 0).mean() * 100,
        "Negative Values (%)": (df.select_dtypes(include=np.number) < 0).mean() * 100
    })

    print("=== Data Quality Report ===")
    display(report.sort_values("Missing (%)", ascending=False))
    return report



# ==============================================================
# 7. Additional Visualization Utilities
# ==============================================================

def plot_multiple_distributions(df, columns, bins=30):
    """
    Plot histograms of multiple numeric columns in a grid.
    """
    n = len(columns)
    rows = (n + 2) // 3
    plt.figure(figsize=(15, 5 * rows))
    for i, col in enumerate(columns, 1):
        plt.subplot(rows, 3, i)
        sns.histplot(df[col], kde=True, bins=bins)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


# ==============================================================


def plot_categorical_distribution(df, column, top_n=10):
    """
    Plot the most frequent categories for a categorical column.
    """
    plt.figure(figsize=(10, 5))
    counts = df[column].value_counts().nlargest(top_n)
    sns.barplot(x=counts.index, y=counts.values)
    plt.title(f"Top {top_n} Categories in '{column}'")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()


def plot_relationship_grid(df, columns):
    """
    Create pairwise scatter plots with correlation coefficients.
    """
    sns.pairplot(df[columns], kind="reg", diag_kind="kde")
    plt.suptitle("Pairwise Relationships", y=1.02)
    plt.show()



def visualize_outliers(df, column):
    """
    Combine boxplot + swarmplot to show outliers clearly.
    """
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[column], color="skyblue")
    sns.swarmplot(x=df[column], color="red", size=3)
    plt.title(f"Outlier Visualization for '{column}'")
    plt.show()


def plot_spatial_kde(gdf, title="Spatial KDE Density", bw_adjust=0.5):
    """
    Plot a 2D KDE (heatmap) of point geometries.
    """
    x, y = gdf.geometry.x, gdf.geometry.y
    plt.figure(figsize=(10, 7))
    sns.kdeplot(x=x, y=y, fill=True, cmap="Reds", bw_adjust=bw_adjust)
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()
