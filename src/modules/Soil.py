import rasterio
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rasterio.mask import mask
from typing import List, Dict, Tuple
import pyodbc

from scripts.clip_raster_with_shape import clip_raster_with_shape


class SoilVisualizer:
    def __init__(self, raster_path: str, csv_path: str, region_gdf: gpd.GeoDataFrame):
        self.raster_path = raster_path
        self.csv_path = csv_path
        self.region_gdf = region_gdf

        self.hwsd_image, self.hwsd_transform, self.hwsd_meta = (
            self._clip_raster_with_shape()
        )
        self.features_df = self._load_features()
        self.smu_dict = self._build_smu_dict()
        self.numeric_features, self.categorical_features = self._separate_features()

    # === 🔹 Private Helper Methods ===
    def _clip_raster_with_shape(self) -> Tuple[np.ndarray, any, dict]:
        """Clip raster using a GeoDataFrame mask."""
        return clip_raster_with_shape(self.raster_path, self.region_gdf)

    # === 2️⃣ Load shapefile and clip raster ===
    # # Assuming you already have your combined Algeria & Tunisia GeoDataFrame
    # # (you can also use a shapefile path here)
    # hwsd_image, hwsd_transform, hwsd_meta = clip_raster_with_shape(raster_path, alg_tun_gpd)

    def _load_features(self) -> pd.DataFrame:
        """Load and preprocess soil features CSV."""
        df = pd.read_csv(self.csv_path)
        df["HWSD2_SMU_ID"] = df["HWSD2_SMU_ID"].astype(int)
        return df.groupby("HWSD2_SMU_ID", as_index=False).mean(numeric_only=True)

    def _build_smu_dict(self) -> Dict[int, dict]:
        """Map each SMU ID to its soil feature values."""
        return self.features_df.set_index("HWSD2_SMU_ID").to_dict(orient="index")

    def _separate_features(self) -> Tuple[List[str], List[str]]:
        """Separate features into numeric and categorical."""
        numeric, categorical = [], []
        for col in self.features_df.columns:
            if col == "HWSD2_SMU_ID":
                continue
            if pd.api.types.is_numeric_dtype(self.features_df[col]):
                numeric.append(col)
            else:
                categorical.append(col)
        return numeric, categorical

    # === 🔹 Visualization Methods ===
    def _plot_numeric(self, data: np.ndarray, title: str, cmap="viridis"):
        """Plot numeric soil property raster."""
        masked = np.ma.masked_invalid(data)
        plt.figure(figsize=(10, 8))
        im = plt.imshow(masked, cmap=cmap)
        plt.colorbar(im, label=title)
        plt.title(f"Soil Property: {title}")
        plt.axis("off")
        plt.show()

    def _plot_categorical(self, data: np.ndarray, title: str):
        """Plot categorical soil property raster."""
        safe = np.where((data == "") | (data == None), "NoData", data).astype(str)
        unique_vals = np.unique(safe)
        cmap = plt.get_cmap("tab20", len(unique_vals))
        color_map = {v: cmap(i) for i, v in enumerate(unique_vals)}

        rgb = np.zeros((*safe.shape, 3))
        for r in range(safe.shape[0]):
            for c in range(safe.shape[1]):
                rgb[r, c] = color_map[safe[r, c]][:3]

        plt.figure(figsize=(10, 8))
        plt.imshow(rgb)
        plt.title(f"Soil Property: {title}")
        plt.axis("off")
        plt.show()

    # === 🔹 Main Public Method ===
    def visualize_all(self, properties: List[str] = None):
        """Visualize all soil properties or a selected subset."""
        rows, cols = self.hwsd_image.shape
        props_to_plot = properties or self.numeric_features + self.categorical_features

        for prop in props_to_plot:
            if prop not in self.features_df.columns:
                print(f"⚠️ Skipping {prop}: not found in CSV.")
                continue

            print(f"🗺️ Plotting {prop}...")

            # Initialize raster array
            dtype = float if prop in self.numeric_features else object
            prop_raster = np.full(
                (rows, cols), np.nan if dtype == float else "", dtype=dtype
            )

            # Fill raster from SMU mapping
            for r in range(rows):
                for c in range(cols):
                    smu_val = self.hwsd_image[r, c]
                    if not np.isnan(smu_val):
                        su_id = int(smu_val)
                        val = self.smu_dict.get(su_id, {}).get(prop)
                        if val is not None:
                            prop_raster[r, c] = val

            # Plot appropriate type
            if prop in self.numeric_features:
                self._plot_numeric(prop_raster, prop)
            else:
                self._plot_categorical(prop_raster, prop)


class SoilDataExtractor:
    def __init__(
        self,
        mdb_path: str,
        region_gdf: gpd.GeoDataFrame,
        raster_path: str,
        country_shp_path:str,
        output_csv: str = "D1_soil_features_alg_tun.csv",
    ):
        self.mdb_path = mdb_path
        self.country_shp_path = country_shp_path
        self.region_gdf = region_gdf
        self.raster_path = raster_path
        self.output_csv = output_csv

        # Initialize internal variables
        self.conn = None
        self.layers_df = None
        self.filtered_df = None
        self.out_image = None
        self.out_transform = None
        self.out_meta = None

    # === 🔹 Database Operations ===
    def _connect(self):
        """Establish MDB connection."""
        conn_str = (
            r"DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};"
            rf"DBQ={self.mdb_path};"
        )
        self.conn = pyodbc.connect(conn_str)

    def _load_layers(self, table_name: str = "HWSD2_LAYERS"):
        """Load soil layers table from Access database."""
        if self.conn is None:
            self._connect()
        self.layers_df = pd.read_sql(f"SELECT * FROM {table_name}", self.conn)
        self.conn.close()

    # === 🔹 Data Filtering ===
    def _filter_top_layer(self, layer_name: str = "D1"):
        """Keep only the top soil layer."""
        if self.layers_df is not None:
            self.layers_df = self.layers_df[self.layers_df["LAYER"] == layer_name]

    def _select_features(self, features: List[str]):
        """Keep only available and relevant features."""
        available = [f for f in features if f in self.layers_df.columns]
        self.layers_df = self.layers_df[available]

    # === 🔹 Spatial Operations ===
    def _clip_raster_with_shape(
        self, gdf: gpd.GeoDataFrame
    ) -> Tuple[np.ndarray, any, dict]:
        """Clip raster using polygons from GeoDataFrame."""
        return clip_raster_with_shape(self.raster_path, self.region_gdf)

    def _filter_countries(self, iso_codes: List[str]) -> gpd.GeoDataFrame:
        """Load shapefile and filter by ISO3 country codes."""
        countries = gpd.read_file(self.country_shp_path)
        return countries[countries["ISO3CD"].isin(iso_codes)]

    # === 🔹 Processing ===
    def process(self, iso_codes: List[str] = ["DZA", "TUN"], top_layer: str = "D1"):
        """Run the entire extraction and filtering pipeline."""
        print("🔹 Loading layers from MDB...")
        self._load_layers()
        self._filter_top_layer(top_layer)

        features = [
            "HWSD2_SMU_ID",
            "COARSE",
            "SAND",
            "SILT",
            "CLAY",
            "TEXTURE_USDA",
            "TEXTURE_SOTER",
            "BULK",
            "REF_BULK",
            "ORG_CARBON",
            "PH_WATER",
            "TOTAL_N",
            "CN_RATIO",
            "CEC_SOIL",
            "CEC_CLAY",
            "CEC_EFF",
            "TEB",
            "BSAT",
            "ALUM_SAT",
            "ESP",
            "TCARBON_EQ",
            "GYPSUM",
            "ELEC_COND",
        ]
        self._select_features(features)

        print("🔹 Filtering countries...")
        alg_tun_gdf = self._filter_countries(iso_codes)

        print("🔹 Clipping raster...")
        self.out_image, self.out_transform, self.out_meta = (
            self._clip_raster_with_shape(alg_tun_gdf)
        )

        print("🔹 Extracting SMU IDs from raster...")
        unique_ids = np.unique(self.out_image[~np.isnan(self.out_image)]).astype(int)

        print("🔹 Filtering layer data for Algeria & Tunisia...")
        self.filtered_df = self.layers_df[
            self.layers_df["HWSD2_SMU_ID"].isin(unique_ids)
        ]

        print(f"✅ Extracted {len(self.filtered_df)} records.")

    # === 🔹 Export ===
    def save_to_csv(self):
        """Save filtered data to CSV."""
        if self.filtered_df is not None:
            self.filtered_df.to_csv(self.output_csv, index=False)
            print(f"✅ Saved to {self.output_csv}")
        else:
            print("⚠️ No data to save. Run process() first.")
