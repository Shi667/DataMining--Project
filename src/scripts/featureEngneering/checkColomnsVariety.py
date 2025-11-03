import pandas as pd
import matplotlib.pyplot as plt
import re
import os


def analyze_categorical_columns(
    data, pie_threshold=10, save_fig=False, output_folder=None
):
    """
    Analyse les colonnes catégorielles/alphanumériques :
      - data peut être un DataFrame ou un chemin vers un CSV
      - affiche nb et % de valeurs distinctes
      - affiche ou sauvegarde un pie chart si peu de valeurs distinctes
    """

    def is_natural_number(s):
        """Vérifie si une valeur est un entier naturel (0, 1, 2, 3, ...)"""
        try:
            x = float(s)
            return x.is_integer() and x >= 0
        except:
            return False

    def is_alphanumeric_column(series):
        """Vérifie si une colonne contient des valeurs alphanumériques ou naturelles"""
        # Ignore les colonnes float
        if pd.api.types.is_float_dtype(series):
            return False
        # Garde les colonnes entières
        if pd.api.types.is_integer_dtype(series):
            return True
        # Vérifie un échantillon de valeurs textuelles
        sample = series.dropna().astype(str).head(50)
        return all(
            re.match(r"^[a-zA-Z0-9\s_-]*$", val) or is_natural_number(val)
            for val in sample
        )

    # 🔹 Charger le DataFrame si c’est un chemin CSV
    if isinstance(data, str):
        if not os.path.exists(data):
            raise FileNotFoundError(f"❌ Fichier introuvable : {data}")
        df = pd.read_csv(data)
        print(f"📂 Fichier CSV chargé : {data}\n")
    elif isinstance(data, pd.DataFrame):
        df = data
        print("📊 Analyse du DataFrame :\n")
    else:
        raise TypeError(
            "❌ 'data' doit être un DataFrame ou le chemin vers un fichier CSV."
        )

    for col in df.columns:
        s = df[col]
        if not is_alphanumeric_column(s):
            continue

        distinct_vals = s.dropna().unique()
        n_distinct = len(distinct_vals)
        pct_distinct = (n_distinct / len(s)) * 100

        print(f"🔹 {col}: {n_distinct} valeurs distinctes ({pct_distinct:.2f}%)")

        if n_distinct <= pie_threshold:
            counts = s.value_counts().head(pie_threshold)
            plt.figure(figsize=(6, 6))
            plt.pie(
                counts,
                labels=counts.index,
                autopct="%1.1f%%",
                startangle=90,
                textprops={"fontsize": 9},
            )
            plt.title(f"Répartition des valeurs pour '{col}'")

            if save_fig:
                if not output_folder:
                    output_folder = "piecharts"
                os.makedirs(output_folder, exist_ok=True)
                plt.savefig(f"{output_folder}/{col}_piechart.png")
                plt.close()
            else:
                plt.show()
