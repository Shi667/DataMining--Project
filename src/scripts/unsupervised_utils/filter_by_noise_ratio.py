import pandas as pd


def filter_by_noise_ratio(
    input_csv_path: str,
    output_csv_path: str,
    min_noise_ratio: float,
):
    """
    Keep only rows where noise_ratio >= min_noise_ratio.

    Parameters
    ----------
    input_csv_path : str
        Path to input CSV (must contain 'noise_ratio' column)
    output_csv_path : str
        Path where filtered CSV will be saved
    min_noise_ratio : float
        Minimum noise_ratio to keep (e.g. 0.15)
    """
    df = pd.read_csv(input_csv_path)

    if "noise_ratio" not in df.columns:
        raise ValueError("Column 'noise_ratio' not found in CSV")

    df_filtered = df[df["noise_ratio"] <= min_noise_ratio]

    df_filtered.to_csv(output_csv_path, index=False)

    print(f"Saved {len(df_filtered)} / {len(df)} rows to {output_csv_path}")
