import pandas as pd
from typing import List, Union


def merge_data_sources(
    data_list: List[Union[str, pd.DataFrame]],
    on: list[str] | str,
    how: str = "inner",
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Merge multiple CSV files and/or DataFrames on one or more common columns.

    Parameters
    ----------
    data_list : list of (str | pd.DataFrame)
        List of CSV file paths and/or pandas DataFrames to merge.
    on : list[str] | str
        Column(s) to merge on.
    how : str, optional
        Type of merge: 'inner', 'outer', 'left', 'right'. Default is 'inner'.
    output_path : str, optional
        If provided, saves the merged DataFrame to this path.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame containing all data.
    """
    if not data_list:
        raise ValueError(
            "The input list is empty. Please provide CSV paths or DataFrames."
        )

    # Convert all CSVs to DataFrames
    dfs = []
    for item in data_list:
        if isinstance(item, str):  # CSV path
            df = pd.read_csv(item)
        elif isinstance(item, pd.DataFrame):
            df = item
        else:
            raise TypeError(
                f"Unsupported type: {type(item)}. Expected str or pd.DataFrame."
            )
        dfs.append(df)

    # Merge all DataFrames iteratively
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on=on, how=how)

    # Save if output path provided
    if output_path:
        merged_df.to_csv(output_path, index=False)
        print(f"Merged CSV saved to: {output_path}")

    return merged_df
