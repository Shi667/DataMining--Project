import pandas as pd

def calculate_seasonal_fire_percentage(file_path: str) :
    """
    Calculates the percentage of fire occurrences per season from a dataset.

    Seasons are defined as:
    - Winter: Dec, Jan, Feb
    - Spring: Mar, Apr, May
    - Summer: Jun, Jul, Aug
    - Autumn/Fall: Sep, Oct, Nov

    Args:
        file_path (str): The path to the fire dataset (e.g., 'fire_data.csv').

    Returns:
        pd.DataFrame: A DataFrame with 'Season', 'Count', and 'Percentage' of fire occurrences.
    """
    try:
        # 1. Load the data
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at path '{file_path}'")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return pd.DataFrame()

    # Ensure the required column is present
    if 'acq_date' not in df.columns:
        print("Error: The dataset must contain an 'acq_date' column.")
        return pd.DataFrame()

    # 2. Ensure 'acq_date' is in datetime format and extract the month
    # Assuming 'acq_date' is in a format pandas can infer (e.g., 'YYYY-MM-DD')
    df['acq_date'] = pd.to_datetime(df['acq_date'], errors='coerce')
    df.dropna(subset=['acq_date'], inplace=True)

    # Extract the month number (1 for Jan, 12 for Dec)
    df['month'] = df['acq_date'].dt.month

    # 3. Define the season mapping function
    def get_season(month: int) -> str:
        """Maps a month number to a season based on the specified criteria."""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Autumn/Fall'
        else:
            return 'Unknown'

    # 4. Apply the function to create a new 'Season' column
    df['Season'] = df['month'].apply(get_season)

    # 5. Group by 'Season' and count the occurrences
    seasonal_counts = df['Season'].value_counts().reset_index()
    seasonal_counts.columns = ['Season', 'Count']

    # 6. Calculate the percentage of total fire occurrences
    total_fires = seasonal_counts['Count'].sum()
    seasonal_counts['Percentage'] = (seasonal_counts['Count'] / total_fires) * 100

    # Format the percentage for cleaner output
    seasonal_counts['Percentage'] = seasonal_counts['Percentage'].round(2).astype(str) + '%'

    # 7. Return the results, sorted by the season's start
    # Define a custom order for sorting
    season_order = ['Winter', 'Spring', 'Summer', 'Autumn/Fall']
    seasonal_counts['Sort_Key'] = pd.Categorical(
        seasonal_counts['Season'],
        categories=season_order,
        ordered=True
    )
    seasonal_counts.sort_values('Sort_Key', inplace=True)
    seasonal_counts.drop(columns=['Sort_Key'], inplace=True)

    return seasonal_counts

