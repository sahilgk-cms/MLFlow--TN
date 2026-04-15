import pandas as pd
from features.schema import ECO_PROB_COLS

def add_weather_interactions(df: pd.DataFrame,
                             precip_threshold: int,
                             humidity_threshold: int,
                             temp_threshold: int,
                ) -> pd.DataFrame:

    # Create binary variable for precipitation
    df['High_Precip'] = (df['total_precipitation_sum_mm'] >= precip_threshold).astype(int)

    # Create binary variable for relative humidity
    df['High_Humidity'] = (df['relative_humidity_percent'] > humidity_threshold).astype(int)

    # Define interaction variable
    df['High_Precip_Humidity'] = df['High_Precip'] * df['High_Humidity']

    df['temperature_max'] = (df['temperature_2m_max_celsius'] >= temp_threshold).astype(int)

    # Define interaction variable
    df['High_temp_Humidity'] = df['temperature_max'] * df['High_Humidity']
    df['High_temp_Humidity_preci'] = df['temperature_max'] * df['High_Humidity'] * df['temperature_max']

    return df

