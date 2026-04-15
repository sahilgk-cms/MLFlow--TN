from db.db_loader import load_cases_statewise, load_lulc, load_weather_data_statewise
from features.data_processing import  clean_and_merge_statewise_weather_data, calculate_total_cases, merge_statewise_and_lulc, get_static_and_merge_with_temporal, get_statewise_temporal_and_cases, merge_statewise_temporal_and_cases, get_statewise_zones_and_merge_with_final, load_gis_data, preprocess_gis_data
from features.aggregations import aggregate_weekly_median, aggregate_weekly_sum
from features.lag_features import shift_cases_forward, create_lag_features, fill_lag_values
from features.rolling_features import create_rolling_features
from features.interactions import add_weather_interactions
from features.temporal_features import add_month_sin_cos
#from db.db_loader import append_df_to_db
from features.schema import TEMPORAL_COLS, RAIN_COLS, CASE_COL, CASE_COL_LAG_2
import sqlalchemy
import pandas as pd
import argparse


def build_features(engine: sqlalchemy.engine.base.Engine, database_config: dict,  
                   feature_config: dict) -> pd.DataFrame:


    statewise_data = load_cases_statewise(engine, state=database_config.get("state"),
                                          disease=database_config.get("disease"))
    weather_data = load_weather_data_statewise(engine, state=database_config.get("state"))

    statewise_data = clean_and_merge_statewise_weather_data(statewise_data, weather_data)


    df_lulc = load_lulc(engine, state = database_config.get("state"))


    statewise_temporal, statewise_cases = get_statewise_temporal_and_cases(statewise_data)

    statewise_temporal = aggregate_weekly_median(statewise_temporal)
    statewise_cases = aggregate_weekly_sum(statewise_cases)

    statewise_temporal = merge_statewise_temporal_and_cases(statewise_temporal, statewise_cases)

    statewise_temporal = shift_cases_forward(statewise_temporal, shift_by=feature_config.get("shift_by"))

    statewise_temporal = create_lag_features(statewise_temporal, features=TEMPORAL_COLS+RAIN_COLS,
                                             lags=feature_config.get("lags_weather"))
    
    statewise_temporal = create_lag_features(statewise_temporal, features=[CASE_COL],
                                             lags=feature_config.get("lags_cases"))
    
    statewise_temporal = create_rolling_features(statewise_temporal, features=TEMPORAL_COLS+RAIN_COLS,
                                                 windows=feature_config.get("rolling_windows"))
    
    statewise_temporal = create_rolling_features(statewise_temporal, features=[CASE_COL_LAG_2],
                                                 windows=feature_config.get("rolling_windows"))
    
    statewise_temporal = merge_statewise_and_lulc(statewise_temporal, df_lulc)

    statewise_final = get_static_and_merge_with_temporal(statewise_temporal)

    statewise_final = add_weather_interactions(statewise_final,
                                               precip_threshold=feature_config.get("precip_threshold"),
                                               humidity_threshold=feature_config.get("humidity_threshold"),
                                               temp_threshold=feature_config.get("temp_threshold"))
    
    statewise_final = add_month_sin_cos(statewise_final)

    statewise_final = fill_lag_values(statewise_final)

    return statewise_final
    