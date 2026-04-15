import pandas as pd
from features.schema import RENAME_MAPPING, TEMPORAL_COLS, RAIN_COLS, CASE_COL, STATIC_COLS, STATIC_COLS_2, STATIC_COLS_3, DATE_COL, DATE_COL_WEEK_START, GROUP_COL, CASE_COL_LAG_2, MIN_DATE, ECO_PROB_COLS
from typing import Tuple


def clean_and_merge_statewise_weather_data(statewise_data: pd.DataFrame,
                                          weather_data: pd.DataFrame) -> pd.DataFrame:
    statewise_data = weather_data.merge(statewise_data, on = ['date', 'sub_district', 'district', 'state'], how = 'inner')
    statewise_data["date"] = pd.to_datetime(statewise_data["date"])
    statewise_data['year'] = statewise_data['date'].dt.year
    statewise_data = statewise_data.rename(columns=RENAME_MAPPING)
    statewise_data = statewise_data.rename(columns= {'district' : 'dist_name'})
    return statewise_data



def merge_statewise_and_lulc(statewise_data: pd.DataFrame,
                       df_lulc: pd.DataFrame) -> pd.DataFrame:
    df_lulc['sub_district'] = (
                            df_lulc['sub_district']
                            .astype(str)          # ensure string type
                            .str.strip()          # remove leading/trailing spaces
                            .str.title()          # First letter capital, rest small
                        )
    statewise_merge= statewise_data.merge(df_lulc, on= ['year', 'sub_district'], how = 'left')
    return statewise_merge


def get_statewise_temporal_and_cases(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    statewise_temporal = (
        df.groupby(['date', 'sub_district', 'dist_name'])[TEMPORAL_COLS]
        .mean()
        .reset_index()
    )
    statewise_cases = (
        df.groupby(['date', 'sub_district', 'dist_name'])[RAIN_COLS + [CASE_COL]]
        .sum()
        .reset_index()
    )
    return statewise_temporal, statewise_cases

    
def merge_statewise_temporal_and_cases(statewise_temporal: pd.DataFrame,
                                       statewise_cases: pd.DataFrame) -> pd.DataFrame:
    
    statewise_cases = statewise_cases[[GROUP_COL] + [DATE_COL_WEEK_START] + RAIN_COLS + [CASE_COL]]
    statewise_temporal = statewise_temporal.merge(statewise_cases, 
                                                  on = [GROUP_COL, DATE_COL_WEEK_START],
                                                  how="left")
    statewise_temporal['year'] = statewise_temporal[DATE_COL_WEEK_START].dt.year
    return statewise_temporal


def get_static_and_merge_with_temporal(statewise_temporal: pd.DataFrame) -> pd.DataFrame:
    statewise_static = statewise_temporal[STATIC_COLS]
    statewise_static = statewise_temporal.groupby([GROUP_COL])[STATIC_COLS_2].sum().reset_index()
    statewise_static.drop_duplicates(subset= GROUP_COL, inplace= True)
    statewise_final = statewise_temporal.merge(statewise_static, on= GROUP_COL, how= 'left')
    return statewise_final


def calculate_total_cases(df: pd.DataFrame) -> pd.DataFrame:
    df['cases_total_pop'] = (df[CASE_COL_LAG_2]*10000)/(df['total_population']+1)
    df['cases_rural_pop'] = (df[CASE_COL_LAG_2]*10000)/(df['rural_population']+1)
    df['cases_urban_pop'] = (df[CASE_COL_LAG_2]*10000)/(df['urban_population']+1)
    return df

def get_statewise_zones_and_merge_with_final(statewise_data: pd.DataFrame,
                                             statewise_final: pd.DataFrame) -> pd.DataFrame:
    statewise_zones = (statewise_data.groupby(GROUP_COL)['agro_zones'].apply(lambda x: x.value_counts().idxmax()).reset_index())
    statewise_zones = statewise_zones.drop_duplicates(subset= [GROUP_COL, 'agro_zones'])
    statewise_final = statewise_final.merge(statewise_zones, on=GROUP_COL, how="left")
    return statewise_final


def load_gis_data(filepath: str) -> pd.DataFrame:
    dem = pd.read_csv(filepath, 
                      usecols= ['dist_name', 'elev_0_200_area_km2', 'elev_200_400_area_km2',
                                   'elev_400_600_area_km2', 'elev_600_800_area_km2', 'elev_800_1200_area_km2',
                                   'elev_1200_1800_area_km2', 'elev_1800_2600_area_km2', 'elev_2600_3000_area_km2'])
    return dem

def preprocess_gis_data(df: pd.DataFrame) -> pd.DataFrame:
    # 1. identify elevation columns
    elev_cols = [c for c in df.columns if c.startswith("elev_") and c.endswith("_area_km2")]

    # 2. compute total elevation area per district
    df["total_elev_area_km2"] = df[elev_cols].sum(axis=1, skipna=True)

    # 3. create percentage columns
    for col in elev_cols:
        pct_col = col.replace("_area_km2", "_pct")
        df[pct_col] = (df[col] / df["total_elev_area_km2"]) * 100

    # 4. drop the original area columns and the total
    df = df.drop(columns=elev_cols + ["total_elev_area_km2"])
    return df