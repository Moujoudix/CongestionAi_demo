from __future__ import annotations
import numpy as np, pandas as pd
def add_time_features(df: pd.DataFrame, time_col: str = 'time') -> pd.DataFrame:
    df = df.copy(); t = pd.to_datetime(df[time_col])
    df['dow'] = t.dt.dayofweek; df['hour'] = t.dt.hour
    df['is_weekend'] = (df['dow']>=5).astype(int)
    df['sin_hour'] = np.sin(2*np.pi*df['hour']/24.0)
    df['cos_hour'] = np.cos(2*np.pi*df['hour']/24.0)
    return df
def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'precip_mm' in df.columns: df['rain_flag'] = (df['precip_mm']>0.1).astype(int)
    if 'temp_c' in df.columns: df['temp_z'] = (df['temp_c']-df['temp_c'].mean())/(df['temp_c'].std()+1e-6)
    if 'wind_kph' in df.columns: df['wind_z'] = (df['wind_kph']-df['wind_kph'].mean())/(df['wind_kph'].std()+1e-6)
    return df
def add_lags_rollups(df: pd.DataFrame, group_col: str = 'h3', time_col: str = 'time', target_col: str = 'label') -> pd.DataFrame:
    df = df.sort_values([group_col, time_col]).copy()
    for lag in [1,2,3]: df[f'lag_{lag}'] = df.groupby(group_col)[target_col].shift(lag)
    df['roll_mean_3'] = df.groupby(group_col)[target_col].rolling(3).mean().reset_index(level=0, drop=True)
    return df
def merge_osm_density(df: pd.DataFrame, osm_df: pd.DataFrame | None, on: str = 'h3') -> pd.DataFrame:
    if osm_df is None: return df
    return df.merge(osm_df, on=on, how='left')
