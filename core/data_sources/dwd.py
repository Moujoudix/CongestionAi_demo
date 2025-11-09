from __future__ import annotations
import pandas as pd
def load_dwd_forecast(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=['time'])
    need = ['lat','lon','time','temp_c','precip_mm','wind_kph']
    miss = [c for c in need if c not in df.columns]
    if miss: raise ValueError(f"DWD CSV missing {miss}")
    return df
