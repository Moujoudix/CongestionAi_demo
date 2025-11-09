from __future__ import annotations
import pandas as pd
def load_osm_road_density(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'h3' not in df.columns: raise ValueError('Need h3 column')
    return df
