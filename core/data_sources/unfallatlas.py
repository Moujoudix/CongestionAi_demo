from __future__ import annotations
import pandas as pd, h3
def load_unfallatlas(csv_path: str, res: int = 8) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'time' not in df.columns:
        if {'DATUM','UHRZEIT'}.issubset(df.columns):
            df['time'] = pd.to_datetime(df['DATUM'] + ' ' + df['UHRZEIT'])
        else: raise ValueError('Provide time or DATUM+UHRZEIT')
    if 'lat' not in df.columns or 'lon' not in df.columns: raise ValueError('Need lat, lon')
    df['time'] = pd.to_datetime(df['time']).dt.floor('H')
    df['h3'] = df.apply(lambda r: h3.geo_to_h3(r['lat'], r['lon'], res), axis=1)
    g = df.groupby(['h3','time']).size().reset_index(name='incidents')
    g['lat'] = g['h3'].apply(lambda h: h3.h3_to_geo(h)[0])
    g['lon'] = g['h3'].apply(lambda h: h3.h3_to_geo(h)[1])
    g['label'] = (g['incidents']>0).astype(int)
    return g[['h3','lat','lon','time','label']]
