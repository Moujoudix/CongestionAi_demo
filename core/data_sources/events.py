from __future__ import annotations
import pandas as pd, numpy as np, holidays, math

def add_holiday_flag(df: pd.DataFrame, country: str = 'DE', time_col: str = 'time') -> pd.DataFrame:
    df = df.copy(); df[time_col] = pd.to_datetime(df[time_col])
    hol = holidays.country_holidays(country)
    df['is_holiday'] = df[time_col].dt.date.map(lambda d: 1 if d in hol else 0)
    return df

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2-lat1); dl = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def _to_naive_utc(s: pd.Series) -> pd.Series:
    # robust: works if s is already naive or already tz-aware
    s = pd.to_datetime(s, utc=True)
    return s.dt.tz_convert("UTC").dt.tz_localize(None)

def load_events(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["start", "end"])
    # make both columns tz-naive UTC so they can be compared with df["time"]
    df["start"] = _to_naive_utc(df["start"])
    df["end"]   = _to_naive_utc(df["end"])
    return df

def join_events_proximity(df: pd.DataFrame, events: pd.DataFrame, time_col: str = 'time') -> pd.DataFrame:

    df = df.copy()
    events = events.copy()
    # Normalize times (safe even if already normalized)
    df[time_col]        = pd.to_datetime(df[time_col]).dt.tz_localize(None)
    events["start"]     = pd.to_datetime(events["start"]).dt.tz_localize(None)
    events["end"]       = pd.to_datetime(events["end"]).dt.tz_localize(None)

    marks = np.zeros(len(df), dtype=int); counts = np.zeros(len(df), dtype=int)
    for i, row in df.iterrows():
        t = row[time_col]; lat, lon = row['lat'], row['lon']
        sub = events[(events['start'] <= t) & (t <= events['end'])]
        c = 0; flag = 0
        for _, ev in sub.iterrows():
            d = haversine(lat, lon, ev['lat'], ev['lon'])
            if d <= float(ev['radius_m']): flag = 1; c += 1
        marks[i] = flag; counts[i] = c
    df['is_event'] = marks; df['event_count'] = counts
    return df
