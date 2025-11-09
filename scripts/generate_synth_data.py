# generate_synth_data.py
from __future__ import annotations
import pandas as pd, numpy as np, random, json, argparse, math
from datetime import datetime, timedelta, timezone
import h3

random.seed(0)
np.random.seed(0)

CITY_CENTERS = {
    "munich": (48.1351, 11.5820),
    "boston": (42.3601, -71.0589),
}

def make_hexes(center_lat, center_lon, res=8):
    hexes = set()
    for dlat in np.linspace(-0.12, 0.12, 20):
        for dlon in np.linspace(-0.18, 0.18, 30):
            lat = center_lat + dlat
            lon = center_lon + dlon
            hexes.add(h3.geo_to_h3(lat, lon, res))
    # keep a stable subset
    return list(hexes)[:300]

def synth_weather(ts):
    """Simple diurnal cycle + random noise (hourly global baseline)."""
    hour = ts.hour
    temp = 27 - 10 * np.cos(2 * np.pi * (hour / 24)) + np.random.randn() * 0.8
    rain = max(
        0.0,
        np.random.randn() * 0.2
        + (1.0 if (hour in [7, 8, 17, 18] and random.random() < 0.35) else 0.0),
    )
    wind = abs(np.random.randn() * 3 + 8)
    return temp, rain, wind

def write_route(center_lon, center_lat):
    route = [
        [center_lon - 0.15, center_lat - 0.05],
        [center_lon - 0.05, center_lat - 0.01],
        [center_lon + 0.05, center_lat + 0.01],
        [center_lon + 0.12, center_lat + 0.04],
    ]
    with open("data_sample/route.json", "w") as f:
        json.dump(route, f)

# --- helpers for events
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def gen_events_list(center_lat, center_lon, start):
    """A few fixed events + several random ones within the 72h forecast window,
    and a couple in the last 48h of history so they also influence labels."""
    events = []

    # keep your two original fixed future events
    events.append(
        {
            "name": "Stadium Game",
            "lat": center_lat + 0.02,
            "lon": center_lon + 0.03,
            "start": start + timedelta(hours=24),
            "end": start + timedelta(hours=26),
            "radius_m": 1500,
        }
    )
    events.append(
        {
            "name": "Arena Concert",
            "lat": center_lat - 0.03,
            "lon": center_lon - 0.02,
            "start": start + timedelta(hours=63),
            "end": start + timedelta(hours=66),
            "radius_m": 1200,
        }
    )

    # extra random future events (within next 72h)
    CANDIDATES = ["Con Expo", "Marathon", "Festival", "Street Fair", "Sports Derby"]
    for _ in range(4):
        name = random.choice(CANDIDATES)
        lat = center_lat + np.random.uniform(-0.05, 0.05)
        lon = center_lon + np.random.uniform(-0.08, 0.08)
        t0 = start + timedelta(hours=int(np.random.uniform(6, 70)))
        dur_h = int(np.random.uniform(2, 5))
        radius = int(np.random.uniform(800, 2000))
        events.append(
            {"name": name, "lat": lat, "lon": lon, "start": t0, "end": t0 + timedelta(hours=dur_h), "radius_m": radius}
        )

    # a couple of history events (last 48h) to influence labels
    for _ in range(2):
        name = random.choice(CANDIDATES)
        lat = center_lat + np.random.uniform(-0.05, 0.05)
        lon = center_lon + np.random.uniform(-0.08, 0.08)
        t0 = start - timedelta(hours=int(np.random.uniform(6, 48)))
        dur_h = int(np.random.uniform(2, 4))
        radius = int(np.random.uniform(800, 1800))
        events.append(
            {"name": name, "lat": lat, "lon": lon, "start": t0, "end": t0 + timedelta(hours=dur_h), "radius_m": radius}
        )

    return events

def write_events(events):
    # ensure UTC ISO strings in CSV
    rows = []
    for e in events:
        rows.append(
            {
                "name": e["name"],
                "lat": e["lat"],
                "lon": e["lon"],
                "start": e["start"].isoformat(),  # includes +00:00
                "end": e["end"].isoformat(),
                "radius_m": e["radius_m"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv("data_sample/events.csv", index=False)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="munich", choices=["munich", "boston"])
    args = ap.parse_args()

    center_lat, center_lon = CITY_CENTERS[args.city]
    start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    # history & future horizons
    hours = pd.date_range(start=start - timedelta(days=28), periods=28 * 24, freq="h", tz="UTC")
    future = pd.date_range(start=start, periods=72, freq="h", tz="UTC")
    hexes = make_hexes(center_lat, center_lon, res=8)

    # events (shared for CSV + label influence)
    events = gen_events_list(center_lat, center_lon, start)
    write_events(events)

    # simple holiday window (spans a bit before and after 'start')
    hol_start = (start - timedelta(days=1)).replace(hour=0)
    hol_end = (start + timedelta(days=3)).replace(hour=23)

    # per-hex base bias to increase spatial diversity
    hex_bias = {}
    for h in hexes:
        # centered slightly positive; clipped so probs stay sane
        hex_bias[h] = float(np.clip(np.random.normal(0.03, 0.03), -0.03, 0.12))

    # ---------- HISTORY
    hist_rows = []
    for h in hexes:
        lat, lon = h3.h3_to_geo(h)
        hb = hex_bias[h]
        for ts in hours:
            temp, rain, wind = synth_weather(ts)
            # per-hex micro-variance for weather so hexes aren't clones
            temp += np.random.randn() * 0.6
            rain = max(0.0, rain + np.random.randn() * 0.1)
            wind = max(0.0, wind + np.random.randn() * 0.5)

            rush = ts.hour in [7, 8, 17, 18]
            is_weekend = ts.weekday() >= 5
            is_holiday = (ts >= hol_start) and (ts <= hol_end)

            # check event proximity (any active event within radius)
            near_event = False
            for e in events:
                if e["start"] <= ts <= e["end"]:
                    if haversine_m(lat, lon, e["lat"], e["lon"]) <= e["radius_m"]:
                        near_event = True
                        break

            # synth probability
            prob = 0.08
            prob += 0.22 if rush else 0.0
            prob += 0.16 if rain > 0 else 0.0
            prob += 0.10 if is_weekend else 0.0
            prob += 0.20 if is_holiday else 0.0
            prob += 0.30 if near_event else 0.0
            prob += hb
            # tiny noise
            prob += np.random.uniform(-0.02, 0.02)
            prob = float(np.clip(prob, 0.01, 0.95))

            label = int(np.random.rand() < prob)

            hist_rows.append([h, lat, lon, ts, temp, rain, wind, is_weekend, is_holiday, label])

    hist = pd.DataFrame(
        hist_rows,
        columns=["h3", "lat", "lon", "time", "temp_c", "precip_mm", "wind_kph", "is_weekend", "is_holiday", "label"],
    )
    hist.to_csv("data_sample/history.csv", index=False)

    # ---------- FORECAST
    fut_rows = []
    for h in hexes:
        lat, lon = h3.h3_to_geo(h)
        for ts in future:
            temp, rain, wind = synth_weather(ts)
            temp += np.random.randn() * 0.6
            rain = max(0.0, rain + np.random.randn() * 0.1)
            wind = max(0.0, wind + np.random.randn() * 0.5)
            is_weekend = ts.weekday() >= 5
            is_holiday = (ts >= hol_start) and (ts <= hol_end)
            fut_rows.append([h, lat, lon, ts, temp, rain, wind, is_weekend, is_holiday])

    fut = pd.DataFrame(
        fut_rows,
        columns=["h3", "lat", "lon", "time", "temp_c", "precip_mm", "wind_kph", "is_weekend", "is_holiday"],
    )
    fut.to_csv("data_sample/forecast.csv", index=False)

    # route for the UI
    write_route(center_lon, center_lat)

    print(
        f'[{args.city}] wrote history.csv ({len(hist)} rows), '
        f'forecast.csv ({len(fut)} rows), route.json, events.csv ({len(events)} events)'
    )

if __name__ == "__main__":
    main()
