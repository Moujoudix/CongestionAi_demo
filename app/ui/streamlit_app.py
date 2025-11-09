import os, json, h3
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from datetime import timedelta

# ---------- helpers
def _to_naive_utc(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, utc=True)
    return s.dt.tz_convert("UTC").dt.tz_localize(None)

@st.cache_data
def load_preds():
    df = pd.read_csv("data_sample/predictions.csv", parse_dates=["time"])
    df["time"] = _to_naive_utc(df["time"])
    return df

@st.cache_data
def load_events():
    if not os.path.exists("data_sample/events.csv"):
        return None
    ev = pd.read_csv("data_sample/events.csv", parse_dates=["start", "end"])
    ev["start"] = _to_naive_utc(ev["start"])
    ev["end"] = _to_naive_utc(ev["end"])
    return ev

@st.cache_data
def load_route():
    if not os.path.exists("data_sample/route.json"):
        return None
    with open("data_sample/route.json", "r") as f:
        path = json.load(f)  # list of [lon, lat]
    pts = [{"lon": float(lon), "lat": float(lat)} for lon, lat in path]
    return pts

def best_route_shift(df_all: pd.DataFrame, sel_time: pd.Timestamp, route_pts, res=8, shifts=range(-6, 7)):
    if not route_pts:
        return None

    # pre-map route points to H3 cells
    route_map = [{"h3": h3.geo_to_h3(p["lat"], p["lon"], res), "lat": p["lat"], "lon": p["lon"]} for p in route_pts]
    route_df = pd.DataFrame(route_map)

    times = set(df_all["time"].unique())
    valid = [s for s in shifts if (sel_time + pd.to_timedelta(s, "h")) in times]
    if not valid:
        return None

    rows, per_shift_points = [], {}
    for s in valid:
        t = sel_time + pd.to_timedelta(s, "h")
        view = df_all.loc[df_all["time"] == t, ["h3", "risk"]]
        joined = route_df.merge(view, on="h3", how="left")
        joined["risk"] = joined["risk"].astype("float64").fillna(0.5)  # neutral if off-grid
        rows.append({"shift": s, "mean_risk": float(joined["risk"].mean())})
        per_shift_points[s] = joined

    res_df = pd.DataFrame(rows).sort_values("shift")
    best_s = int(res_df.loc[res_df["mean_risk"].idxmin(), "shift"])
    return {"table": res_df, "best_shift": best_s, "best_points": per_shift_points[best_s]}

# ---------- UI
st.set_page_config(page_title="CongestionAI Map", layout="wide")
st.title("CongestionAI — 72h Congestion Risk (demo)")
st.caption("Swap synthetic data with real sources; see README.")

df = load_preds()
min_t, max_t = df["time"].min(), df["time"].max()

col1, col2 = st.columns([2, 1])
with col2:
    st.subheader("Controls")
    ts = st.slider(
        "Select time",
        min_value=min_t.to_pydatetime(),
        max_value=max_t.to_pydatetime(),
        value=min_t.to_pydatetime(),
        format="YYYY-MM-DD HH:mm",
        step=timedelta(hours=1),
    )
    shift = st.slider("What-if departure shift (hours)", -6, 6, 0)
    # Dots first -> default; index=0 by default
    mode = st.radio("Layer", ["Dots", "Hexagons"], horizontal=True)

sel_time = pd.to_datetime(ts) + pd.to_timedelta(int(shift), "h")
sel_time = pd.to_datetime(sel_time).tz_localize(None)
view = df[df["time"] == sel_time]

layers = []
if not view.empty:
    data = view.rename(columns={"lat": "latitude", "lon": "longitude"})

    if mode == "Hexagons":
        color_range = [
            [0, 48, 240, 40],
            [65, 105, 225, 90],
            [0, 200, 100, 140],
            [255, 215, 0, 180],
            [255, 140, 0, 220],
            [220, 20, 60, 255],
        ]
        layers.append(
            pdk.Layer(
                "HexagonLayer",
                data=data,
                get_position="[longitude, latitude]",
                radius=250,
                extruded=True,
                pickable=True,
                elevation_scale=60,
                get_elevation_weight="risk",
                elevation_aggregation="MEAN",
                get_color_weight="risk",
                color_aggregation="MEAN",
                color_range=color_range,
                elevation_range=[0, 1000],
                coverage=1,
            )
        )
    else:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=data,
                get_position="[longitude, latitude]",
                get_fill_color="[risk*255, 50, (1-risk)*255]",
                get_radius=80,
                pickable=True,
            )
        )

    # Active event pins (map overlay) if any
    ev_df = load_events()
    if ev_df is not None:
        ev_now = ev_df[(ev_df["start"] <= sel_time) & (sel_time <= ev_df["end"])]
        if not ev_now.empty:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=ev_now.rename(columns={"lat": "latitude", "lon": "longitude"}),
                    get_position="[longitude, latitude]",
                    get_radius=220,
                    get_fill_color=[255, 255, 0, 180],
                    pickable=True,
                )
            )

    # IMPORTANT: do NOT draw the route here (only after analysis)

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=float(view["lat"].mean()),
            longitude=float(view["lon"].mean()),
            zoom=10,
            pitch=40,
        ),
        tooltip={"text": "risk={risk:.2f}\nlat={latitude}\nlon={longitude}"},
    )
    col1.pydeck_chart(deck)

# Summary + tables
st.write(f"## {sel_time} — Mean risk: {view['risk'].mean():.3f}" if not view.empty else "## No data for selected time.")
# 1) Active events FIRST (always show)
st.markdown("### Active events (at selected time)")
ev_df = load_events()
if ev_df is None:
    st.dataframe(pd.DataFrame([{"name": "— none —", "start": "", "end": "", "lat": "", "lon": ""}]))
else:
    active = ev_df[(ev_df["start"] <= sel_time) & (sel_time <= ev_df["end"])][
        ["name", "start", "end", "lat", "lon"]
    ].copy()
    if active.empty:
        st.dataframe(pd.DataFrame([{"name": "— none —", "start": "", "end": "", "lat": "", "lon": ""}]))
    else:
        st.dataframe(active)

# 2) Then Top hotspots (only if we have data)
if not view.empty:
    st.markdown("### Top hotspots")
    st.dataframe(view.sort_values("risk", ascending=False).head(20)[["h3", "risk"]])

# ---------- Action insights
st.markdown("---")
st.subheader("Action insights")
if not view.empty:
    top = view.sort_values("risk", ascending=False).head(3)
    window = f"{sel_time.strftime('%a %H:%M')}"
    st.write(f"Increase readiness near top H3 cells: {', '.join(top['h3'].tolist())} around {window}.")
    if abs(shift) > 0:
        st.write(f"Departure shift {shift:+d}h applied in view; compare mean risk to choose safer window.")

# ---------- Route exposure (button -> analyze & draw)
st.markdown("---")
st.subheader("Route exposure (find safest departure)")
route_pts = load_route()
if not route_pts:
    st.caption("No route.json found. Place a polyline in `data_sample/route.json` as [[lon,lat], ...].")
else:
    if st.button("🔎 Analyze route & find safest departure"):
        res = best_route_shift(df, sel_time, route_pts, res=8, shifts=range(-6, 7))
        if res is None:
            st.warning("No valid shifts within the 72h window.")
        else:
            tbl = res["table"].set_index("shift")
            best_s = res["best_shift"]
            best_pts = res["best_points"]

            st.write(f"**Best departure shift:** {best_s:+d}h  —  mean risk = {tbl.loc[best_s, 'mean_risk']:.3f}")
            st.line_chart(tbl)

            # draw route in NAVY blue + risk-colored dots for the best shift
            path_layer = pdk.Layer(
                "PathLayer",
                data=[{"path": [[p["lon"], p["lat"]] for p in route_pts], "name": "route"}],
                get_path="path",
                get_color=[0, 0, 128],  # navy blue
                width_scale=4,
                width_min_pixels=3,
            )
            dots_layer = pdk.Layer(
                "ScatterplotLayer",
                data=best_pts.rename(columns={"lat": "latitude", "lon": "longitude"}),
                get_position="[longitude, latitude]",
                get_fill_color="[risk*255, 50, (1-risk)*255]",
                get_radius=120,
                pickable=True,
            )

            deck2 = pdk.Deck(
                layers=[path_layer, dots_layer],
                initial_view_state=pdk.ViewState(
                    latitude=float(best_pts["lat"].mean()),
                    longitude=float(best_pts["lon"].mean()),
                    zoom=10,
                    pitch=40,
                ),
                tooltip={"text": "route risk={risk:.2f}\nlat={latitude}\nlon={longitude}"},
            )
            st.pydeck_chart(deck2)

st.markdown("---")
