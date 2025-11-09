# CongestionAI (starter v3)

Organizer-aligned scaffold to forecast congestion risk per H3 cell×hour with weather, time, events, and static road context.
Includes: purged time CV, calibrated probabilities, map UI, and route what-if.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Generate synthetic data (Munich or Boston)
python scripts/generate_synth_data.py --city munich
# or
python scripts/generate_synth_data.py --city boston

# 2) Train baseline (LightGBM + isotonic) & save predictions + calibration plot
python scripts/train.py

# 3) Launch the map
streamlit run app/ui/streamlit_app.py
```

## Swap to real sources
- Unfallatlas → aggregate to H3×hour labels via `core/data_sources/unfallatlas.py`
- DWD (or NOAA/NWS for US) → `core/data_sources/dwd.py`
- OSM density (precomputed CSV) → `core/data_sources/osm.py`
- Holidays/events → `core/data_sources/events.py`

## What-if route exposure
```bash
python scripts/route_exposure.py --depart 2025-11-08T09:00
```
Edit `data_sample/route.json` to your polyline ([ [lon,lat], ... ]).

KPIs auto-logged to `data_sample/metrics.csv`; calibration plot at `data_sample/calibration.png`.

## Live Demo
👉 **Try it here:** [https://congestionai-demo.streamlit.app/](https://congestionai-demo.streamlit.app/)
