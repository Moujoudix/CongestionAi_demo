# CongestionAI — 72h Congestion Risk (demo)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([YOUR_STREAMLIT_URL](https://congestionai-demo.streamlit.app/))

Cities, commuters, and responders all struggle with the same question: **“When and where will congestion spike next?”**
**CongestionAI** forecasts **72-hour congestion risk** at street-grid scale and turns it into **actionable decisions**.

### What’s inside
- **Forecasting pipeline** — learns from time, weather, recent conditions, and local context (events/venues) to produce **well-calibrated risk probabilities** per H3 cell.
- **Interactive map (Streamlit)** — scrub through the next 72 hours, switch **Dots/Hex** layers, see **active events** at any moment, and view **top hotspots**.
- **One-click “Analyze route”** — tests departure times (±6h) and visualizes the **safest window** so users can **shift plans**, not just stare at a heatmap.

### Who benefits
- **Commuters & logistics** — pick safer, faster departure times.
- **City ops & venue managers** — stage resources near upcoming hotspots.
- **Emergency planning** — calibrated early-warning surface for staffing & routing.

### Why this is compelling
- **Fast & privacy-friendly** (ships with synthetic data, easily swaps to real feeds).
- **Probabilities, not scores** — thresholds/alerts are meaningful.
- **Actionable UI** — “increase readiness near these cells,” “leave at +2h.”
- **Production-ready stack** — simple to retrain, easy to extend (holidays, incidents, live sensors).

> In short: **CongestionAI turns future traffic risk into decisions you can act on — in minutes, not months.**
