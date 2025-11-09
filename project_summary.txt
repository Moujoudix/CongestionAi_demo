Cities, commuters, and responders all struggle with the same question: “When and where will congestion spike next?” We built CongestionAI, a lightweight system that forecasts 72-hour congestion risk at street-grid scale and turns it into actionable decisions.

During the hackathon we delivered an end-to-end product:

A forecasting pipeline that learns from time, weather, recent conditions, and local context (events/venues) to produce well-calibrated risk probabilities for each hex cell in a city.

An interactive Streamlit map that lets anyone scrub through the next 72 hours, switch between dot and hex layers, see active events at any moment, and view top hotspots.

A one-click “Analyze route” tool that tests departure times (±6h) and visually draws the safest route window, so users can shift plans—not just stare at a heatmap.

Who benefits?
Commuters and logistics teams pick safer, faster departure times. City ops and venue managers prepare resources near upcoming hotspots. Emergency planners get a calibrated “early-warning” surface for staffing and routing.

What works today—and why it’s impressive:
The app is fast, self-contained, and privacy-friendly (ships with synthetic data but can swap to real feeds). Predictions are probabilities, not arbitrary scores, so thresholds and alerts are meaningful. The UI surfaces clear actions (“increase readiness near these cells,” “leave at +2h”), and the stack is production-ready: simple to retrain, easy to extend (holidays, incidents, live sensors).

In short: CongestionAI turns future traffic risk into decisions you can act on—in minutes, not months.
