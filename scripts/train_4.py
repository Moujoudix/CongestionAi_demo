# scripts/train_2.py  (drop-in)
from __future__ import annotations
import os, joblib, numpy as np, pandas as pd
from time import time
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

from core.validation import PurgedTimeSeriesSplit
from core.features import (
    add_time_features, add_weather_features, add_lags_rollups, merge_osm_density
)
from core.data_sources.events import load_events, join_events_proximity
from core.kpi import KPILogger
from core.calibration_plot import save_calibration_plot

RUN_ID = os.environ.get("RUN_ID", "baseline")

def to_naive_utc(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, utc=True)
    return s.dt.tz_convert('UTC').dt.tz_localize(None)

def load_data():
    return pd.read_csv("data_sample/history.csv", parse_dates=["time"])

def make_features(df, osm_df=None, events_df=None, compute_lags: bool = True):
    df = add_time_features(df, "time")
    df = add_weather_features(df)
    if events_df is not None:
        for col in ("start", "end"):
            if col in events_df.columns:
                events_df[col] = pd.to_datetime(events_df[col], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
        df = join_events_proximity(df, events_df, time_col="time")
    if compute_lags and "label" in df.columns:
        df = add_lags_rollups(df, group_col="h3", time_col="time", target_col="label")
    df = merge_osm_density(df, osm_df)
    df = df.dropna().reset_index(drop=True)
    return df

def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def main():
    kpi = KPILogger("data_sample/metrics.csv")

    # optional events
    events_df = None
    if os.path.exists("data_sample/events.csv"):
        try:
            events_df = load_events("data_sample/events.csv")
        except Exception as e:
            print("Warning: events.csv not loaded:", e)

    # ==== TRAINING FRAME
    df = make_features(load_data(), events_df=events_df)
    df["time"] = to_naive_utc(df["time"])

    # --- add geo prior BEFORE selecting feats, and keep same prior for future
    hist_prior = (
        df.groupby("h3", as_index=False)["label"]
          .mean()
          .rename(columns={"label": "hex_prior"})
    )
    df = df.merge(hist_prior, on="h3", how="left")

    # now select features (hex_prior included)
    num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    feats = [c for c in num_cols if c != "label"]
    Xdf, y = df[feats], df["label"].values

    splitter = PurgedTimeSeriesSplit(n_splits=4, purge_hours=12)

    final_cfg = dict(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=256,
        min_child_samples=40,
        max_bin=511,
        feature_fraction=0.9,
        bagging_fraction=0.85,
        bagging_freq=1,
        reg_lambda=1.0,
        random_state=777,
        verbose=-1,
    )

    # ---- OOF (aligned with final)
    oof = np.zeros(len(df), dtype=float)
    oof_logits = np.zeros(len(df), dtype=float)
    fold_models = []

    for i, (tr, te) in enumerate(splitter.split(df)):
        if len(tr) == 0 or len(np.unique(y[tr])) < 2:
            print(f"[fold {i}] skipped (train size={len(tr)})")
            continue
        base = LGBMClassifier(**{**final_cfg, "random_state": 42 + i})
        base.fit(Xdf.iloc[tr], y[tr])
        proba = base.predict_proba(Xdf.iloc[te])[:, 1]
        oof[te] = proba
        oof_logits[te] = logit(proba)
        kpi.log(RUN_ID, f"fold_{i}_auc", float(roc_auc_score(y[te], proba)), {"n": int(len(te))})
        fold_models.append(base)

    if not fold_models:
        base_all = LGBMClassifier(**final_cfg)
        base_all.fit(Xdf, y)
        proba_all = base_all.predict_proba(Xdf)[:, 1]
        oof[:] = proba_all
        oof_logits[:] = logit(proba_all)
        fold_models = [base_all]

    # ---- KPIs + calibration (on OOF logits)
    auc = roc_auc_score(y, oof); brier = brier_score_loss(y, oof)
    kpi.log(RUN_ID, "oof_auc", float(auc), {}); kpi.log(RUN_ID, "oof_brier", float(brier), {})
    save_calibration_plot(y, oof, "data_sample/calibration.png")
    cal = LogisticRegression(max_iter=50_000, C=5.0, random_state=123)
    cal.fit(oof_logits.reshape(-1, 1), y)

    # ---- FINAL model
    final = LGBMClassifier(**final_cfg)
    final.fit(Xdf, y)

    os.makedirs("data_sample/models", exist_ok=True)
    joblib.dump(final, "data_sample/models/final_lgbm.joblib")
    joblib.dump(cal,   "data_sample/models/cal_logreg.joblib")
    for i, m in enumerate(fold_models):
        joblib.dump(m, f"data_sample/models/fold_{i}.joblib")

    # ==== INFERENCE ON FUTURE 72h
    fut = pd.read_csv("data_sample/forecast.csv", parse_dates=["time"])
    fut["time"] = to_naive_utc(fut["time"])
    fut_feat = make_features(fut.assign(label=0), events_df=events_df, compute_lags=False)
    fut_feat["time"] = to_naive_utc(fut_feat["time"])

    # bring the SAME hex_prior to future
    fut_feat = fut_feat.merge(hist_prior, on="h3", how="left")
    # fill inference-time holes with a sensible prior instead of zeros
    global_prior = float(df["label"].mean())
    fut_feat["hex_prior"] = fut_feat["hex_prior"].astype("float64").fillna(global_prior)
    # time join hygiene
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    fut_feat["time"] = pd.to_datetime(fut_feat["time"]).dt.tz_localize(None)

    # build lags for future from history (t+1 has signal; later horizons mostly zeros)
    hist_min = df[["h3", "time", "label"]].copy()
    future_stub = fut[["h3", "time"]].copy(); future_stub["label"] = np.nan
    combo = pd.concat([hist_min, future_stub], ignore_index=True).sort_values(["h3", "time"])
    g = combo.groupby("h3", sort=False)
    combo["lag_1"] = g["label"].shift(1)
    combo["lag_2"] = g["label"].shift(2)
    combo["lag_3"] = g["label"].shift(3)
    combo["roll_mean_3"] = g["label"].shift(1).rolling(3).mean().reset_index(level=0, drop=True)

    lags_future = combo.loc[
        combo["time"].isin(fut_feat["time"]),
        ["h3", "time", "lag_1", "lag_2", "lag_3", "roll_mean_3"]
    ]
    fut_feat = fut_feat.merge(lags_future, on=["h3", "time"], how="left", validate="m:1")

    # fill any lag holes with global prior (so later horizons aren’t all zeros)
    prior = float(df["label"].mean())
    for c in ["lag_1", "lag_2", "lag_3", "roll_mean_3", "hex_prior"]:
        if c not in fut_feat.columns:
            fut_feat[c] = prior
        fut_feat[c] = fut_feat[c].astype("float64").fillna(prior)

    # PREDICT: average folds → logit → calibrator → soft temperature
    num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    feats = [c for c in num_cols if c != "label"]  # hex_prior will be in num_cols now
    Xdf, y = df[feats], df["label"].values
    Xf_df = fut_feat[feats]
    Xf_df = fut_feat[feats]  # feats contains hex_prior now
    raw_folds = np.column_stack([m.predict_proba(Xf_df)[:, 1] for m in fold_models])
    raw = raw_folds.mean(axis=1)
    logits = logit(raw).reshape(-1, 1)
    cal = joblib.load("data_sample/models/cal_logreg.joblib")
    preds = cal.predict_proba(logits)[:, 1]

    def soften(p, T=1.30, eps=1e-6):
        lg = logit(p, eps)
        return 1.0 / (1.0 + np.exp(-lg / T))
    preds = soften(preds, T=1.30)

    out = fut_feat[["h3", "lat", "lon", "time"]].copy()
    out["risk"] = preds
    n_per_t = out.groupby("time")["risk"].transform("size")
    rank = out.groupby("time")["risk"].rank(method="first")
    out["risk_idx"] = (rank - 0.5) / n_per_t
    out.to_csv("data_sample/predictions.csv", index=False)

    print("OOF AUC=", round(auc, 4), "Brier=", round(brier, 4))
    assert len(out) == len(fut_feat)
    assert out["time"].nunique() == 72

if __name__ == "__main__":
    t0 = time(); print("Start…"); main(); print(f"Done in {time()-t0:.2f}s")
