from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
from time import time

from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss


from core.validation import PurgedTimeSeriesSplit
from core.features import (
    add_time_features,
    add_weather_features,
    add_lags_rollups,
    merge_osm_density,
)
from core.data_sources.events import load_events, join_events_proximity
from core.kpi import KPILogger
from core.calibration_plot import save_calibration_plot

RUN_ID = os.environ.get("RUN_ID", "baseline")


def load_data():
    return pd.read_csv("data_sample/history.csv", parse_dates=["time"])


def make_features(df, osm_df=None, events_df=None, compute_lags: bool = True):
    df = add_time_features(df, "time")
    df = add_weather_features(df)
    if events_df is not None:
        df = join_events_proximity(df, events_df, time_col="time")
    if compute_lags and "label" in df.columns:
        df = add_lags_rollups(df, group_col="h3", time_col="time", target_col="label")
    df = merge_osm_density(df, osm_df)
    df = df.dropna().reset_index(drop=True)
    return df

def to_naive_utc(s: pd.Series) -> pd.Series:
    # Works whether s is naive or tz-aware; returns tz-naive UTC
    s = pd.to_datetime(s, utc=True)
    return s.dt.tz_convert('UTC').dt.tz_localize(None)


def main():
    kpi = KPILogger("data_sample/metrics.csv")

    # optional events
    events_df = None
    if os.path.exists("data_sample/events.csv"):
        try:
            events_df = load_events("data_sample/events.csv")
        except Exception as e:
            print("Warning: events.csv not loaded:", e)

    # --- training frame
    df = make_features(load_data(), events_df=events_df)
    df["time"] = to_naive_utc(df["time"])
    splitter = PurgedTimeSeriesSplit(n_splits=4, purge_hours=12)

    # numeric features only (avoid strings like h3)
    num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    feats = [c for c in num_cols if c != "label"]

    Xdf = df[feats]                  # keep as DataFrame to preserve column names
    y = df["label"].values

    oof = np.zeros(len(df), dtype=float)
    models = []
    oof_base = np.zeros(len(df), dtype=float)  # raw LGBM OOF for LOGIT calibration

    for i, (tr, te) in enumerate(splitter.split(df)):
        if len(tr) == 0 or len(np.unique(y[tr])) < 2:
            print(f"[fold {i}] skipped (train size={len(tr)})")
            continue

        # ---- raw base for OOF calibration data
        base = LGBMClassifier(
            n_estimators=300, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42 + i, verbose=-1
        )
        base.fit(Xdf.iloc[tr], y[tr])
        proba_base = base.predict_proba(Xdf.iloc[te])[:, 1]
        oof_base[te] = proba_base

        # ---- (optional) calibrated model for your fold KPIs (leave as you had)
        base_for_cal = LGBMClassifier(
            n_estimators=1_500, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=50, num_leaves=64,
            class_weight="balanced",
            random_state=42 + i, verbose=-1
        )
        clf = CalibratedClassifierCV(base_for_cal, method="isotonic", cv=3)
        clf.fit(Xdf.iloc[tr], y[tr])
        proba = clf.predict_proba(Xdf.iloc[te])[:, 1]

        oof[te] = proba
        kpi.log(RUN_ID, f"fold_{i}_auc", float(roc_auc_score(y[te], proba)), {"n": int(len(te))})
        models.append(clf)

    # Fallback if all folds were skipped
    if len(models) == 0:
        print("[fallback] No valid CV folds; fitting single calibrated model on all data.")
        base_all = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        clf_all = CalibratedClassifierCV(base_all, method="isotonic", cv=3)
        clf_all.fit(Xdf, y)
        models = [clf_all]
        oof = clf_all.predict_proba(Xdf)[:, 1]

    # KPIs + calibration
    auc = roc_auc_score(y, oof)
    brier = brier_score_loss(y, oof)
    kpi.log(RUN_ID, "oof_auc", float(auc), {})
    kpi.log(RUN_ID, "oof_brier", float(brier), {})
    save_calibration_plot(y, oof, "data_sample/calibration.png")

    """# Fit an isotonic calibrator on OOF to map scores -> calibrated prob
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof, y)
    joblib.dump(iso, "data_sample/models/iso_oof.joblib")"""

    # Logistic calibration trained on **LOGITS of raw LGBM OOF** (smoother, no plateaus)
    oofb = np.clip(oof_base, 1e-6, 1-1e-6)
    oofb_logit = np.log(oofb/(1-oofb)).reshape(-1, 1)
    cal = LogisticRegression(max_iter=20_000, C=10.0)
    cal.fit(oofb_logit, y)
    joblib.dump(cal, "data_sample/models/cal_logreg.joblib")

    # ---- final model for inference (no calibrator; trained on all data)
    final = LGBMClassifier(
        n_estimators=1_500, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_samples=50, num_leaves=64,
        class_weight="balanced",
        random_state=777, verbose=-1
    )
    final.fit(Xdf, y)
    os.makedirs("data_sample/models", exist_ok=True)
    joblib.dump(final, "data_sample/models/final_lgbm.joblib")

    # save fold models
    os.makedirs("data_sample/models", exist_ok=True)
    for i, m in enumerate(models):
        joblib.dump(m, f"data_sample/models/model_{i}.joblib")

    # --- inference on future 72h
    fut = pd.read_csv("data_sample/forecast.csv", parse_dates=["time"])
    fut["time"] = to_naive_utc(fut["time"])

    # build base features *without* lags for the future
    fut_feat = make_features(fut.assign(label=0), events_df=events_df, compute_lags=False)
    fut_feat["time"] = to_naive_utc(fut_feat["time"])

    # make both sides tz-naive for safe joins
    fut_feat["time"] = pd.to_datetime(fut_feat["time"]).dt.tz_localize(None)
    df["time"]       = pd.to_datetime(df["time"]).dt.tz_localize(None)

    # build lag features for future using history labels
    hist_min    = df[["h3", "time", "label"]].copy()
    hist_min["time"] = to_naive_utc(hist_min["time"])
    future_stub = fut[["h3", "time"]].copy()
    future_stub["time"] = to_naive_utc(future_stub["time"])
    future_stub["label"] = np.nan

    combo = (
        pd.concat([hist_min, future_stub], ignore_index=True)
        .sort_values(["h3", "time"])
    )
    combo["time"] = to_naive_utc(combo["time"])
    # In-place lag calc that preserves future rows
    g = combo.groupby("h3", sort=False)
    combo["lag_1"] = g["label"].shift(1)
    combo["lag_2"] = g["label"].shift(2)
    combo["lag_3"] = g["label"].shift(3)
    # rolling mean of the *previous* 3 labels (exclude current by shifting first)
    roll_base = g["label"].shift(1)
    combo["roll_mean_3"] = (
        roll_base.rolling(3).mean().reset_index(level=0, drop=True)
    )

    # Now pick only future rows (same times as fut_feat) and merge
    lags_future = combo.loc[
        combo["time"].isin(fut_feat["time"]),
        ["h3", "time", "lag_1", "lag_2", "lag_3", "roll_mean_3"]
    ]
    fut_feat = fut_feat.merge(lags_future, on=["h3","time"], how="left", validate="m:1")

    # ensure the lag columns exist and are numeric
    for c in ["lag_1","lag_2","lag_3","roll_mean_3"]:
        if c not in fut_feat.columns:
            fut_feat[c] = 0.0
        fut_feat[c] = fut_feat[c].astype("float64").fillna(0.0)

     # DEBUG: write lag coverage snapshot for sanity checks
    fut_feat[["h3","time","lag_1","lag_2","lag_3","roll_mean_3"]].to_csv(
        "data_sample/fut_lags_debug.csv", index=False
    )

    # predict with the same feature set as training
    Xf_df = fut_feat[feats]
    try:
        final
    except NameError:
        final = joblib.load("data_sample/models/final_lgbm.joblib")
    preds = final.predict_proba(Xf_df)[:, 1]

    # ---- debug: are raw final preds two-valued?
    preds_raw = final.predict_proba(Xf_df)[:, 1]
    print("raw preds unique:", np.unique(np.round(preds_raw, 6)).shape[0])

    # quick feature importance peek
    fi = pd.Series(final.feature_importances_, index=Xf_df.columns).sort_values(ascending=False)
    print("TOP 8 FI:\n", fi.head(8).to_string())

    # how much does lag_1 drive it?
    if "lag_1" in Xf_df.columns:
        c = np.corrcoef(preds_raw, Xf_df["lag_1"].values)[0,1]
        print("corr(preds_raw, lag_1) =", round(float(c), 4))


    """# apply OOF-fitted isotonic to the final model's probs (stabilizes extremes)
    try:
        iso = joblib.load("data_sample/models/iso_oof.joblib")
        preds = iso.transform(preds)
    except Exception as e:
        print("Warning: isotonic calibrator not found/usable:", e)"""

    # Apply logistic calibrator on **logits**
    preds_raw = final.predict_proba(Xf_df)[:, 1]
    preds_safe  = np.clip(preds_raw, 1e-6, 1-1e-6)
    preds_logit = np.log(preds_safe/(1-preds_safe)).reshape(-1, 1)
    try:
        cal = joblib.load("data_sample/models/cal_logreg.joblib")
        preds = cal.predict_proba(preds_logit)[:, 1]
    except Exception as e:
        print("Warning: logistic calibrator not found/usable:", e)
        preds = preds_safe

    # Optional : gentle temperature smoothing
    def soften_probs(p, T=1.3, eps=1e-6):
        p = np.clip(p, eps, 1-eps)
        lg = np.log(p/(1-p))
        return 1/(1+np.exp(-lg/T))
    preds = soften_probs(preds, T=1.3)

    out = fut_feat[["h3", "lat", "lon", "time"]].copy()
    out["risk"] = preds
    # Per-hour monotone rank 0..1, robust to ties - great for visualization & “Top hotspots”
    n_per_t = out.groupby("time")["risk"].transform("size")
    rank = out.groupby("time")["risk"].rank(method="first")
    out["risk_idx"] = (rank - 0.5) / n_per_t  # (0,1) roughly uniform each hour
    out.to_csv("data_sample/predictions.csv", index=False)

    print("Training done. OOF AUC=", round(auc, 4), "Brier=", round(brier, 4))
    print("Wrote predictions.csv & calibration.png")


if __name__ == "__main__":
    start = time()
    print("Start of the training process...\n")
    main()
    print(f"Execution time : {time()-start:.2f}s")
