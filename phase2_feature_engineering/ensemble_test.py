# -*- coding: utf-8 -*-
"""
Does blending LightGBM and XGBoost's production predictions beat either
model alone? Motivated by model_divergence_analysis.py: the two algorithms
agree on 67-68% of fraud cases but diverge meaningfully elsewhere (Spearman
rank correlation ~0.75, not ~1.0), and XGBoost has more model-exclusive
catches than LightGBM. Divergence alone doesn't guarantee an ensemble helps
-- this tests it directly on val, the same way every other design choice in
this project has been tested rather than assumed.

Each algorithm keeps its own production routing (LightGBM: global model for
all rows; XGBoost: hybrid routing for has_identity=1) -- this script blends
the two algorithms' final outputs, not their internal segment routing.

Blends compared:
  - Raw probability average (assumes the two algorithms' probabilities are
    on a roughly comparable scale -- calibration check in inference.py
    showed both in the 0.75-0.91x mean-pred/actual-rate range, close enough
    to be defensible, not identical)
  - Rank-percentile average (robust to any residual scale mismatch --
    blends relative ranking instead of raw probability)
  - Weighted average grid search over both blends, swept 0.0-1.0 in steps
    of 0.05, weight on XGBoost (the stronger standalone model)
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import load_data, make_split
from phase1_baseline.pipeline import evaluate
from phase2_feature_engineering.inference import build_features, HybridModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_COLOR_RAW = "#2a78d6"
_COLOR_RANK = "#1baf7a"
_COLOR_BASELINE = "#898781"
_COLOR_BEST = "#d03b3b"


def main():
    print("=== Ensemble test: does blending LightGBM + XGBoost beat either alone? ===\n")

    df = load_data()
    df = build_features(df)
    train_raw, val_raw, test_raw = make_split(df)
    del train_raw, test_raw

    model = HybridModel()
    lgb_prob = model.predict(val_raw, algorithm="lgb")
    xgb_prob = model.predict(val_raw, algorithm="xgb")
    y_val = val_raw["isFraud"].values

    print("--- Standalone (production routing, as shipped) ---")
    lgb_metrics = evaluate(y_val, lgb_prob.values, "LightGBM alone")
    xgb_metrics = evaluate(y_val, xgb_prob.values, "XGBoost alone")

    lgb_rank = lgb_prob.rank(pct=True)
    xgb_rank = xgb_prob.rank(pct=True)

    print("\n--- Fixed 50/50 blends ---")
    raw_avg = (lgb_prob + xgb_prob) / 2
    rank_avg = (lgb_rank + xgb_rank) / 2
    evaluate(y_val, raw_avg.values, "50/50 raw probability average")
    evaluate(y_val, rank_avg.values, "50/50 rank-percentile average")

    print("\n--- Weighted average grid search (weight = share on XGBoost) ---")
    print(f"{'Weight (XGB)':>12}  {'Raw-avg PR-AUC':>15}  {'Rank-avg PR-AUC':>16}")
    best_raw = (-1, None)
    best_rank = (-1, None)
    sweep = []
    for w in np.arange(0.0, 1.01, 0.05):
        raw_blend = w * xgb_prob + (1 - w) * lgb_prob
        rank_blend = w * xgb_rank + (1 - w) * lgb_rank
        raw_pr = average_precision_score(y_val, raw_blend.values)
        rank_pr = average_precision_score(y_val, rank_blend.values)
        print(f"{w:>12.2f}  {raw_pr:>15.4f}  {rank_pr:>16.4f}")
        sweep.append({"weight_xgb": round(float(w), 2), "raw_avg_pr_auc": raw_pr, "rank_avg_pr_auc": rank_pr})
        if raw_pr > best_raw[0]:
            best_raw = (raw_pr, w)
        if rank_pr > best_rank[0]:
            best_rank = (rank_pr, w)

    print(f"\nBest raw-avg blend:  weight(XGB)={best_raw[1]:.2f}  PR-AUC={best_raw[0]:.4f}")
    print(f"Best rank-avg blend: weight(XGB)={best_rank[1]:.2f}  PR-AUC={best_rank[0]:.4f}")

    print(f"\n--- Summary vs. standalone XGBoost (the stronger single model) ---")
    print(f"  XGBoost alone:            PR-AUC={xgb_metrics['pr_auc']:.4f}")
    print(f"  LightGBM alone:           PR-AUC={lgb_metrics['pr_auc']:.4f}")
    print(f"  Best weighted raw blend:  PR-AUC={best_raw[0]:.4f}  "
          f"(delta vs XGB alone: {best_raw[0]-xgb_metrics['pr_auc']:+.4f})")
    print(f"  Best weighted rank blend: PR-AUC={best_rank[0]:.4f}  "
          f"(delta vs XGB alone: {best_rank[0]-xgb_metrics['pr_auc']:+.4f})")
    print("\nTest set NOT evaluated here -- touch it exactly once at final reporting.")

    weights = [r["weight_xgb"] for r in sweep]
    raw_prs = [r["raw_avg_pr_auc"] for r in sweep]
    rank_prs = [r["rank_avg_pr_auc"] for r in sweep]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weights, y=raw_prs, mode="lines+markers", name="Raw probability average",
        line=dict(color=_COLOR_RAW, width=2), marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=weights, y=rank_prs, mode="lines+markers", name="Rank-percentile average",
        line=dict(color=_COLOR_RANK, width=2), marker=dict(size=6),
    ))
    fig.add_hline(y=lgb_metrics["pr_auc"], line=dict(color=_COLOR_BASELINE, width=1, dash="dot"),
                  annotation_text="LightGBM alone", annotation_position="bottom right")
    fig.add_hline(y=xgb_metrics["pr_auc"], line=dict(color=_COLOR_BASELINE, width=1, dash="dot"),
                  annotation_text="XGBoost alone", annotation_position="top right")
    fig.add_trace(go.Scatter(
        x=[best_raw[1]], y=[best_raw[0]], mode="markers", name=f"Best blend (w={best_raw[1]:.2f})",
        marker=dict(size=13, color=_COLOR_BEST, symbol="star"),
    ))
    fig.update_layout(
        title="Ensemble weight sweep: val PR-AUC vs. weight on XGBoost",
        xaxis_title="Weight on XGBoost (1 - weight = share on LightGBM)",
        yaxis_title="Val PR-AUC",
        xaxis=dict(range=[0, 1], gridcolor="#e1e0d9"),
        yaxis=dict(gridcolor="#e1e0d9"),
        plot_bgcolor="#fcfcfb",
        legend=dict(title="Blend method"),
        height=550,
        width=800,
    )
    out_html = os.path.join(SCRIPT_DIR, "ensemble_weight_sweep.html")
    fig.write_html(out_html)
    print(f"\nSaved: {out_html}")

    out_json = os.path.join(SCRIPT_DIR, "ensemble_weight_sweep.json")
    with open(out_json, "w") as f:
        json.dump({
            "sweep": sweep,
            "lgb_alone_pr_auc": lgb_metrics["pr_auc"],
            "xgb_alone_pr_auc": xgb_metrics["pr_auc"],
            "best_raw_blend": {"weight_xgb": best_raw[1], "pr_auc": best_raw[0]},
            "best_rank_blend": {"weight_xgb": best_rank[1], "pr_auc": best_rank[0]},
        }, f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
