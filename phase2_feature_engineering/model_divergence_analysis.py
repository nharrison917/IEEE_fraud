# -*- coding: utf-8 -*-
"""
Where do LightGBM and XGBoost notably disagree, on the actual production
models (algorithm-specific routing from inference.py), not a fresh ablation
retrain? error_analysis.py already showed each model's hardest-quartile
misses overlap 66.5% -- this looks at the other third directly: which
fraud (and non-fraud) rows get a high risk score from one production model
and a low one from the other, and what those rows look like.

Both models' raw probabilities are compared by percentile rank within the
full val set (not raw probability), since the two algorithms sit on
different scales after algorithm-specific routing (XGBoost hybrid,
LightGBM global-only -- see PLAN.md "Inference pipeline"). Rank comparison
avoids treating a calibration difference as a disagreement.
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import load_data, make_split
from phase2_feature_engineering.inference import build_features, HybridModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_CONTEXT_COLS = [
    "TransactionAmt", "ProductCD", "has_identity", "DeviceType",
    "card4", "card6", "hour_of_day", "is_round_amount",
    "combo_prior_txn_count", "addr1_changed_from_prev",
    "addr1_change_x_inverse_time", "amt_ratio_vs_prev",
]

_CATCH_THRESH = 0.90  # top decile by rank = "confidently flagged"
_MISS_THRESH = 0.50   # bottom half by rank = "not flagged"

_COLOR_BOTH = "#0ca30c"
_COLOR_NEITHER = "#d03b3b"
_COLOR_LGB_ONLY = "#2a78d6"
_COLOR_XGB_ONLY = "#eb6834"


def describe_group(df, label):
    print(f"\n  --- {label} (n={len(df)}) ---")
    if len(df) == 0:
        return
    print(f"  TransactionAmt: mean={df['TransactionAmt'].mean():.2f} "
          f"median={df['TransactionAmt'].median():.2f}")
    print(f"  has_identity rate: {df['has_identity'].mean():.3f}")
    print(f"  is_round_amount rate: {df['is_round_amount'].mean():.3f}")
    print("  ProductCD distribution:")
    print(df["ProductCD"].value_counts(normalize=True).round(3).to_string())
    print(f"  combo_prior_txn_count: mean={df['combo_prior_txn_count'].mean():.1f} "
          f"median={df['combo_prior_txn_count'].median():.1f}")
    print(f"  addr1_changed_from_prev rate: {df['addr1_changed_from_prev'].mean():.3f}")


def main():
    print("=== Model divergence analysis: production LightGBM vs. XGBoost, val fold ===\n")

    df = load_data()
    df = build_features(df)
    train_raw, val_raw, test_raw = make_split(df)
    del train_raw, test_raw

    model = HybridModel()
    lgb_prob = model.predict(val_raw, algorithm="lgb")
    xgb_prob = model.predict(val_raw, algorithm="xgb")

    context = val_raw[_CONTEXT_COLS + ["isFraud"]].copy()
    context["lgb_prob"] = lgb_prob
    context["xgb_prob"] = xgb_prob
    context["lgb_rank_pct"] = context["lgb_prob"].rank(pct=True)
    context["xgb_rank_pct"] = context["xgb_prob"].rank(pct=True)
    context["rank_diff"] = context["lgb_rank_pct"] - context["xgb_rank_pct"]

    corr_df = context[["lgb_prob", "xgb_prob"]]
    spearman_corr = corr_df.corr(method="spearman").iloc[0, 1]
    print(f"Spearman rank correlation (full val set, both algorithms): {spearman_corr:.4f}")

    fraud = context[context["isFraud"] == 1].copy()
    print(f"Total fraud cases in val fold: {len(fraud)}")

    lgb_catches = fraud["lgb_rank_pct"] >= _CATCH_THRESH
    xgb_catches = fraud["xgb_rank_pct"] >= _CATCH_THRESH
    lgb_misses = fraud["lgb_rank_pct"] < _MISS_THRESH
    xgb_misses = fraud["xgb_rank_pct"] < _MISS_THRESH

    both_catch = fraud[lgb_catches & xgb_catches]
    neither_catch = fraud[lgb_misses & xgb_misses]
    lgb_only = fraud[lgb_catches & xgb_misses]
    xgb_only = fraud[xgb_catches & lgb_misses]

    print(f"\n{'='*70}\nAgreement breakdown (top-decile catch vs. bottom-half miss, "
          f"among {len(fraud)} fraud cases)\n{'='*70}")
    print(f"  Both catch:                {len(both_catch):>5}  "
          f"({len(both_catch)/len(fraud)*100:.1f}%)")
    print(f"  Neither catches:           {len(neither_catch):>5}  "
          f"({len(neither_catch)/len(fraud)*100:.1f}%)")
    print(f"  LightGBM only catches:     {len(lgb_only):>5}  "
          f"({len(lgb_only)/len(fraud)*100:.1f}%)")
    print(f"  XGBoost only catches:      {len(xgb_only):>5}  "
          f"({len(xgb_only)/len(fraud)*100:.1f}%)")

    describe_group(both_catch, "Both models catch")
    describe_group(neither_catch, "Neither model catches (shared blind spot)")
    describe_group(lgb_only, "LightGBM catches, XGBoost misses")
    describe_group(xgb_only, "XGBoost catches, LightGBM misses")

    print(f"\n{'='*70}\nTop 15 most divergent fraud cases (by |rank_diff|)\n{'='*70}")
    top_divergent = fraud.reindex(fraud["rank_diff"].abs().sort_values(ascending=False).index).head(15)
    print(top_divergent[["TransactionAmt", "ProductCD", "has_identity", "lgb_rank_pct",
                          "xgb_rank_pct", "rank_diff"]].to_string())

    print(f"\n{'='*70}\nFalse-positive angle: legitimate transactions scored very differently\n{'='*70}")
    legit = context[context["isFraud"] == 0].copy()
    top_legit_divergent = legit.reindex(
        legit["rank_diff"].abs().sort_values(ascending=False).index
    ).head(10)
    print(top_legit_divergent[["TransactionAmt", "ProductCD", "has_identity", "lgb_rank_pct",
                                "xgb_rank_pct", "rank_diff"]].to_string())

    # --- Chart: fraud cases by rank percentile in each model, colored by
    # which model(s) caught it at the top-decile threshold. ---
    context.loc[both_catch.index, "_group"] = "Both catch"
    context.loc[neither_catch.index, "_group"] = "Neither catches"
    context.loc[lgb_only.index, "_group"] = "LightGBM only"
    context.loc[xgb_only.index, "_group"] = "XGBoost only"
    fraud_plot = fraud.copy()
    fraud_plot["_group"] = context.loc[fraud_plot.index, "_group"]
    fraud_plot["_group"] = fraud_plot["_group"].fillna("Mixed (below top decile for both)")

    group_colors = {
        "Both catch": _COLOR_BOTH,
        "Neither catches": _COLOR_NEITHER,
        "LightGBM only": _COLOR_LGB_ONLY,
        "XGBoost only": _COLOR_XGB_ONLY,
        "Mixed (below top decile for both)": "#898781",
    }

    fig = go.Figure()
    for group_name, color in group_colors.items():
        sub = fraud_plot[fraud_plot["_group"] == group_name]
        if len(sub) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=sub["lgb_rank_pct"], y=sub["xgb_rank_pct"],
            mode="markers",
            name=f"{group_name} (n={len(sub)})",
            marker=dict(color=color, size=8, opacity=0.7, line=dict(width=0)),
        ))

    fig.add_shape(type="line", x0=_CATCH_THRESH, x1=_CATCH_THRESH, y0=0, y1=1,
                  line=dict(color="#c3c2b7", width=1, dash="dot"))
    fig.add_shape(type="line", x0=0, x1=1, y0=_CATCH_THRESH, y1=_CATCH_THRESH,
                  line=dict(color="#c3c2b7", width=1, dash="dot"))

    fig.update_layout(
        title="Fraud cases: LightGBM vs. XGBoost risk-rank percentile (val fold)",
        xaxis_title="LightGBM rank percentile (higher = riskier)",
        yaxis_title="XGBoost rank percentile (higher = riskier)",
        xaxis=dict(range=[0, 1], gridcolor="#e1e0d9"),
        yaxis=dict(range=[0, 1], gridcolor="#e1e0d9"),
        plot_bgcolor="#fcfcfb",
        legend=dict(title="Catch status (top-decile threshold)"),
        height=650,
        width=750,
    )
    out_html = os.path.join(SCRIPT_DIR, "model_divergence.html")
    fig.write_html(out_html)
    print(f"\nSaved: {out_html}")

    out_json = os.path.join(SCRIPT_DIR, "model_divergence.json")
    summary = {
        "spearman_corr_full_val": float(spearman_corr),
        "n_fraud": int(len(fraud)),
        "both_catch": int(len(both_catch)),
        "neither_catch": int(len(neither_catch)),
        "lgb_only": int(len(lgb_only)),
        "xgb_only": int(len(xgb_only)),
        "top_divergent_fraud": top_divergent[
            ["TransactionAmt", "ProductCD", "has_identity", "lgb_rank_pct", "xgb_rank_pct", "rank_diff"]
        ].reset_index().to_dict(orient="records"),
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
