# -*- coding: utf-8 -*-
"""
IEEE-CIS Fraud Detection - Training-Set EDA
Phase 1: Missingness structure and preprocessing decision support.
All analysis is on the training fold only.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import load_data, make_split

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Load and split -- analysis below uses train only
# ---------------------------------------------------------------------------
df = load_data()
train, val, test = make_split(df)
print(f"\nWorking on training fold only: {train.shape[0]:,} rows x {train.shape[1]} columns")

# ---------------------------------------------------------------------------
# Relative time features (correctly labelled)
# TransactionDT is seconds from an unknown reference point.
# hour_of_day and day_of_week capture daily/weekly periodicity only --
# absolute hour and day name cannot be confirmed.
# ---------------------------------------------------------------------------
train = train.copy()
train["hour_of_day"] = (train["TransactionDT"] % 86400) // 3600       # 0-23, relative
train["day_of_week"] = (train["TransactionDT"] // 86400) % 7          # 0-6, relative

# ---------------------------------------------------------------------------
# Missingness overview
# ---------------------------------------------------------------------------
print("\n--- Missingness Overview (training fold) ---")
null_pct = train.isnull().mean() * 100
n_any    = (null_pct > 0).sum()
n_over10 = (null_pct > 10).sum()
n_over50 = (null_pct > 50).sum()
n_over90 = (null_pct > 90).sum()
n_zero   = (null_pct == 0).sum()

print(f"  Total columns:          {train.shape[1]}")
print(f"  No missing:             {n_zero}")
print(f"  Any missing:            {n_any}")
print(f"  >10% missing:           {n_over10}")
print(f"  >50% missing:           {n_over50}")
print(f"  >90% missing:           {n_over90}")

# ---------------------------------------------------------------------------
# Missingness by feature group
# ---------------------------------------------------------------------------
print("\n--- Missingness by Feature Group ---")
groups = {
    "V (Vesta engineered)": [c for c in train.columns if c.startswith("V")],
    "C (counting)":         [c for c in train.columns if c.startswith("C") and c[1:].isdigit()],
    "D (timedelta)":        [c for c in train.columns if c.startswith("D") and c[1:].isdigit()],
    "M (match flags)":      [c for c in train.columns if c.startswith("M") and c[1:].isdigit()],
    "id_ (identity)":       [c for c in train.columns if c.startswith("id_")],
    "card":                 [c for c in train.columns if c.startswith("card")],
    "addr":                 [c for c in train.columns if c.startswith("addr")],
    "dist":                 [c for c in train.columns if c.startswith("dist")],
}

group_summary = []
for name, cols in groups.items():
    if not cols:
        continue
    pcts = null_pct[cols]
    group_summary.append({
        "group":        name,
        "n_cols":       len(cols),
        "n_complete":   (pcts == 0).sum(),
        "mean_missing": pcts.mean(),
        "max_missing":  pcts.max(),
        "n_over50":     (pcts > 50).sum(),
        "n_over90":     (pcts > 90).sum(),
    })
    print(f"\n  {name} ({len(cols)} cols)")
    print(f"    Mean missing: {pcts.mean():.1f}%  |  Max: {pcts.max():.1f}%  "
          f"|  Fully present: {(pcts == 0).sum()}")
    print(f"    >50% missing: {(pcts > 50).sum()}  |  >90% missing: {(pcts > 90).sum()}")

# ---------------------------------------------------------------------------
# Missingness combinations -- are columns missing as a block?
# Strategy: create binary missing indicators, then check whether certain
# columns are always missing together.
# ---------------------------------------------------------------------------
print("\n--- Missingness Block Analysis ---")

# Identity block -- do all id_ columns follow the same pattern?
id_cols = [c for c in train.columns if c.startswith("id_")]
id_missing = train[id_cols].isnull()
id_missing_rowsum = id_missing.sum(axis=1)

n_all_id_missing   = (id_missing_rowsum == len(id_cols)).sum()
n_no_id_missing    = (id_missing_rowsum == 0).sum()
n_partial_id       = ((id_missing_rowsum > 0) & (id_missing_rowsum < len(id_cols))).sum()

print(f"\n  Identity columns ({len(id_cols)} total):")
print(f"    All id_ columns missing:    {n_all_id_missing:,} rows ({n_all_id_missing/len(train)*100:.1f}%)")
print(f"    No id_ columns missing:     {n_no_id_missing:,} rows ({n_no_id_missing/len(train)*100:.1f}%)")
print(f"    Partial id_ present:        {n_partial_id:,} rows ({n_partial_id/len(train)*100:.1f}%)")
print("    Interpretation: identity data arrives as a block -- "
      "partial presence means a few id_ columns have within-block variation")

# V column missingness distribution -- how many distinct missing patterns?
v_cols = [c for c in train.columns if c.startswith("V")]
v_missing_pct = null_pct[v_cols].sort_values()

# Bin V columns by missingness tier
bins   = [0, 0.01, 10, 50, 90, 100]
labels = ["0% (complete)", "0-10%", "10-50%", "50-90%", "90-100%"]
v_tiers = pd.cut(v_missing_pct, bins=bins, labels=labels, include_lowest=True)
print(f"\n  V column missingness tiers ({len(v_cols)} columns):")
for tier, count in v_tiers.value_counts().sort_index().items():
    print(f"    {tier:<20} {count:>3} columns")

# Check if V-column missingness clusters -- do groups of V cols have identical rates?
v_rate_counts = v_missing_pct.round(1).value_counts().sort_index()
print(f"\n  Distinct missingness rates among V columns: {len(v_rate_counts)}")
print("  Most common rates (rate -> n columns):")
for rate, count in v_rate_counts[v_rate_counts > 3].sort_values(ascending=False).items():
    print(f"    {rate:.1f}% missing -> {count} columns")

# ---------------------------------------------------------------------------
# Fraud rate by relative hour
# ---------------------------------------------------------------------------
hour_stats = train.groupby("hour_of_day")["isFraud"].agg(["sum", "count"])
hour_stats["rate"] = hour_stats["sum"] / hour_stats["count"] * 100

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
print("\nGenerating plots...")

# 1 - Distribution of missingness rates across all columns
fig1 = px.histogram(
    x=null_pct[null_pct > 0],
    nbins=50,
    labels={"x": "Missing Rate (%) per Column"},
    title="Distribution of Column Missingness Rates (training fold, columns with any missing)",
)
fig1.add_vline(x=50, line_dash="dash", line_color="red",
               annotation_text="50%", annotation_position="top right")
fig1.add_vline(x=90, line_dash="dash", line_color="darkred",
               annotation_text="90%", annotation_position="top right")
fig1.write_html(os.path.join(OUT_DIR, "eda_missingness_distribution.html"))

# 2 - Mean missingness by feature group
summary_df = pd.DataFrame(group_summary)
fig2 = px.bar(
    summary_df,
    x="group", y="mean_missing",
    text="n_cols",
    labels={"group": "Feature Group", "mean_missing": "Mean Missing Rate (%)"},
    title="Mean Missingness by Feature Group (training fold)",
)
fig2.update_traces(texttemplate="%{text} cols", textposition="outside")
fig2.update_layout(yaxis_range=[0, 100])
fig2.write_html(os.path.join(OUT_DIR, "eda_missingness_by_group.html"))

# 3 - V column missingness rates sorted
fig3 = px.bar(
    x=v_missing_pct.index,
    y=v_missing_pct.values,
    labels={"x": "V Column", "y": "Missing Rate (%)"},
    title="Missingness Rate for Each V Column (sorted, training fold)",
)
fig3.update_layout(xaxis_tickangle=-90, xaxis_showticklabels=False,
                   xaxis_title="V Columns (sorted by missing rate)")
fig3.add_hline(y=50, line_dash="dash", line_color="red")
fig3.add_hline(y=90, line_dash="dash", line_color="darkred")
fig3.write_html(os.path.join(OUT_DIR, "eda_v_missingness.html"))

# 4 - Fraud rate by relative hour of day
fig4 = px.bar(
    hour_stats.reset_index(),
    x="hour_of_day", y="rate",
    labels={"hour_of_day": "Relative Hour of Day (0-23, reference point unknown)",
            "rate": "Fraud Rate (%)"},
    title="Fraud Rate by Relative Hour of Day (training fold only)",
)
fig4.update_layout(xaxis=dict(tickmode="linear", dtick=1))
fig4.write_html(os.path.join(OUT_DIR, "eda_hour_fraud_rate_train.html"))

print("\nDone. Output files written to phase1_baseline/:")
print("  eda_missingness_distribution.html")
print("  eda_missingness_by_group.html")
print("  eda_v_missingness.html")
print("  eda_hour_fraud_rate_train.html")
