# -*- coding: utf-8 -*-
"""
IEEE-CIS Fraud Detection - Missingness Cross-Correlation Analysis
Phase 1 EDA: Do the V-column missingness groups correlate with missingness
in other feature groups (id_, D, M, addr, dist)?

If V-group membership predicts missingness elsewhere, the transaction
sub-types implied by V structure are the same organizing principle
driving missingness across the whole dataset.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import load_data, make_split

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Load training fold only
# ---------------------------------------------------------------------------
df = load_data()
train, val, test = make_split(df)
print(f"Training fold: {train.shape[0]:,} rows")

# ---------------------------------------------------------------------------
# Step 1: Identify V column groups by missingness rate
# ---------------------------------------------------------------------------
v_cols   = [c for c in train.columns if c.startswith("V")]
v_null   = train[v_cols].isnull().mean() * 100
v_rates  = v_null.round(1)

# Group V columns by their (rounded) missingness rate
rate_groups = {}
for col, rate in v_rates.items():
    rate_groups.setdefault(rate, []).append(col)

print(f"\nV column missingness groups ({len(rate_groups)} distinct rates):")
for rate in sorted(rate_groups):
    cols = rate_groups[rate]
    print(f"  {rate:.1f}% missing -> {len(cols):3d} columns  "
          f"(representative: {cols[0]})")

# ---------------------------------------------------------------------------
# Step 2: Create binary "group present" indicators from one rep per V group
# A rep column being non-null means the whole group is present for that row.
# ---------------------------------------------------------------------------
v_group_indicators = {}
for rate in sorted(rate_groups):
    rep = rate_groups[rate][0]
    label = f"V_group_{rate:.1f}pct_missing"
    v_group_indicators[label] = train[rep].notnull().astype(int)

v_group_df = pd.DataFrame(v_group_indicators, index=train.index)

# ---------------------------------------------------------------------------
# Step 3: Create binary "is missing" indicators for other feature groups
# ---------------------------------------------------------------------------
other_indicators = {}

# Identity -- single flag covers the block
other_indicators["has_identity"] = train["id_01"].notnull().astype(int)

# D columns -- pick a spread across missingness rates
d_cols_sample = ["D1", "D2", "D3", "D4", "D5", "D7", "D10", "D11", "D14", "D15"]
for col in d_cols_sample:
    if col in train.columns:
        other_indicators[f"{col}_present"] = train[col].notnull().astype(int)

# M columns (all 9)
m_cols = [c for c in train.columns if c.startswith("M") and c[1:].isdigit()]
for col in m_cols:
    other_indicators[f"{col}_present"] = train[col].notnull().astype(int)

# addr and dist
for col in ["addr1", "addr2", "dist1", "dist2"]:
    if col in train.columns:
        other_indicators[f"{col}_present"] = train[col].notnull().astype(int)

other_df = pd.DataFrame(other_indicators, index=train.index)

# ---------------------------------------------------------------------------
# Step 4: Correlation matrix between V group indicators and other indicators
# ---------------------------------------------------------------------------
combined = pd.concat([v_group_df, other_df], axis=1)
corr = combined.corr()

# Extract the cross-section we care about: V groups vs other features
v_labels     = list(v_group_indicators.keys())
other_labels = list(other_indicators.keys())
cross_corr   = corr.loc[v_labels, other_labels]

print("\n--- Correlation: V Group Presence vs Other Feature Presence ---")
print("(Values near +1 = tend to appear together; near -1 = tend to be opposite)\n")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.2f}".format)
print(cross_corr.to_string())

# ---------------------------------------------------------------------------
# Step 5: Conditional missingness table
# For each V group, what % of rows also have identity / addr / dist present?
# More interpretable than correlation for non-statisticians.
# ---------------------------------------------------------------------------
print("\n--- Conditional Presence Rates (% of rows where OTHER feature is present) ---")
print("Rows grouped by whether each V group's representative column is present.\n")

summary_rows = []
key_others = ["has_identity", "addr1_present", "addr2_present",
              "dist1_present", "dist2_present"]

for v_label, v_series in v_group_df.items():
    row = {"V_group": v_label}
    n_present = v_series.sum()
    n_absent  = (v_series == 0).sum()
    row["n_rows_present"] = n_present
    row["n_rows_absent"]  = n_absent
    for other_label in key_others:
        o = other_df[other_label]
        pct_when_v_present = o[v_series == 1].mean() * 100
        pct_when_v_absent  = o[v_series == 0].mean() * 100
        row[f"{other_label} | V present"] = round(pct_when_v_present, 1)
        row[f"{other_label} | V absent"]  = round(pct_when_v_absent, 1)
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows).set_index("V_group")
print(summary_df.to_string())

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
print("\nGenerating plots...")

# Heatmap of cross-correlation
fig1 = px.imshow(
    cross_corr,
    color_continuous_scale="RdBu",
    zmin=-1, zmax=1,
    aspect="auto",
    title="Correlation: V Group Presence vs Other Feature Presence (training fold)",
    labels={"x": "Other Features", "y": "V Missingness Group", "color": "Correlation"},
)
fig1.update_layout(height=500)
fig1.write_html(os.path.join(OUT_DIR, "eda_missingness_correlation.html"))

# Conditional presence of identity by V group
id_rates = []
for v_label, v_series in v_group_df.items():
    rate_when_present = other_df["has_identity"][v_series == 1].mean() * 100
    rate_when_absent  = other_df["has_identity"][v_series == 0].mean() * 100
    id_rates.append({
        "V_group":        v_label.replace("V_group_", "").replace("_missing", ""),
        "V present":      rate_when_present,
        "V absent":       rate_when_absent,
    })

id_rates_df = pd.DataFrame(id_rates)
id_melt = id_rates_df.melt(id_vars="V_group", var_name="Condition",
                             value_name="% rows with identity")

fig2 = px.bar(
    id_melt,
    x="V_group", y="% rows with identity",
    color="Condition",
    barmode="group",
    title="Identity Record Presence Rate: When V Group Is Present vs Absent",
    labels={"V_group": "V Column Group", "% rows with identity": "% Rows with Identity Record"},
)
fig2.update_layout(xaxis_tickangle=-30)
fig2.write_html(os.path.join(OUT_DIR, "eda_vgroup_identity_overlap.html"))

print("\nDone. Output files written to phase1_baseline/:")
print("  eda_missingness_correlation.html")
print("  eda_vgroup_identity_overlap.html")
