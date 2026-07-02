# -*- coding: utf-8 -*-
"""
IEEE-CIS Fraud Detection - Exploratory Data Analysis
Phase 1: Initial data sanity checks and feature landscape overview
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUT_DIR = SCRIPT_DIR

TRAIN_TXN = os.path.join(DATA_DIR, "train_transaction.csv")
TRAIN_ID  = os.path.join(DATA_DIR, "train_identity.csv")

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print("Loading transaction data...")
txn = pd.read_csv(TRAIN_TXN)
print(f"  train_transaction: {txn.shape[0]:,} rows x {txn.shape[1]} cols")

print("Loading identity data...")
idf = pd.read_csv(TRAIN_ID)
print(f"  train_identity:    {idf.shape[0]:,} rows x {idf.shape[1]} cols")

print("Joining on TransactionID (left join - not all transactions have identity data)...")
df = txn.merge(idf, on="TransactionID", how="left")
print(f"  Combined:          {df.shape[0]:,} rows x {df.shape[1]} cols")
identity_match_pct = idf.shape[0] / txn.shape[0] * 100
print(f"  {identity_match_pct:.1f}% of transactions have identity records")

# ---------------------------------------------------------------------------
# Fraud rate
# ---------------------------------------------------------------------------
print("\n--- Fraud Rate ---")
fraud_counts = df["isFraud"].value_counts()
fraud_rate = df["isFraud"].mean() * 100
print(f"  Not fraud: {fraud_counts[0]:,}")
print(f"  Fraud:     {fraud_counts[1]:,}")
print(f"  Fraud rate: {fraud_rate:.2f}%")

# ---------------------------------------------------------------------------
# TransactionDT - convert to usable time
# TransactionDT is seconds elapsed from an undisclosed reference point.
# The competition start date is commonly used to anchor it; here we treat
# it as a relative offset only - the absolute date is unknown and irrelevant
# for feature engineering purposes.
# ---------------------------------------------------------------------------
print("\n--- Time Range ---")
dt_min = df["TransactionDT"].min()
dt_max = df["TransactionDT"].max()
span_days = (dt_max - dt_min) / 86400
print(f"  TransactionDT range: {dt_min:,} to {dt_max:,} seconds")
print(f"  Span: {span_days:.1f} days (~{span_days/30:.1f} months)")

# Derive relative time features for inspection
df["day_of_week"] = (df["TransactionDT"] // 86400) % 7
df["hour_of_day"] = (df["TransactionDT"] % 86400) // 3600

# ---------------------------------------------------------------------------
# TransactionAmt
# ---------------------------------------------------------------------------
print("\n--- Transaction Amount ---")
print(f"  Min:    ${df['TransactionAmt'].min():.2f}")
print(f"  Median: ${df['TransactionAmt'].median():.2f}")
print(f"  Mean:   ${df['TransactionAmt'].mean():.2f}")
print(f"  Max:    ${df['TransactionAmt'].max():.2f}")
amt_by_fraud = df.groupby("isFraud")["TransactionAmt"].describe()
print("\n  Amount by fraud label:")
print(amt_by_fraud.to_string())

# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------
print("\n--- Missing Values ---")
null_counts = df.isnull().sum()
null_pct = (null_counts / len(df) * 100).round(1)
missing = pd.DataFrame({"missing_count": null_counts, "missing_pct": null_pct})
missing = missing[missing["missing_count"] > 0].sort_values("missing_pct", ascending=False)
print(f"  {len(missing)} of {df.shape[1]} columns have missing values")
print(f"  Columns with >90% missing: {(missing['missing_pct'] > 90).sum()}")
print(f"  Columns with >50% missing: {(missing['missing_pct'] > 50).sum()}")
print(f"  Columns with >10% missing: {(missing['missing_pct'] > 10).sum()}")
print("\n  Top 20 most missing:")
print(missing.head(20).to_string())

# ---------------------------------------------------------------------------
# Categorical feature overview
# ---------------------------------------------------------------------------
cat_cols = df.select_dtypes(include="object").columns.tolist()
print(f"\n--- Categorical Features ({len(cat_cols)} columns) ---")
for col in cat_cols:
    n_unique = df[col].nunique()
    top_val = df[col].value_counts().index[0]
    top_pct = df[col].value_counts().iloc[0] / df[col].notna().sum() * 100
    print(f"  {col:<20} {n_unique:>4} unique  |  top: '{top_val}' ({top_pct:.1f}%)")

# ---------------------------------------------------------------------------
# Feature group summary
# Groups in this dataset: C, D, M, V columns plus card, addr, email, id
# ---------------------------------------------------------------------------
print("\n--- Feature Group Sizes ---")
groups = {
    "V (engineered, Vesta)": [c for c in df.columns if c.startswith("V")],
    "C (counting)":          [c for c in df.columns if c.startswith("C") and c[1:].isdigit()],
    "D (timedelta)":         [c for c in df.columns if c.startswith("D") and c[1:].isdigit()],
    "M (match flags)":       [c for c in df.columns if c.startswith("M") and c[1:].isdigit()],
    "id_ (identity)":        [c for c in df.columns if c.startswith("id_")],
    "card":                  [c for c in df.columns if c.startswith("card")],
}
for name, cols in groups.items():
    print(f"  {name:<30} {len(cols)} features")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
print("\nGenerating plots...")

# 1 - Fraud rate by hour of day
hour_fraud = df.groupby("hour_of_day")["isFraud"].agg(["sum", "count"])
hour_fraud["rate"] = hour_fraud["sum"] / hour_fraud["count"] * 100

fig1 = px.bar(
    hour_fraud.reset_index(),
    x="hour_of_day", y="rate",
    labels={"hour_of_day": "Hour of Day (relative)", "rate": "Fraud Rate (%)"},
    title="Fraud Rate by Hour of Day",
)
fig1.update_layout(xaxis=dict(tickmode="linear", dtick=1))
fig1.write_html(os.path.join(OUT_DIR, "eda_hour_fraud_rate.html"))

# 2 - Transaction amount distribution (fraud vs not), log scale
fig2 = px.histogram(
    df[df["TransactionAmt"] < 2000],
    x="TransactionAmt",
    color="isFraud",
    barmode="overlay",
    nbins=100,
    opacity=0.6,
    log_y=True,
    labels={"TransactionAmt": "Transaction Amount", "isFraud": "Is Fraud"},
    title="Transaction Amount Distribution (capped at $2000, log scale)",
    color_discrete_map={0: "steelblue", 1: "crimson"},
)
fig2.write_html(os.path.join(OUT_DIR, "eda_amount_distribution.html"))

# 3 - Missing value heatmap by feature group
fig3 = px.bar(
    missing.reset_index().rename(columns={"index": "column"}),
    x="column", y="missing_pct",
    labels={"column": "Column", "missing_pct": "Missing (%)"},
    title="Missing Value Rate by Column (columns with any missing)",
)
fig3.update_layout(xaxis_tickangle=-90, height=500)
fig3.write_html(os.path.join(OUT_DIR, "eda_missing_values.html"))

# 4 - Fraud rate by product code
if "ProductCD" in df.columns:
    prod_fraud = df.groupby("ProductCD")["isFraud"].agg(["mean", "count"]).reset_index()
    prod_fraud["fraud_rate_pct"] = prod_fraud["mean"] * 100
    fig4 = px.bar(
        prod_fraud.sort_values("fraud_rate_pct", ascending=False),
        x="ProductCD", y="fraud_rate_pct", text="count",
        labels={"ProductCD": "Product Code", "fraud_rate_pct": "Fraud Rate (%)"},
        title="Fraud Rate by Product Code (count = total transactions)",
    )
    fig4.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig4.write_html(os.path.join(OUT_DIR, "eda_product_fraud_rate.html"))

print("\nDone. Output files written to phase1_baseline/:")
print("  eda_hour_fraud_rate.html")
print("  eda_amount_distribution.html")
print("  eda_missing_values.html")
print("  eda_product_fraud_rate.html")
