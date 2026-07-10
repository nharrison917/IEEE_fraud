# -*- coding: utf-8 -*-
"""
Streamlit dashboard: IEEE-CIS Fraud Detection -- Cost-Sensitive Decision Framework
Run: streamlit run app.py

Reads models/cost_dashboard_data.json (produced by
phase2_cost_analysis/cost_threshold_analysis.py). All cost curves are
recomputed live in the browser from the validation threshold sweep via the
same base_cost/friction decomposition trick used in the prior
credit_card_fraud project's dashboard -- fp_cost at any (base, friction)
pair is linear in both, so a single stored sweep (at base=$3, friction=3%)
is enough to reconstruct the cost curve for ANY assumption the user picks,
without re-scoring the model.
"""

import os
import faulthandler

# Prints a C-level stack trace to stderr on a native crash (segfault) instead
# of the bare "Segmentation fault" the OS reports on its own -- diagnostic
# only, near-zero overhead, safe to leave on permanently.
faulthandler.enable()

# Must be set before numpy is imported (numpy's CPU/SIMD dispatch happens at
# import time) -- pandas also imports numpy internally, so this has to sit
# above both. Works around a segfault observed on Streamlit Community Cloud's
# containerized environment: numpy's own troubleshooting docs describe this
# exact failure mode -- a docker/VM misreporting CPU features it doesn't
# actually support (AVX512 in particular), causing numpy to crash trying to
# use them. Disabling the AVX512 family forces the safe fallback path;
# doesn't affect correctness, at most a minor performance cost that's
# irrelevant for this dashboard's data sizes.
os.environ.setdefault(
    "NPY_DISABLE_CPU_FEATURES",
    "AVX512F,AVX512CD,AVX512_KNL,AVX512_KNM,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL",
)

# Same class of bug, different library: pyarrow has its own C++ core with its
# own independent SIMD dispatch, completely separate from numpy's -- the env
# var above has no effect on it. Three consecutive Streamlit Cloud crash
# traces all rooted here: pandas' StringArray.__arrow_array__() (called
# unavoidably by Streamlit's own st.dataframe() -> convert_pandas_df_to_arrow_bytes,
# since sending data to the frontend grid widget requires Arrow format
# regardless of pandas' internal storage choice) segfaults constructing a
# pyarrow array from Python strings on this container. ARROW_USER_SIMD_LEVEL
# is Arrow's own official mechanism for this (see
# https://arrow.apache.org/docs/cpp/env_vars.html) -- NONE, not just
# disabling AVX512, since AVX512-only disabling already proved insufficient
# once for the sibling numpy issue and this has already cost several
# redeploy cycles.
os.environ.setdefault("ARROW_USER_SIMD_LEVEL", "NONE")

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# The actual root cause of the segfault the two fixes above didn't catch:
# pandas 3.0.3 defaults to future.infer_string=True + mode.string_storage=
# "auto", which resolves to pyarrow-backed string storage for ANY string
# data -- not just the one DataFrame construction originally patched. That
# fix (columnar instead of row-wise) only avoided ONE call path into pandas'
# pyarrow-backed string array code (pandas/core/arrays/string_arrow.py); the
# same crash reappeared through dict_to_mgr()'s own Index construction on
# the next attempt, confirming the bug is in constructing ANY pandas Index
# from Python strings on this specific container, regardless of code path.
# Verified locally (identical pandas 3.0.3): this option actually switches
# the backend to plain pandas.arrays.StringArray, not ArrowStringArray.
pd.set_option("mode.string_storage", "python")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="IEEE Fraud Detection -- Cost Dashboard",
    layout="wide",
)

# ============================================================
# DATA LOAD
# ============================================================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "models", "cost_dashboard_data.json")


@st.cache_data
def load_data():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


data = load_data()
sweep = data["validation_sweep"]
val_summary = data["validation_summary"]
test_results = data["test_results"]
meta = data["metadata"]

ALGO_KEYS = list(sweep.keys())  # ["lgb", "xgb", "ensemble"]
ALGO_LABELS = {"lgb": "LightGBM", "xgb": "XGBoost", "ensemble": "Ensemble (0.70 XGB / 0.30 LGB)"}
ALGO_COLORS = {"lgb": "#636EFA", "xgb": "#00CC96", "ensemble": "#EF553B"}

ORIG_BASE_COST = meta["base_fp_cost"]      # 3.0
ORIG_FRICTION  = meta["friction_rate"]     # 0.03
WINNER_KEY     = meta["winner_algorithm"]  # "ensemble"
WINNER_THRESH  = meta["winner_threshold"]  # 0.03


# ============================================================
# HELPERS -- cost-curve reconstruction at any (base, friction)
# ============================================================
def sweep_arrays(algo_key):
    m = sweep[algo_key]
    return {k: np.array(v) for k, v in m.items()}


@st.cache_data
def fp_amount_sums(sweep_json_str, orig_base, orig_friction):
    """
    Back out the sum of FP transaction amounts at each threshold, per
    algorithm, from the stored sweep (computed at base=$3, friction=3%):
        fp_cost[t] = base * n_fp[t] + friction * sum_fp_amt[t]
      =>  sum_fp_amt[t] = (fp_cost[t] - base * n_fp[t]) / friction
    Cached so it runs once at load, not on every slider move.
    """
    sweep_data = json.loads(sweep_json_str)
    out = {}
    for algo_key, m in sweep_data.items():
        n_fp = np.array(m["n_fp"], dtype=float)
        fp_c = np.array(m["fp_cost"], dtype=float)
        out[algo_key] = ((fp_c - orig_base * n_fp) / orig_friction).tolist()
    return out


_fp_amt_sums = fp_amount_sums(json.dumps(sweep), ORIG_BASE_COST, ORIG_FRICTION)


def recompute_total_cost(algo_key, base_cost, friction_rate):
    """Total cost curve (all thresholds) for algo_key at any (base, friction)."""
    m = sweep[algo_key]
    n_fp = np.array(m["n_fp"], dtype=float)
    fn_c = np.array(m["fn_cost"], dtype=float)
    fp_amt = np.array(_fp_amt_sums[algo_key])
    new_fp_cost = base_cost * n_fp + friction_rate * fp_amt
    return new_fp_cost + fn_c, new_fp_cost, fn_c


@st.cache_data
def sensitivity_sweep(algo_key, fixed_friction, fp_range):
    """Min achievable total cost + optimal threshold across a range of FP base costs."""
    t_arr = np.array(sweep[algo_key]["threshold"])
    min_costs, opt_thresholds = [], []
    for base in fp_range:
        total, _, _ = recompute_total_cost(algo_key, base, fixed_friction)
        idx = int(np.argmin(total))
        min_costs.append(float(total[idx]))
        opt_thresholds.append(float(t_arr[idx]))
    return min_costs, opt_thresholds


@st.cache_data
def friction_sensitivity_sweep(algo_key, fixed_base, friction_range):
    t_arr = np.array(sweep[algo_key]["threshold"])
    min_costs, opt_thresholds = [], []
    for fr in friction_range:
        total, _, _ = recompute_total_cost(algo_key, fixed_base, fr)
        idx = int(np.argmin(total))
        min_costs.append(float(total[idx]))
        opt_thresholds.append(float(t_arr[idx]))
    return min_costs, opt_thresholds


# ============================================================
# SIDEBAR -- all interactive controls
# ============================================================
st.sidebar.header("Controls")
st.sidebar.markdown(
    "Settings that affect the Threshold Explorer, Sensitivity Analysis, and "
    "Operational Reality sections below."
)

selected_algo = st.sidebar.selectbox(
    "Model",
    options=ALGO_KEYS,
    index=ALGO_KEYS.index(WINNER_KEY),
    format_func=lambda k: ALGO_LABELS[k],
)

cost_opt_t = val_summary[selected_algo]["cost_optimal_threshold"]

threshold = st.sidebar.slider(
    "Classification Threshold",
    min_value=0.01, max_value=0.99, value=float(round(cost_opt_t, 2)),
    step=0.01, format="%.2f",
)
st.sidebar.caption(f"Cost-optimal for {ALGO_LABELS[selected_algo]} (at \\${ORIG_BASE_COST:.0f} / {ORIG_FRICTION:.0%} assumptions): {cost_opt_t:.2f}")

st.sidebar.divider()

fp_base = st.sidebar.slider(
    "FP Base Cost ($ -- dispute/support handling per declined checkout)",
    min_value=0.5, max_value=20.0, value=float(ORIG_BASE_COST), step=0.5, format="$%.1f",
)
friction_pct = st.sidebar.slider(
    "Friction Rate (% of Amount -- forfeited fee revenue + retention-risk allowance)",
    min_value=1, max_value=15, value=int(ORIG_FRICTION * 100), step=1, format="%d%%",
)
friction_rate = friction_pct / 100.0
st.sidebar.caption("Both affect the Threshold Explorer and Sensitivity Analysis. Operational Reality uses the values above at the selected Threshold.")

# ============================================================
# SECTION 1 -- Header
# ============================================================
st.title("IEEE-CIS Fraud Detection")
st.markdown(
    "**Dataset:** 590,540 e-commerce transactions (Vesta) | 20,663 confirmed fraud cases | 3.5% fraud rate | "
    "chronological 60/20/20 train/val/test split"
)
st.info(
    "**Cost lens: payment processor**, not the merchant or the card-issuing bank. "
    "False negatives are priced as the full transaction amount (a chargeback-guarantee "
    "liability model common for fraud-decisioning vendors in this market); false positives "
    "are priced as the processor's own forfeited fee revenue plus a friction/retention-risk "
    "allowance -- not a merchant's lost margin or an issuer's investigation cost."
)
st.warning(
    "**The pure cost-minimizing threshold is aggressive:** at the original \\$3 / 3% "
    f"assumptions, the cost-optimal threshold ({WINNER_THRESH:.2f}) flags roughly **15.6%** "
    "of all transactions -- far above typical real-world decline rates (1-5%). This is a "
    "correct consequence of the ~19:1 FN:FP cost ratio, not a bug -- see **Limitations** "
    "for why a purely linear, per-transaction cost function can't see the reputational cost "
    "of a systematically high decline rate."
)
st.divider()

# ============================================================
# SECTION 2 -- Bottom Line
# ============================================================
st.header("Bottom Line")

winner_test = next(r for r in test_results if r["algo_key"] == WINNER_KEY and r["threshold_type"] == "Cost-Optimal (val)")
winner_default = next(r for r in test_results if r["algo_key"] == WINNER_KEY and r["threshold_type"] == "0.5 Default")
savings = winner_default["total_cost"] - winner_test["total_cost"]
sav_pct = savings / winner_default["total_cost"] * 100

st.caption(f"Production choice: **{ALGO_LABELS[WINNER_KEY]}**")
c1, c2, c3 = st.columns(3)
c1.metric(
    label=f"Cost-Optimal (threshold {WINNER_THRESH:.2f})",
    value=f"${winner_test['total_cost']:,.0f}",
    help="Production choice: chosen on validation, confirmed once on test.",
)
c2.metric(
    label="Default Threshold (0.50)",
    value=f"${winner_default['total_cost']:,.0f}",
)
c3.metric(
    label="Savings from Threshold Optimization",
    value=f"${savings:,.0f}",
    delta=f"-{sav_pct:.0f}% vs 0.50 default",
    delta_color="inverse",
)
st.caption(
    "Costs are illustrative -- TransactionAmt's currency/units are not confirmed by Kaggle. "
    f"Test set: {meta['n_test']:,} transactions, evaluated once. "
    "Figures are relative comparisons, not absolute dollar amounts."
)
st.divider()

# ============================================================
# SECTION 3 -- Threshold Explorer
# ============================================================
st.header("Threshold Explorer")
st.markdown(
    "Cost curves recompute live as you move the **FP Base Cost** / **Friction Rate** sliders -- "
    "no re-scoring needed, since total cost is linear in both. Use this to confirm the "
    "cost-optimal threshold for the selected model under any cost assumption."
)

total_c, fp_c, fn_c = recompute_total_cost(selected_algo, fp_base, friction_rate)
t_arr = sweep[selected_algo]["threshold"]
n_fp_arr = sweep[selected_algo]["n_fp"]
n_fn_arr = sweep[selected_algo]["n_fn"]
recall_arr = sweep[selected_algo]["recall"]

live_opt_idx = int(np.argmin(total_c))
live_opt_threshold = float(np.array(t_arr)[live_opt_idx])

fig = go.Figure()
fig.add_trace(go.Scatter(x=t_arr, y=total_c, name="Total Cost",
                          line=dict(color=ALGO_COLORS[selected_algo], width=2.5)))
fig.add_trace(go.Scatter(x=t_arr, y=fp_c, name="FP Cost (fee loss + friction)",
                          line=dict(color="#636EFA", width=1.5, dash="dot"), opacity=0.75))
fig.add_trace(go.Scatter(x=t_arr, y=fn_c, name="FN Cost (missed fraud)",
                          line=dict(color="#EF553B", width=1.5, dash="dot"), opacity=0.75))
fig.add_vline(x=live_opt_threshold, line_dash="dash", line_color="green", line_width=1.5,
              annotation_text=f"Cost-optimal at current sliders: {live_opt_threshold:.2f}",
              annotation_position="top right", annotation_font_color="green")
if abs(threshold - live_opt_threshold) > 0.015:
    fig.add_vline(x=threshold, line_dash="dash", line_color="grey", line_width=1.5,
                  annotation_text=f"Selected: {threshold:.2f}", annotation_position="top left")
fig.update_layout(
    xaxis_title="Threshold", yaxis_title="Cost ($)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=420, margin=dict(t=60, b=40),
)
st.plotly_chart(fig, width="stretch")

idx = int(np.argmin(np.abs(np.array(t_arr) - threshold)))
live_tot = float(total_c[idx])
live_fp = float(fp_c[idx])
live_fn = float(fn_c[idx])
live_rec = float(np.array(recall_arr)[idx])
live_nfp = int(np.array(n_fp_arr)[idx])
live_nfn = int(np.array(n_fn_arr)[idx])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Cost", f"${live_tot:,.0f}")
m2.metric("FP Cost", f"${live_fp:,.0f}", help=f"{live_nfp:,} false alarms flagged")
m3.metric("FN Cost", f"${live_fn:,.0f}", help=f"{live_nfn:,} fraud cases missed")
m4.metric("Recall", f"{live_rec:.1%}", help="Fraction of actual fraud caught")

flag_rate_pct = (live_nfp + (int(meta["n_fraud_val"]) - live_nfn)) / meta["n_val"] * 100
st.caption(
    f"Costs computed on validation set ({meta['n_val']:,} transactions) at FP base cost "
    f"\\${fp_base:.1f} / friction {friction_rate:.0%}. Flag rate at selected threshold: "
    f"**{flag_rate_pct:.1f}%** of all transactions. Official test set results in the table below."
)
st.divider()

# ============================================================
# SECTION 4 -- Algorithm Comparison Table (Test Set)
# ============================================================
st.header("Algorithm Comparison -- Test Set")
st.markdown(
    "All six combinations (3 algorithms x default / cost-optimal threshold, thresholds chosen "
    "on validation and confirmed once here). The production choice is highlighted."
)

# Built columnar (dict-of-lists), not row-wise (list-of-dicts), deliberately --
# pd.DataFrame([{...}, {...}]) routes column-name construction through
# pandas.core.internals.construction._list_of_dict_to_arrays(), which calls
# ensure_index() on the collected keys and segfaulted on Streamlit Community
# Cloud's container (crash trace pointed at pandas' pyarrow-backed string
# array code specifically in that call, not in any row value). Columnar
# construction never calls that function at all, so it isn't a workaround
# for the bug's symptom -- it avoids the buggy code path entirely.
columns = {
    "Algorithm": [], "Type": [], "Threshold": [], "Total Cost": [], "FP Cost": [],
    "FN Cost": [], "FP Count": [], "FN Count": [], "Precision": [], "Recall": [], "F1": [],
}
best_flags_list = []
for r in test_results:
    is_best = (r["algo_key"] == WINNER_KEY and r["threshold_type"] == "Cost-Optimal (val)")
    columns["Algorithm"].append(("* " if is_best else "  ") + r["algorithm"])
    columns["Type"].append(r["threshold_type"])
    columns["Threshold"].append(f"{r['threshold']:.2f}")
    columns["Total Cost"].append(f"${r['total_cost']:,.0f}")
    columns["FP Cost"].append(f"${r['fp_cost']:,.0f}")
    columns["FN Cost"].append(f"${r['fn_cost']:,.0f}")
    columns["FP Count"].append(r["n_fp"])
    columns["FN Count"].append(r["n_fn"])
    columns["Precision"].append(f"{r['precision']:.3f}")
    columns["Recall"].append(f"{r['recall']:.3f}")
    columns["F1"].append(f"{r['f1']:.3f}")
    best_flags_list.append(is_best)

table_df = pd.DataFrame(columns)
best_flags = pd.Series(best_flags_list)


def highlight_best(row):
    if best_flags.iloc[row.name]:
        return ["background-color: rgba(0, 204, 150, 0.18); font-weight: bold"] * len(row)
    return [""] * len(row)


st.dataframe(table_df.style.apply(highlight_best, axis=1), width="stretch", hide_index=True)
st.caption("* Production choice (chosen on validation, confirmed once on test -- not re-picked based on test performance).")
st.divider()

# ============================================================
# SECTION 5 -- Sensitivity Analysis
# ============================================================
st.header("Sensitivity Analysis")
st.markdown(
    "Does the algorithm choice or the aggressive flag rate depend on the exact cost "
    "assumption? Each panel sweeps one cost parameter (holding the other at the slider "
    "values in Controls) and shows the minimum achievable cost per algorithm."
)

fp_range = np.arange(0.5, 20.5, 0.5)
friction_range = np.arange(0.01, 0.155, 0.005)

sens_col1, sens_col2 = st.columns(2)

with sens_col1:
    fig_s1 = go.Figure()
    for algo_key in ALGO_KEYS:
        min_costs, opt_thresholds = sensitivity_sweep(algo_key, friction_rate, tuple(fp_range))
        fig_s1.add_trace(go.Scatter(x=fp_range, y=min_costs, name=ALGO_LABELS[algo_key],
                                     line=dict(color=ALGO_COLORS[algo_key], width=2)))
    fig_s1.add_vline(x=fp_base, line_dash="dash", line_color="grey", line_width=1,
                      annotation_text=f"${fp_base:.1f}", annotation_position="top right")
    fig_s1.update_layout(
        title=f"vs. FP Base Cost (friction fixed at {friction_rate:.0%})",
        xaxis_title="FP Base Cost ($)", yaxis_title="Min. Achievable Total Cost ($)",
        legend_title="Algorithm", height=380, margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig_s1, width="stretch")

with sens_col2:
    fig_s2 = go.Figure()
    for algo_key in ALGO_KEYS:
        min_costs, opt_thresholds = friction_sensitivity_sweep(algo_key, fp_base, tuple(friction_range))
        fig_s2.add_trace(go.Scatter(x=friction_range * 100, y=min_costs, name=ALGO_LABELS[algo_key],
                                     line=dict(color=ALGO_COLORS[algo_key], width=2)))
    fig_s2.add_vline(x=friction_pct, line_dash="dash", line_color="grey", line_width=1,
                      annotation_text=f"{friction_pct}%", annotation_position="top right")
    fig_s2.update_layout(
        title=f"vs. Friction Rate (base cost fixed at ${fp_base:.1f})",
        xaxis_title="Friction Rate (%)", yaxis_title="Min. Achievable Total Cost ($)",
        legend_title="Algorithm", height=380, margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig_s2, width="stretch")

st.caption(
    "The Ensemble's win over XGBoost on validation is thin (\\$255,922 vs \\$260,113 at the "
    "original assumptions) -- watch whether the ranking flips as you move either slider. "
    "The *aggressiveness* of the optimal threshold (not just which algorithm wins) also "
    "scales with these assumptions -- see the base-cost/friction-rate sweep printed in "
    "`phase2_cost_analysis/results_summary.md` for the exact numbers behind this chart."
)
st.divider()

# ============================================================
# SECTION 6 -- Operational Reality
# ============================================================
st.header("Operational Reality")
st.markdown(
    f"Using **{ALGO_LABELS[selected_algo]}** at threshold **{threshold:.2f}** "
    f"(FP base \\${fp_base:.1f}, friction {friction_rate:.0%}), scaled to 1,000,000 transactions:"
)

n_val = meta["n_val"]
n_fraud_val = meta["n_fraud_val"]
total_fraud_amt_val = meta["total_fraud_amount_val"]
scale_1m = 1_000_000 / n_val

tp_count = live_rec * n_fraud_val
flagged_1m = (tp_count + live_nfp) * scale_1m
genuine_1m = tp_count * scale_1m
false_alarm_1m = live_nfp * scale_1m
fp_cost_1m = live_fp * scale_1m
fn_cost_1m = live_fn * scale_1m
no_det_cost_1m = total_fraud_amt_val * scale_1m
fraud_prevented = no_det_cost_1m - fn_cost_1m
net_saving_1m = no_det_cost_1m - live_tot * scale_1m

o1, o2, o3 = st.columns(3)
o1.metric("Transactions Flagged", f"~{flagged_1m:,.0f}")
o2.metric("Genuine Fraud Caught", f"~{genuine_1m:,.0f}")
o3.metric("False Alarms (Declined Legit Checkouts)", f"~{false_alarm_1m:,.0f}")

o4, o5, o6 = st.columns(3)
o4.metric("False-Decline Cost", f"~${fp_cost_1m:,.0f}")
o5.metric("Fraud Loss Prevented", f"~${fraud_prevented:,.0f}")
o6.metric("Net Saving vs. No Detection", f"~${net_saving_1m:,.0f}")

st.caption(
    f"Estimates scaled from validation set ({n_val:,} transactions). "
    "Fraud loss prevented = total fraud amount minus undetected losses at this threshold."
)
st.divider()

# ============================================================
# SECTION 7 -- Limitations
# ============================================================
with st.expander("Limitations", expanded=True):
    st.markdown(rf"""
**Cost lens is a modeling choice.** The payment-processor framing (fee loss + retention risk
for false positives, chargeback-guarantee liability for false negatives) was chosen to match
this project's job-application target. A merchant or an issuing bank would see a materially
different cost structure -- see `phase2_cost_analysis/results_summary.md` for the reasoning.

**The cost function has no term for the AGGREGATE false-decline rate, only its sum.** The pure
cost-minimizing threshold ({WINNER_THRESH:.2f} at the original \${ORIG_BASE_COST:.0f}/{ORIG_FRICTION:.0%}
assumptions) flags approximately **15.6%** of all validation transactions. Each individual false
decline is priced correctly, but a processor declining 1 in 6 legitimate checkouts would in
practice trigger merchant churn through a threshold/nonlinear effect this linear, per-transaction
cost function cannot see. Documented here deliberately rather than masked behind an unjustified
capacity constraint (e.g. "cap flag rate at 5%") that would need its own separate justification.

**Chargeback-guarantee assumption is general industry knowledge, not verified from this
project's own data.** Not every processor contract works this way.

**Currency/units unconfirmed.** TransactionAmt's scale is not officially documented by Kaggle.
All cost figures are illustrative relative comparisons, not absolute dollar amounts.

**Real-time scoring gap.** Tier 2 entity-velocity features require each entity's full prior
transaction history, computed offline for this dashboard's data. A production service would
need a persisted per-entity running-state store (see
`phase2_feature_engineering/inference.py` docstring) rather than recomputing from raw history
on every call.

**Retention-risk cost is not independently estimated.** Folded into the friction rate as an
allowance, not derived from a separate churn model.
""")
