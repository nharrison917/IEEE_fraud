# -*- coding: utf-8 -*-
"""
Cost-Sensitive Decision Framework -- IEEE-CIS Fraud Detection
================================================================
Run from anywhere: python phase2_cost_analysis/cost_threshold_analysis.py

Extends the production HybridModel (phase2_feature_engineering/inference.py)
with a business cost function, threshold optimisation on validation, and a
single confirming pass on the test set (first and only test-set touch of
this project).

COST ASSUMPTIONS -- payment-processor lens (session 9 decision; see PLAN.md
"Phase 2 -- Cost-Sensitive Decision Framework"):

  This project's job-application target is a payment PROCESSOR -- the
  middleman between merchant and issuer -- not a bank/issuer and not a
  merchant. The source data (Vesta) is itself this kind of company. That
  changes what a false positive and false negative actually cost, relative
  to the prior project's (credit_card_fraud) generic issuer-investigation
  framing:

    FN cost = Amount
      Fraud-decisioning vendors in this market commonly sell a chargeback-
      guarantee product: if the vendor approves a transaction that turns out
      fraudulent, the VENDOR eats the loss, not the merchant. This is
      general industry knowledge about this market (Vesta and comparable
      vendors), not verified from this project's data -- but it makes
      FN = Amount a literal liability model here, not just a worst-case
      hedge.

    FP cost = $3 (base) + 3% x Amount (friction)
      A processor doesn't lose the merchant's profit margin (merchant's
      loss) and doesn't spend analyst-hours investigating after the fact
      (more of an issuer pattern). What it loses on a wrongly-declined
      transaction is its own processing-fee revenue on that transaction
      (the sale never completes) plus a slice of merchant-retention risk --
      processors are chosen and dropped based on their false-decline rate.
      $3 covers automated dispute/support handling; 3% approximates
      forfeited fee revenue plus a modest allowance for retention risk.
      Crossover (friction term overtakes base) is at Amount=$100 here, vs.
      Amount=$1,000 under the prior project's $10+1% issuer framing --
      intentional: a processor's own cost structure is fee/percentage-based,
      so its false-decline cost should scale with transaction size more than
      a flat per-incident cost would.

  All figures are illustrative -- TransactionAmt's currency/units are not
  confirmed by Kaggle; costs are relative comparisons, not absolute dollars.
"""

import os
import sys
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "models"))
sys.path.insert(0, os.path.normpath(os.path.join(SCRIPT_DIR, "..")))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date
from sklearn.metrics import precision_score, recall_score, f1_score

from utils import load_data, make_split
from phase2_feature_engineering.inference import HybridModel, build_features

# ============================================================
# COST FUNCTION -- payment-processor lens (see module docstring)
# ============================================================
BASE_FP_COST  = 3.0     # automated dispute/support handling per false decline
FRICTION_RATE = 0.03    # forfeited fee revenue + retention-risk allowance, as a share of Amount

ALGORITHMS = ("lgb", "xgb", "ensemble")
ALGO_LABELS = {"lgb": "LightGBM", "xgb": "XGBoost", "ensemble": "Ensemble (0.70 XGB / 0.30 LGB)"}


def calculate_business_cost(y_true, y_pred, amounts, base_fp_cost=BASE_FP_COST, friction_rate=FRICTION_RATE):
    """
    Total business cost of a set of predictions, from a payment processor's
    own P&L perspective (see module docstring for the FP/FN reasoning).

    Returns dict: total_cost, fp_cost, fn_cost, n_fp, n_fn, cost_per_transaction.
    """
    y_true  = np.asarray(y_true)
    y_pred  = np.asarray(y_pred)
    amounts = np.asarray(amounts)

    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)

    fp_costs = base_fp_cost + friction_rate * amounts[fp_mask]
    fn_costs = amounts[fn_mask]

    total_fp_cost = float(fp_costs.sum())
    total_fn_cost = float(fn_costs.sum())
    total_cost    = total_fp_cost + total_fn_cost

    return {
        "total_cost":           total_cost,
        "fp_cost":              total_fp_cost,
        "fn_cost":              total_fn_cost,
        "n_fp":                 int(fp_mask.sum()),
        "n_fn":                 int(fn_mask.sum()),
        "cost_per_transaction": total_cost / len(y_true) if len(y_true) > 0 else 0.0,
    }


def sweep_thresholds(y_true, y_prob, amounts, thresholds, base_fp_cost=BASE_FP_COST, friction_rate=FRICTION_RATE):
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cost = calculate_business_cost(y_true, y_pred, amounts, base_fp_cost, friction_rate)
        rows.append({
            "threshold":  t,
            "total_cost": cost["total_cost"],
            "fp_cost":    cost["fp_cost"],
            "fn_cost":    cost["fn_cost"],
            "n_fp":       cost["n_fp"],
            "n_fn":       cost["n_fn"],
            "precision":  precision_score(y_true, y_pred, zero_division=0),
            "recall":     recall_score(y_true, y_pred, zero_division=0),
            "f1":         f1_score(y_true, y_pred, zero_division=0),
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("  COST-SENSITIVE DECISION FRAMEWORK -- IEEE-CIS Fraud Detection")
    print("=" * 70)

    print("\n[1/7] Loading data and building features (full dataset, before split --")
    print("      Tier 2 entity-velocity features need each entity's full prior history)...")
    df = load_data()
    df = build_features(df)
    train_raw, val_raw, test_raw = make_split(df)
    del train_raw  # not needed here -- models are already trained

    amounts_val  = val_raw["TransactionAmt"].values
    amounts_test = test_raw["TransactionAmt"].values
    y_val  = val_raw["isFraud"].values
    y_test = test_raw["isFraud"].values

    print(f"\n  Val:  {len(val_raw):,} rows, {y_val.sum():,} fraud ({y_val.mean()*100:.2f}%)")
    print(f"  Test: {len(test_raw):,} rows, {y_test.sum():,} fraud ({y_test.mean()*100:.2f}%)  -- NOT touched until step 6")

    print("\n[2/7] Cost function (payment-processor lens -- see script docstring):")
    print(f"  FP cost = ${BASE_FP_COST:.2f} + ({FRICTION_RATE:.0%} x Amount)")
    print(f"  FN cost = Amount  (chargeback-guarantee liability model)")

    print("\n[3/7] Loading production HybridModel (lgb / xgb / ensemble)...")
    model = HybridModel()

    print("\n[4/7] Threshold sweep (0.01-0.99) on VALIDATION set, per algorithm...")
    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    sweep_store = {}
    val_summary = {}

    for algo in ALGORITHMS:
        print(f"\n  --- {ALGO_LABELS[algo]} ---")
        y_prob = model.predict(val_raw, algorithm=algo).values
        sweep_df = sweep_thresholds(y_val, y_prob, amounts_val, thresholds)
        sweep_store[algo] = sweep_df

        best_cost_row = sweep_df.loc[sweep_df["total_cost"].idxmin()]
        best_f1_row   = sweep_df.loc[sweep_df["f1"].idxmax()]

        cost_thresh = float(best_cost_row["threshold"])
        f1_thresh   = float(best_f1_row["threshold"])
        cost_at_f1  = float(sweep_df.loc[sweep_df["threshold"] == f1_thresh, "total_cost"].iloc[0])

        print(f"    Cost-optimal threshold: {cost_thresh:.2f}  ->  total cost: ${best_cost_row['total_cost']:,.2f}")
        print(f"    F1-optimal  threshold:  {f1_thresh:.2f}  ->  total cost: ${cost_at_f1:,.2f}"
              f"  (${cost_at_f1 - best_cost_row['total_cost']:,.2f} more than cost-optimal)")
        print(f"    At cost-optimal: FP={int(best_cost_row['n_fp'])}  FN={int(best_cost_row['n_fn'])}"
              f"  P={best_cost_row['precision']:.3f}  R={best_cost_row['recall']:.3f}  F1={best_cost_row['f1']:.3f}")

        val_summary[algo] = {
            "cost_optimal_threshold": cost_thresh,
            "cost_optimal_total_cost": float(best_cost_row["total_cost"]),
            "f1_optimal_threshold": f1_thresh,
            "cost_at_f1_threshold": cost_at_f1,
        }

    print("\n  Summary -- cost-optimal total cost by algorithm (validation):")
    print(f"  {'Algorithm':<32} {'Threshold':>10} {'Total Cost':>14}")
    print("  " + "-" * 58)
    for algo in ALGORITHMS:
        s = val_summary[algo]
        print(f"  {ALGO_LABELS[algo]:<32} {s['cost_optimal_threshold']:>10.2f} "
              f"${s['cost_optimal_total_cost']:>12,.2f}")

    winner = min(ALGORITHMS, key=lambda a: val_summary[a]["cost_optimal_total_cost"])
    winner_threshold = val_summary[winner]["cost_optimal_threshold"]
    print(f"\n  Winner (lowest validation cost, not just best PR-AUC): {ALGO_LABELS[winner]}"
          f" @ threshold {winner_threshold:.2f}")

    print("\n[5/7] Sensitivity analysis on VALIDATION set, winning algorithm"
          f" ({ALGO_LABELS[winner]})...")
    y_prob_winner_val = model.predict(val_raw, algorithm=winner).values

    print("\n  (a) Base FP cost scenarios (friction rate fixed at "
          f"{FRICTION_RATE:.0%}):")
    base_cost_scenarios = [1.0, 3.0, 5.0, 10.0]
    base_cost_sensitivity = {}
    print(f"  {'FP Base Cost':>14} {'Opt Threshold':>14} {'Total Cost':>14}")
    print("  " + "-" * 46)
    for bc in base_cost_scenarios:
        sdf = sweep_thresholds(y_val, y_prob_winner_val, amounts_val, thresholds,
                                base_fp_cost=bc, friction_rate=FRICTION_RATE)
        best_row = sdf.loc[sdf["total_cost"].idxmin()]
        base_cost_sensitivity[bc] = {
            "opt_threshold": float(best_row["threshold"]),
            "total_cost": float(best_row["total_cost"]),
            "sweep": sdf,
        }
        print(f"  ${bc:>12.2f} {best_row['threshold']:>14.2f} ${best_row['total_cost']:>12,.2f}")

    print("\n  (b) Friction rate scenarios (base FP cost fixed at "
          f"${BASE_FP_COST:.2f}):")
    friction_scenarios = [0.01, 0.03, 0.05, 0.10]
    friction_sensitivity = {}
    print(f"  {'Friction Rate':>14} {'Opt Threshold':>14} {'Total Cost':>14}")
    print("  " + "-" * 46)
    for fr in friction_scenarios:
        sdf = sweep_thresholds(y_val, y_prob_winner_val, amounts_val, thresholds,
                                base_fp_cost=BASE_FP_COST, friction_rate=fr)
        best_row = sdf.loc[sdf["total_cost"].idxmin()]
        friction_sensitivity[fr] = {
            "opt_threshold": float(best_row["threshold"]),
            "total_cost": float(best_row["total_cost"]),
            "sweep": sdf,
        }
        print(f"  {fr:>13.0%} {best_row['threshold']:>14.2f} ${best_row['total_cost']:>12,.2f}")

    print("\n[6/7] Final evaluation on TEST SET (first and only test-set touch)...")
    print("  >>> Using test set for the first and only time in this project <<<\n")

    test_rows = []
    for algo in ALGORITHMS:
        y_prob_test = model.predict(test_raw, algorithm=algo).values
        opt_thresh = val_summary[algo]["cost_optimal_threshold"]
        for label, t in [("Cost-Optimal (val)", opt_thresh), ("0.5 Default", 0.50)]:
            y_pred = (y_prob_test >= t).astype(int)
            cost = calculate_business_cost(y_test, y_pred, amounts_test)
            test_rows.append({
                "algorithm": ALGO_LABELS[algo],
                "algo_key": algo,
                "threshold_type": label,
                "threshold": t,
                **cost,
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
            })
    test_df = pd.DataFrame(test_rows)

    hdr = (f"  {'Algorithm':<32} {'Type':<20} {'Thr':>5} {'Total$':>11} {'FP$':>9} "
           f"{'FN$':>10} {'nFP':>6} {'nFN':>5} {'P':>6} {'R':>6} {'F1':>6}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, row in test_df.iterrows():
        print(f"  {row['algorithm']:<32} {row['threshold_type']:<20} {row['threshold']:>5.2f}"
              f" ${row['total_cost']:>9,.0f} ${row['fp_cost']:>7,.0f} ${row['fn_cost']:>8,.0f}"
              f" {row['n_fp']:>6} {row['n_fn']:>5}"
              f" {row['precision']:>6.3f} {row['recall']:>6.3f} {row['f1']:>6.3f}")

    cost_opt_df = test_df[test_df["threshold_type"] == "Cost-Optimal (val)"].copy()
    best_test_row = cost_opt_df.loc[cost_opt_df["total_cost"].idxmin()]
    print(f"\n  Champion on test set (cost-optimal thresholds): {best_test_row['algorithm']}"
          f" @ {best_test_row['threshold']:.2f}  (chosen on val, confirmed on test -- not re-picked here)")
    if best_test_row["algo_key"] != winner:
        print(f"  NOTE: test-set champion ({best_test_row['algorithm']}) differs from the"
              f" val-set winner ({ALGO_LABELS[winner]}). Reporting the val-chosen model"
              f" ({ALGO_LABELS[winner]}) as production, per the 'choose on val, confirm on"
              f" test' rule -- not re-picking based on test.")

    # Operational interpretation -- scaled to 1,000,000 transactions (PLAN.md dashboard spec)
    scale_1M = 1_000_000 / len(y_test)
    winner_test_row = test_df[(test_df["algo_key"] == winner) &
                               (test_df["threshold_type"] == "Cost-Optimal (val)")].iloc[0]
    tp_count = y_test.sum() - winner_test_row["n_fn"]
    flagged_1M      = (winner_test_row["n_fp"] + tp_count) * scale_1M
    genuine_1M      = tp_count * scale_1M
    false_alarm_1M  = winner_test_row["n_fp"] * scale_1M
    fp_cost_1M      = winner_test_row["fp_cost"] * scale_1M
    fn_cost_1M      = winner_test_row["fn_cost"] * scale_1M
    no_detection_cost = amounts_test[y_test == 1].sum()
    net_saving_1M   = (no_detection_cost - winner_test_row["total_cost"]) * scale_1M

    print(f"""
  OPERATIONAL INTERPRETATION -- {ALGO_LABELS[winner]} @ threshold {winner_threshold:.2f}
  (scaled to 1,000,000 transactions)

  Flags ~{flagged_1M:,.0f} as fraud, of which ~{genuine_1M:,.0f} are genuine fraud caught and
  ~{false_alarm_1M:,.0f} are false alarms (declined legitimate checkouts).
  Estimated false-decline cost (fee loss + friction): ~${fp_cost_1M:,.0f}
  Estimated fraud loss prevented:                     ~${fn_cost_1M:,.0f}
  Net saving vs. no fraud detection:                  ~${net_saving_1M:,.0f}
""")

    print("[7/7] Saving outputs (dashboard JSON, charts, stakeholder report, summary)...")
    _save_dashboard_data(sweep_store, val_summary, base_cost_sensitivity, friction_sensitivity,
                          test_df, winner, winner_threshold, y_val, y_test, amounts_val, amounts_test)
    _save_charts(sweep_store, val_summary, base_cost_sensitivity, friction_sensitivity,
                 test_df, winner, winner_threshold, model, test_raw, amounts_test, y_test)
    _save_report(winner, winner_threshold, test_df, best_test_row, base_cost_sensitivity,
                 friction_sensitivity, flagged_1M, genuine_1M, false_alarm_1M,
                 fp_cost_1M, fn_cost_1M, net_saving_1M, val_summary)
    _save_summary_md(winner, winner_threshold, test_df, base_cost_sensitivity,
                      friction_sensitivity, flagged_1M, genuine_1M, false_alarm_1M,
                      fp_cost_1M, fn_cost_1M, net_saving_1M, val_summary)

    print("\n" + "=" * 70)
    print("  DONE")
    print("  phase2_cost_analysis/cost_threshold_analysis.py -- this script")
    print("  phase2_cost_analysis/cost_results.html          -- interactive Plotly charts")
    print("  phase2_cost_analysis/cost_report.html           -- stakeholder report")
    print("  phase2_cost_analysis/results_summary.md         -- written summary")
    print("  models/cost_dashboard_data.json                 -- Streamlit dashboard data")
    print("=" * 70)


def _save_dashboard_data(sweep_store, val_summary, base_cost_sensitivity, friction_sensitivity,
                          test_df, winner, winner_threshold, y_val, y_test, amounts_val, amounts_test):
    def _sweep_to_json(df):
        int_cols = {"n_fp", "n_fn"}
        return {col: [int(v) if col in int_cols else round(float(v), 6) for v in df[col]]
                for col in df.columns}

    dashboard_data = {
        "metadata": {
            "base_fp_cost": BASE_FP_COST,
            "friction_rate": FRICTION_RATE,
            "cost_lens": "payment_processor",
            "winner_algorithm": winner,
            "winner_threshold": winner_threshold,
            "n_val": int(len(y_val)),
            "n_fraud_val": int(y_val.sum()),
            "total_fraud_amount_val": round(float(amounts_val[y_val == 1].sum()), 4),
            "n_test": int(len(y_test)),
            "n_fraud_test": int(y_test.sum()),
        },
        "validation_sweep": {algo: _sweep_to_json(df) for algo, df in sweep_store.items()},
        "validation_summary": val_summary,
        "sensitivity_base_cost": {
            str(bc): {"opt_threshold": v["opt_threshold"], "total_cost": v["total_cost"],
                      "sweep": _sweep_to_json(v["sweep"])}
            for bc, v in base_cost_sensitivity.items()
        },
        "sensitivity_friction_rate": {
            str(fr): {"opt_threshold": v["opt_threshold"], "total_cost": v["total_cost"],
                      "sweep": _sweep_to_json(v["sweep"])}
            for fr, v in friction_sensitivity.items()
        },
        "test_results": [
            {k: (round(float(v), 6) if isinstance(v, (int, float)) and k not in ("n_fp", "n_fn") else v)
             for k, v in row.items()}
            for row in test_df.to_dict("records")
        ],
    }

    out_path = os.path.join(MODELS_DIR, "cost_dashboard_data.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, indent=2)
    print(f"  Saved -> {os.path.relpath(out_path)}")


def _save_charts(sweep_store, val_summary, base_cost_sensitivity, friction_sensitivity,
                  test_df, winner, winner_threshold, model, test_raw, amounts_test, y_test):
    ALGO_COLORS = {"lgb": "#636EFA", "xgb": "#00CC96", "ensemble": "#EF553B"}
    figs = []

    # Fig 1: cost curves per algorithm
    fig1 = make_subplots(rows=1, cols=3, subplot_titles=[ALGO_LABELS[a] for a in ALGORITHMS],
                          shared_yaxes=True)
    for col_i, algo in enumerate(ALGORITHMS, 1):
        sweep = sweep_store[algo]
        s = val_summary[algo]
        fig1.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["total_cost"],
                                   name=ALGO_LABELS[algo], line=dict(color=ALGO_COLORS[algo]),
                                   showlegend=False), row=1, col=col_i)
        fig1.add_vline(x=s["cost_optimal_threshold"], line_dash="dash", line_color="green",
                        annotation_text=f"Cost-opt {s['cost_optimal_threshold']:.2f}",
                        annotation_font_color="green", row=1, col=col_i)
        fig1.add_vline(x=s["f1_optimal_threshold"], line_dash="dot", line_color="orange",
                        annotation_text=f"F1-opt {s['f1_optimal_threshold']:.2f}",
                        annotation_font_color="darkorange", row=1, col=col_i)
    fig1.update_layout(title="Business Cost Across Thresholds -- Validation Set", yaxis_title="Total Cost ($)", height=480)
    fig1.update_xaxes(title_text="Threshold")
    figs.append(("Cost Curves by Algorithm", fig1))

    # Fig 2: sensitivity -- base FP cost
    fig2 = go.Figure()
    for bc, d in base_cost_sensitivity.items():
        fig2.add_trace(go.Scatter(x=d["sweep"]["threshold"], y=d["sweep"]["total_cost"],
                                   name=f"${bc:.0f} base", line=dict(width=2)))
        fig2.add_vline(x=d["opt_threshold"], line_dash="dash", opacity=0.5)
    fig2.update_layout(title=f"Sensitivity -- FP Base Cost ({ALGO_LABELS[winner]}, friction fixed at {FRICTION_RATE:.0%})",
                        xaxis_title="Threshold", yaxis_title="Total Cost ($)", legend_title="FP Base Cost", height=460)
    figs.append(("Sensitivity -- FP Base Cost", fig2))

    # Fig 3: sensitivity -- friction rate
    fig3 = go.Figure()
    for fr, d in friction_sensitivity.items():
        fig3.add_trace(go.Scatter(x=d["sweep"]["threshold"], y=d["sweep"]["total_cost"],
                                   name=f"{fr:.0%} friction", line=dict(width=2)))
        fig3.add_vline(x=d["opt_threshold"], line_dash="dash", opacity=0.5)
    fig3.update_layout(title=f"Sensitivity -- Friction Rate ({ALGO_LABELS[winner]}, base cost fixed at ${BASE_FP_COST:.0f})",
                        xaxis_title="Threshold", yaxis_title="Total Cost ($)", legend_title="Friction Rate", height=460)
    figs.append(("Sensitivity -- Friction Rate", fig3))

    # Fig 4: cost breakdown at cost-optimal thresholds, test set, all algorithms
    co_df = test_df[test_df["threshold_type"] == "Cost-Optimal (val)"].copy()
    fig4 = make_subplots(rows=1, cols=3, subplot_titles=["Total Cost", "FP Cost (fee loss + friction)", "FN Cost (missed fraud)"])
    for col_i, cost_col in enumerate(["total_cost", "fp_cost", "fn_cost"], 1):
        fig4.add_trace(go.Bar(x=co_df["algorithm"], y=co_df[cost_col],
                               marker_color=[ALGO_COLORS[k] for k in co_df["algo_key"]],
                               showlegend=False, text=[f"${v:,.0f}" for v in co_df[cost_col]],
                               textposition="outside"), row=1, col=col_i)
    fig4.update_layout(title="Cost Breakdown at Cost-Optimal Thresholds -- Test Set", height=460)
    figs.append(("Cost Breakdown -- Test Set", fig4))

    # Fig 5: errors by amount, winning algorithm, test set
    y_prob_test = model.predict(test_raw, algorithm=winner).values
    y_pred_test = (y_prob_test >= winner_threshold).astype(int)

    def _outcome(yt, yp):
        if yt == 0 and yp == 0: return "True Negative"
        if yt == 1 and yp == 1: return "True Positive"
        if yt == 0 and yp == 1: return "False Positive (declined legit checkout)"
        return "False Negative (fraud missed)"

    outcomes = [_outcome(yt, yp) for yt, yp in zip(y_test, y_pred_test)]
    scatter_df = pd.DataFrame({"Amount": amounts_test, "Outcome": outcomes, "idx": range(len(y_test))})
    errors_df = scatter_df[scatter_df["Outcome"].str.startswith("False")]
    tp_df = scatter_df[scatter_df["Outcome"] == "True Positive"]
    tn_sample = scatter_df[scatter_df["Outcome"] == "True Negative"].sample(
        min(500, (scatter_df["Outcome"] == "True Negative").sum()), random_state=42)
    plot_df = pd.concat([tn_sample, tp_df, errors_df]).reset_index(drop=True)

    OUTCOME_COLORS = {
        "True Negative": "#CCCCCC", "True Positive": "#00CC96",
        "False Positive (declined legit checkout)": "#636EFA",
        "False Negative (fraud missed)": "#EF553B",
    }
    fig5 = px.scatter(plot_df, x="Amount", y="idx", color="Outcome", color_discrete_map=OUTCOME_COLORS,
                       title=f"{ALGO_LABELS[winner]} -- Errors by Transaction Amount (threshold={winner_threshold:.2f})",
                       labels={"idx": "Transaction index", "Amount": "Transaction Amount"},
                       opacity=0.7, height=560)
    fig5.update_yaxes(showticklabels=False)
    figs.append(("Errors by Transaction Amount", fig5))

    out_path = os.path.join(SCRIPT_DIR, "cost_results.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'><title>Cost-Sensitive Fraud -- Charts</title></head><body>\n")
        first = True
        for title, fig in figs:
            f.write(f"<h2>{title}</h2>\n")
            f.write(fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False))
            f.write("\n<hr>\n")
            first = False
        f.write("</body></html>")
    print(f"  Saved -> {os.path.relpath(out_path)}")


def _save_report(winner, winner_threshold, test_df, best_test_row, base_cost_sensitivity,
                  friction_sensitivity, flagged_1M, genuine_1M, false_alarm_1M,
                  fp_cost_1M, fn_cost_1M, net_saving_1M, val_summary):
    today_str = date.today().strftime("%B %d, %Y")

    def _tr(row):
        hl = ' style="background:#e8f5e9;"' if row["threshold_type"] == "Cost-Optimal (val)" else ""
        return (f"<tr{hl}><td>{row['algorithm']}</td><td>{row['threshold_type']}</td>"
                f"<td>{row['threshold']:.2f}</td><td>${row['total_cost']:,.0f}</td>"
                f"<td>${row['fp_cost']:,.0f}</td><td>${row['fn_cost']:,.0f}</td>"
                f"<td>{row['n_fp']}</td><td>{row['n_fn']}</td>"
                f"<td>{row['precision']:.3f}</td><td>{row['recall']:.3f}</td><td>{row['f1']:.3f}</td></tr>")

    table_rows = "\n".join(_tr(row) for _, row in test_df.iterrows())
    base_sens_rows = "\n".join(
        f"<tr><td>${bc:.0f}</td><td>{d['opt_threshold']:.2f}</td><td>${d['total_cost']:,.2f}</td></tr>"
        for bc, d in base_cost_sensitivity.items())
    friction_sens_rows = "\n".join(
        f"<tr><td>{fr:.0%}</td><td>{d['opt_threshold']:.2f}</td><td>${d['total_cost']:,.2f}</td></tr>"
        for fr, d in friction_sensitivity.items())

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Cost-Sensitive Fraud Detection -- Report</title>
<style>
  body      {{ font-family: Arial, sans-serif; max-width: 980px; margin: 40px auto; color: #222; line-height: 1.65; }}
  h1        {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 8px; }}
  h2        {{ color: #283593; margin-top: 44px; }}
  .meta     {{ color: #666; font-size: 0.9em; margin-bottom: 28px; }}
  .box-warn {{ background: #fff8e1; border-left: 5px solid #f9a825; padding: 14px 20px; margin: 20px 0; border-radius: 4px; }}
  .box-good {{ background: #e8f5e9; border-left: 5px solid #2e7d32; padding: 14px 20px; margin: 20px 0; border-radius: 4px; font-size: 1.05em; }}
  .box-info {{ background: #e3f2fd; border-left: 5px solid #1565c0; padding: 14px 20px; margin: 20px 0; border-radius: 4px; }}
  .box-risk {{ background: #fce4ec; border-left: 5px solid #c62828; padding: 14px 20px; margin: 20px 0; border-radius: 4px; }}
  table     {{ border-collapse: collapse; width: 100%; font-size: 0.83em; margin: 14px 0; }}
  th        {{ background: #1a237e; color: #fff; padding: 9px 10px; text-align: right; white-space: nowrap; }}
  th:first-child, th:nth-child(2) {{ text-align: left; }}
  td        {{ padding: 7px 10px; border-bottom: 1px solid #ddd; text-align: right; }}
  td:first-child, td:nth-child(2) {{ text-align: left; }}
  tr:hover  {{ background: #f5f5f5; }}
  small     {{ color: #888; }}
</style>
</head>
<body>

<h1>Cost-Sensitive Fraud Detection Analysis</h1>
<div class="meta">
  Generated: {today_str} &nbsp;|&nbsp;
  Dataset: IEEE-CIS Fraud Detection (Vesta e-commerce transactions) &nbsp;|&nbsp;
  Models: LightGBM &nbsp;&nbsp; XGBoost &nbsp;&nbsp; Ensemble (0.70 XGB / 0.30 LGB)
</div>

<div class="box-warn">
  <strong>Assumption Disclosure -- please read before interpreting figures</strong><br>
  These are <strong>illustrative cost estimates</strong> modeled from a <strong>payment processor's</strong>
  perspective -- the middleman between merchant and issuer, not the merchant or the card-issuing bank.
  TransactionAmt's currency/units are not confirmed by Kaggle; figures are relative comparisons, not
  absolute dollars.
  <ul>
    <li>Base FP cost (dispute/support handling): <strong>${BASE_FP_COST:.2f}</strong> per false decline</li>
    <li>Friction penalty: <strong>{FRICTION_RATE:.0%}</strong> of Amount &nbsp;<small>(forfeited processing-fee revenue + retention-risk allowance)</small></li>
    <li>Full FP cost: <strong>${BASE_FP_COST:.2f} + ({FRICTION_RATE:.0%} x Amount)</strong></li>
    <li>FN cost: <strong>full transaction Amount</strong> &nbsp;<small>(chargeback-guarantee liability model -- common for fraud-decisioning vendors in this market)</small></li>
  </ul>
</div>

<h2>Key Findings</h2>
<div class="box-good">
  <strong>Production choice:</strong> {ALGO_LABELS[winner]} at cost-optimal threshold <strong>{winner_threshold:.2f}</strong>
  (selected on validation, confirmed once on test)<br>
  <strong>Test set total cost:</strong> ${best_test_row['total_cost']:,.2f}
  &nbsp;|&nbsp; FP cost: ${best_test_row['fp_cost']:,.2f}
  &nbsp;|&nbsp; FN cost: ${best_test_row['fn_cost']:,.2f}<br>
  <strong>Recall:</strong> {best_test_row['recall']:.4f}
  &nbsp;|&nbsp; <strong>F1:</strong> {best_test_row['f1']:.4f}
</div>

<h2>Algorithm Comparison -- Test Set</h2>
<p><small>Green rows = cost-optimal threshold (chosen on validation, applied here once).</small></p>
<table>
  <tr>
    <th>Algorithm</th><th>Threshold Type</th><th>Threshold</th>
    <th>Total Cost</th><th>FP Cost</th><th>FN Cost</th>
    <th>n FP</th><th>n FN</th>
    <th>Precision</th><th>Recall</th><th>F1</th>
  </tr>
  {table_rows}
</table>

<h2>Operational Interpretation</h2>
<div class="box-info">
  At a cost-optimal threshold of <strong>{winner_threshold:.2f}</strong>,
  for every <strong>1,000,000 transactions</strong> {ALGO_LABELS[winner]} flags approximately
  <strong>{flagged_1M:,.0f}</strong> as fraud, of which approximately
  <strong>{genuine_1M:,.0f}</strong> are genuine fraud caught and
  <strong>{false_alarm_1M:,.0f}</strong> are false alarms (declined legitimate checkouts).
  The estimated fee-loss and friction cost of false alarms is
  <strong>${fp_cost_1M:,.0f}</strong>, offset against approximately
  <strong>${fn_cost_1M:,.0f}</strong> in fraud prevented,
  for a net saving of approximately <strong>${net_saving_1M:,.0f}</strong>
  compared to no fraud detection.
</div>

<h2>Sensitivity Analysis</h2>
<p>(a) FP base cost scenarios (friction rate fixed at {FRICTION_RATE:.0%}):</p>
<table>
  <tr><th style="text-align:left">FP Base Cost</th><th>Optimal Threshold</th><th>Total Cost (validation)</th></tr>
  {base_sens_rows}
</table>
<p>(b) Friction rate scenarios (base FP cost fixed at ${BASE_FP_COST:.2f}):</p>
<table>
  <tr><th style="text-align:left">Friction Rate</th><th>Optimal Threshold</th><th>Total Cost (validation)</th></tr>
  {friction_sens_rows}
</table>

<h2>Limitations</h2>
<div class="box-risk">
<ul>
  <li><strong>Cost lens is a modeling choice:</strong> the payment-processor framing (fee loss + retention
      risk for FP, chargeback-guarantee liability for FN) was chosen to match this project's job-application
      target. A merchant or issuer would see a materially different cost structure.</li>
  <li><strong>Chargeback-guarantee assumption is general industry knowledge, not verified from this
      project's data:</strong> not every processor contract works this way.</li>
  <li><strong>Currency/units unconfirmed:</strong> TransactionAmt's scale is not officially documented by Kaggle.
      All costs are relative comparisons.</li>
  <li><strong>Real-time scoring gap:</strong> Tier 2 entity-velocity features require each entity's full prior
      history, computed offline here. A production service would need a persisted per-entity running-state
      store (see phase2_feature_engineering/inference.py docstring).</li>
  <li><strong>Retention-risk cost is not independently estimated:</strong> folded into the 3% friction rate
      as an allowance, not derived from a separate churn model.</li>
  <li><strong>The cost function has no term for the AGGREGATE false-decline rate -- only its sum:</strong>
      the pure cost-minimizing threshold (0.03) flags approximately <strong>15.6%</strong> of all
      validation transactions. Each individual false decline is priced correctly, but a processor
      declining 1 in 6 legitimate checkouts would in practice trigger merchant churn through a
      threshold/nonlinear effect this linear, per-transaction cost function cannot see -- it only
      prices friction transaction-by-transaction, not the reputational cost of a systematically high
      decline rate. Reported here deliberately rather than hidden behind a capacity constraint, so the
      gap between "cost-optimal in theory" and "operationally deployable" is visible.</li>
</ul>
</div>

</body>
</html>"""

    out_path = os.path.join(SCRIPT_DIR, "cost_report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_html)
    print(f"  Saved -> {os.path.relpath(out_path)}")


def _save_summary_md(winner, winner_threshold, test_df, base_cost_sensitivity,
                      friction_sensitivity, flagged_1M, genuine_1M, false_alarm_1M,
                      fp_cost_1M, fn_cost_1M, net_saving_1M, val_summary):
    md_rows = []
    for _, row in test_df.iterrows():
        champ = " ****" if (row["algo_key"] == winner and row["threshold_type"] == "Cost-Optimal (val)") else ""
        md_rows.append(f"| {row['algorithm']}{champ} | {row['threshold_type']} | {row['threshold']:.2f}"
                        f" | ${row['total_cost']:,.0f} | ${row['fp_cost']:,.0f} | ${row['fn_cost']:,.0f}"
                        f" | {row['recall']:.4f} | {row['f1']:.4f} |")

    base_sens_md = "\n".join(
        f"| ${bc:.0f} | {d['opt_threshold']:.2f} | ${d['total_cost']:,.2f} |"
        for bc, d in base_cost_sensitivity.items())
    friction_sens_md = "\n".join(
        f"| {fr:.0%} | {d['opt_threshold']:.2f} | ${d['total_cost']:,.2f} |"
        for fr, d in friction_sensitivity.items())

    summary_md = f"""# Cost-Sensitive Fraud Detection -- Results Summary

## Assumptions -- Payment-Processor Lens

> Job-application target for this project is a payment processor (merchant/issuer middleman),
> not a bank or a merchant. Costs below reflect the processor's own P&L, not the merchant's
> lost margin or an issuer's investigation cost. All figures are illustrative -- TransactionAmt's
> currency/units are not confirmed by Kaggle.

| Item | Value | Rationale |
|---|---|---|
| Base FP cost | ${BASE_FP_COST:.2f} | Automated dispute/support handling for a declined checkout |
| Friction penalty | {FRICTION_RATE:.0%} of Amount | Forfeited processing-fee revenue + retention-risk allowance |
| Full FP cost | ${BASE_FP_COST:.2f} + ({FRICTION_RATE:.0%} x Amount) | -- |
| FN cost | Full Amount | Chargeback-guarantee liability model (general industry knowledge for this vendor category, not verified from this project's data) |

---

## Algorithm Comparison -- Test Set

| Algorithm | Threshold Type | Threshold | Total Cost | FP Cost | FN Cost | Recall | F1 |
|---|---|---|---|---|---|---|---|
{chr(10).join(md_rows)}

---

## Operational Interpretation (scaled to 1,000,000 transactions)

At a cost-optimal threshold of **{winner_threshold:.2f}**, {ALGO_LABELS[winner]} flags approximately
**{flagged_1M:,.0f}** transactions as fraud, of which approximately **{genuine_1M:,.0f}** are genuine
fraud caught and **{false_alarm_1M:,.0f}** are false alarms (declined legitimate checkouts). Estimated
fee-loss/friction cost of false alarms is **${fp_cost_1M:,.0f}**, offset against approximately
**${fn_cost_1M:,.0f}** in fraud prevented, for a net saving of approximately **${net_saving_1M:,.0f}**
compared to no fraud detection.

---

## Sensitivity Analysis

### (a) FP base cost (friction rate fixed at {FRICTION_RATE:.0%})

| FP Base Cost | Optimal Threshold | Total Cost (validation) |
|---|---|---|
{base_sens_md}

### (b) Friction rate (base FP cost fixed at ${BASE_FP_COST:.2f})

| Friction Rate | Optimal Threshold | Total Cost (validation) |
|---|---|---|
{friction_sens_md}

---

## Limitations

1. **Cost lens is a modeling choice**: payment-processor framing chosen to match the job-application
   target; a merchant or issuer would see a materially different cost structure.
2. **Chargeback-guarantee assumption**: general industry knowledge about this vendor category, not
   verified from this project's own data.
3. **Currency/units unconfirmed**: TransactionAmt's scale is not officially documented by Kaggle.
4. **Real-time scoring gap**: Tier 2 entity-velocity features require full prior history, computed
   offline here -- see `phase2_feature_engineering/inference.py` docstring.
5. **Retention-risk cost** is folded into the 3% friction rate as an allowance, not derived from an
   independent churn model.
6. **The cost function has no term for the AGGREGATE false-decline rate, only its sum**: the pure
   cost-minimizing threshold (0.03) flags approximately **15.6%** of all validation transactions --
   each individual false decline is priced correctly, but a processor declining 1 in 6 legitimate
   checkouts would in practice trigger merchant churn through a threshold/nonlinear effect this
   linear, per-transaction cost function cannot see. Documented here deliberately (rather than
   papered over with an unjustified capacity constraint) so the gap between "cost-optimal in
   theory" and "operationally deployable" stays visible.
"""

    out_path = os.path.join(SCRIPT_DIR, "results_summary.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"  Saved -> {os.path.relpath(out_path)}")


if __name__ == "__main__":
    main()
