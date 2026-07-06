# -*- coding: utf-8 -*-
"""
Phase 2: segmented model -- separate LightGBM/XGBoost pipelines for
has_identity=1 vs. has_identity=0, compared against a global (unsegmented)
baseline trained on the identical Tier 1 + Tier 2 feature set.

Motivation (see PLAN.md, "Segmented model" under Phase 2 Tier 2, and
error_analysis.py): identity presence is the dominant driver of catchability,
and segment fraud rates diverge 3.2x in train (has_identity=1: 6.73%,
has_identity=0: 2.12%). A single global model may under-serve the harder
segment because gradient boosting's limited round budget gets pulled toward
whichever population offers bigger loss reductions per split.

Every model in this script is trained with column subsampling disabled
(feature_fraction=1.0, colsample_bytree=1.0) -- the same ablation-mode
methodology adopted for Tier 2 (see PLAN.md's feature-count-sensitivity
note) -- so the global vs. segmented comparison isn't confounded by the
fixed-seed subsampling reshuffle that a differing column/row count would
otherwise trigger.

Recency weighting: segment_analysis.py showed the has_identity=1 segment's
fraud rate undergoes a regime shift partway through train (Q1 3.60% -> Q2
5.18% -> Q3 13.48% -> Q4 10.14%), and val (11.02%) continues the Q3/Q4
regime, not Q1/Q2's. Per user decision (session 7), the has_identity=1
segment's training rows get an exponential recency weight rather than a
hard cutoff, so old-regime rows are downweighted but not discarded. The
has_identity=0 segment showed no such drift and trains unweighted.
"""

import os
import json
import pickle
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
import plotly.graph_objects as go

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUT_DIR    = SCRIPT_DIR
os.makedirs(MODELS_DIR, exist_ok=True)

sys.path.insert(0, ROOT_DIR)
from utils import load_data, make_split
from phase1_baseline.pipeline import Preprocessor, train_lgb, train_xgb, evaluate
from phase2_feature_engineering.feature_engineering import add_tier1_features
from phase2_feature_engineering.tier2_features import add_entity_velocity_features

_SECONDS_IN_DAY = 86400

# Half-life for the has_identity=1 segment's recency weight. 30 days sets a
# meaningful but not extreme decay relative to this segment's ~25-day
# chronological quarters (segment_analysis.py): a row 87 days old (roughly
# Q1's midpoint) gets weight ~0.13x a row from the most recent day, while a
# row 12 days old (roughly Q4's midpoint) keeps ~0.76x -- downweighted, not
# discarded, per the user's explicit choice against a hard Q1/Q2 cutoff.
_RECENCY_HALFLIFE_DAYS = 30


def compute_recency_weights(df, halflife_days=_RECENCY_HALFLIFE_DAYS):
    """
    Exponential-decay sample weights: weight = 0.5 ** (age_days / halflife_days),
    where age_days is measured from this segment's own most recent training
    transaction (not the global train fold's), so the decay is relative to
    the population actually being weighted.

    Rescaled to mean 1.0 so LightGBM's min_child_samples / XGBoost's
    min_child_weight -- both evaluated against the sum of weights, not the
    raw row count -- keep roughly their usual meaning instead of silently
    shrinking the effective sample size.
    """
    max_dt = df["TransactionDT"].max()
    age_days = (max_dt - df["TransactionDT"]) / _SECONDS_IN_DAY
    raw_weight = 0.5 ** (age_days / halflife_days)
    return (raw_weight / raw_weight.mean()).values


def print_weight_diagnostic(df, weights):
    """Chronological-quarter breakdown of the recency weights (same quartering
    scheme as segment_analysis.py, but within this segment only) -- confirms
    the decay behaves as intended before trusting it in training."""
    order = np.argsort(df["TransactionDT"].values)
    sorted_w = weights[order]
    n = len(sorted_w)
    quarter = n // 4
    print("\n  Recency-weight diagnostic (has_identity=1 train segment, chronological quarters):")
    for i in range(4):
        start = i * quarter
        end = (i + 1) * quarter if i < 3 else n
        chunk = sorted_w[start:end]
        print(f"    Q{i+1}: n={len(chunk):>7,}  mean_weight={chunk.mean():.3f}  "
              f"min={chunk.min():.3f}  max={chunk.max():.3f}")
    print()


def run_segment(name, train_df, val_df, sample_weight=None):
    """
    Fits a Preprocessor (on train_df only) and trains LightGBM + XGBoost in
    ablation mode (no column subsampling). Used identically for the global
    baseline and each has_identity segment so all three are directly
    comparable.
    """
    n_fraud = int(train_df["isFraud"].sum())
    rate = train_df["isFraud"].mean() * 100
    print(f"\n{'='*70}\nSegment: {name}\n"
          f"  n_train={len(train_df):,}  fraud={n_fraud:,}  fraud_rate={rate:.2f}%\n{'='*70}")

    prep = Preprocessor()
    X_train = prep.fit_transform(train_df)
    y_train = train_df["isFraud"].values
    X_val = prep.transform(val_df)
    y_val = val_df["isFraud"].values
    print(f"  Columns: {X_train.shape[1]}")

    lgb_model = train_lgb(
        X_train, y_train, X_val, y_val, prep.numeric_cat_present,
        feature_fraction=1.0, sample_weight=sample_weight,
    )
    lgb_val_prob = lgb_model.predict(X_val)
    lgb_metrics = evaluate(y_val, lgb_val_prob, f"{name} LightGBM val")

    xgb_model, X_tr_xgb, X_vl_xgb = train_xgb(
        X_train, y_train, X_val, y_val,
        colsample_bytree=1.0, sample_weight=sample_weight,
    )
    xgb_val_prob = xgb_model.predict_proba(X_vl_xgb)[:, 1]
    xgb_metrics = evaluate(y_val, xgb_val_prob, f"{name} XGBoost val")

    return {
        "name": name,
        "preprocessor": prep,
        "lgb_model": lgb_model,
        "xgb_model": xgb_model,
        "y_val": y_val,
        "lgb_val_prob": lgb_val_prob,
        "xgb_val_prob": xgb_val_prob,
        "lgb_metrics": lgb_metrics,
        "xgb_metrics": xgb_metrics,
        "n_train": len(train_df),
        "n_train_fraud": n_fraud,
        "train_fraud_rate": rate,
    }


def save_feature_importance(model, model_name, segment_slug, top_n=30):
    if isinstance(model, lgb.Booster):
        imp   = model.feature_importance(importance_type="gain")
        names = model.feature_name()
    else:
        imp   = model.feature_importances_
        names = model.get_booster().feature_names

    imp_df = (
        pd.DataFrame({"feature": names, "importance": imp})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    print(f"\n  Top 15 features ({model_name}, {segment_slug}):")
    for _, row in imp_df.head(15).iterrows():
        print(f"    {row['feature']:<30} {row['importance']:>12.1f}")

    fig = go.Figure(go.Bar(x=imp_df["importance"], y=imp_df["feature"], orientation="h"))
    fig.update_layout(
        title=f"{model_name} ({segment_slug}) -- Top {top_n} Features by Gain",
        xaxis_title="Gain",
        yaxis=dict(autorange="reversed"),
        height=max(400, top_n * 20),
        margin=dict(l=220),
    )

    slug = model_name.lower().replace(" ", "_")
    out_path = os.path.join(OUT_DIR, f"feature_importance_{slug}_{segment_slug}.html")
    fig.write_html(out_path)
    data_path = os.path.join(OUT_DIR, f"feature_importance_{slug}_{segment_slug}.json")
    imp_df.to_json(data_path, orient="records", indent=2)
    print(f"  Saved: {out_path}")
    print(f"  Saved: {data_path}")


def main():
    print("=== Phase 2: Segmented Model (has_identity=1 vs. has_identity=0) ===\n")

    df = load_data()
    df["has_identity"] = df["id_01"].notnull().astype(int)
    print("\nComputing Tier 2 entity-velocity features (full dataset, causal, pre-split)...")
    df = add_entity_velocity_features(df)

    train_raw, val_raw, test_raw = make_split(df)

    print("\nAdding Tier 1 within-row features...")
    train_raw = add_tier1_features(train_raw)
    val_raw   = add_tier1_features(val_raw)
    test_raw  = add_tier1_features(test_raw)

    # ---- Global (unsegmented) baseline: same feature set, ablation config ----
    global_res = run_segment("Global (unsegmented)", train_raw, val_raw)

    # ---- Split into segments ----
    train_id1 = train_raw[train_raw["has_identity"] == 1].copy()
    train_id0 = train_raw[train_raw["has_identity"] == 0].copy()
    val_id1   = val_raw[val_raw["has_identity"] == 1].copy()
    val_id0   = val_raw[val_raw["has_identity"] == 0].copy()

    print(f"\nSegment sizes (train): has_identity=1 -> {len(train_id1):,}  "
          f"has_identity=0 -> {len(train_id0):,}")
    print(f"Segment sizes (val):   has_identity=1 -> {len(val_id1):,}  "
          f"has_identity=0 -> {len(val_id0):,}")

    weights_id1 = compute_recency_weights(train_id1)
    print_weight_diagnostic(train_id1, weights_id1)

    id1_res = run_segment("Segment has_identity=1 (recency-weighted)",
                           train_id1, val_id1, sample_weight=weights_id1)
    id0_res = run_segment("Segment has_identity=0 (no adjustment)",
                           train_id0, val_id0, sample_weight=None)

    # ---- Combine segment val predictions into one system-level metric ----
    combined_y = np.concatenate([id1_res["y_val"], id0_res["y_val"]])
    combined_lgb_prob = np.concatenate([id1_res["lgb_val_prob"], id0_res["lgb_val_prob"]])
    combined_xgb_prob = np.concatenate([id1_res["xgb_val_prob"], id0_res["xgb_val_prob"]])

    print(f"\n{'='*70}\nCombined segmented system (val)\n{'='*70}")
    combined_lgb_metrics = evaluate(combined_y, combined_lgb_prob, "Segmented (combined) LightGBM val")
    combined_xgb_metrics = evaluate(combined_y, combined_xgb_prob, "Segmented (combined) XGBoost val")

    # ---- Calibration check ----
    # is_unbalance (LightGBM) / scale_pos_weight (XGBoost) are computed per
    # segment from that segment's own class ratio -- has_identity=0's ratio
    # (~46:1) is far more extreme than has_identity=1's (~14:1). That inflates
    # each segment's raw predicted probabilities by a different amount, so a
    # combined/pooled ROC-AUC or PR-AUC over both segments' scores is not
    # comparing apples to apples even if each segment's own internal ranking
    # (its standalone ROC-AUC/PR-AUC) is fine. This check confirms whether
    # that's actually happening here before treating the pooled comparison
    # above as evidence about segmentation itself.
    print(f"\n{'='*70}\nCalibration check: mean predicted probability vs. actual fraud rate, by segment\n{'='*70}")
    for model_key, id1_prob, id0_prob in [
        ("LightGBM", id1_res["lgb_val_prob"], id0_res["lgb_val_prob"]),
        ("XGBoost",  id1_res["xgb_val_prob"], id0_res["xgb_val_prob"]),
    ]:
        print(f"  {model_key}:")
        print(f"    has_identity=1: mean_pred_prob={id1_prob.mean():.4f}  "
              f"actual_fraud_rate={id1_res['y_val'].mean():.4f}  "
              f"ratio={id1_prob.mean() / id1_res['y_val'].mean():.2f}x")
        print(f"    has_identity=0: mean_pred_prob={id0_prob.mean():.4f}  "
              f"actual_fraud_rate={id0_res['y_val'].mean():.4f}  "
              f"ratio={id0_prob.mean() / id0_res['y_val'].mean():.2f}x")

    # ---- The fair test: global model vs. segment model, scored on the SAME
    #      segment's rows, using each model's own single internally-consistent
    #      probability scale (no cross-segment pooling, so the calibration
    #      mismatch above can't contaminate this comparison). ----
    val_mask_id1 = (val_raw["has_identity"] == 1).values
    val_mask_id0 = ~val_mask_id1
    global_y = global_res["y_val"]

    print(f"\n{'='*70}\n=== Fair per-segment comparison: global model vs. segment model, "
          f"each scored only on that segment's val rows ===\n{'='*70}")
    fair_comparison = {}
    for model_key, global_prob, seg_res, mask in [
        ("LightGBM", global_res["lgb_val_prob"], id1_res, val_mask_id1),
        ("XGBoost",  global_res["xgb_val_prob"], id1_res, val_mask_id1),
        ("LightGBM", global_res["lgb_val_prob"], id0_res, val_mask_id0),
        ("XGBoost",  global_res["xgb_val_prob"], id0_res, val_mask_id0),
    ]:
        seg_label = "has_identity=1" if mask is val_mask_id1 else "has_identity=0"
        global_on_seg = evaluate(
            global_y[mask], global_prob[mask], f"Global {model_key} on {seg_label} val rows",
        )
        seg_prob = seg_res["lgb_val_prob"] if model_key == "LightGBM" else seg_res["xgb_val_prob"]
        seg_metrics = seg_res["lgb_metrics"] if model_key == "LightGBM" else seg_res["xgb_metrics"]
        print(f"    {'Segment ' + model_key + ' on ' + seg_label + ' val rows':<48} "
              f"ROC-AUC: {seg_metrics['roc_auc']:.4f}   PR-AUC: {seg_metrics['pr_auc']:.4f}")
        fair_comparison[f"{model_key}_{seg_label}"] = {
            "global_on_segment": global_on_seg, "segment_model": seg_metrics,
        }

    # ---- Comparison table: global vs. segmented-combined ----
    print(f"\n{'='*70}\n=== Global (unsegmented) vs. Segmented (combined) -- validation set ===\n{'='*70}")
    print(f"{'Model':<10} {'Metric':<8} {'Global':>8} {'Segmented':>10} {'Delta':>8}")
    print("-" * 48)
    for model_name, g_metrics, s_metrics in [
        ("LightGBM", global_res["lgb_metrics"], combined_lgb_metrics),
        ("XGBoost",  global_res["xgb_metrics"], combined_xgb_metrics),
    ]:
        for metric_key, label in [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]:
            v1, v2 = g_metrics[metric_key], s_metrics[metric_key]
            print(f"{model_name:<10} {label:<8} {v1:>8.4f} {v2:>10.4f} {v2 - v1:>+8.4f}")

    # ---- Per-segment breakdown (own local metrics, already printed inline
    #      by run_segment/evaluate -- restated here as a compact summary) ----
    print(f"\n{'='*70}\n=== Per-segment validation metrics ===\n{'='*70}")
    print(f"{'Segment':<32} {'Model':<10} {'ROC-AUC':>8} {'PR-AUC':>8}")
    print("-" * 62)
    for seg_res in (id1_res, id0_res):
        for model_name, m in [("LightGBM", seg_res["lgb_metrics"]), ("XGBoost", seg_res["xgb_metrics"])]:
            print(f"{seg_res['name']:<32} {model_name:<10} {m['roc_auc']:>8.4f} {m['pr_auc']:>8.4f}")

    # ---- Feature importance ----
    save_feature_importance(id1_res["lgb_model"], "LightGBM", "seg_id1")
    save_feature_importance(id1_res["xgb_model"], "XGBoost", "seg_id1")
    save_feature_importance(id0_res["lgb_model"], "LightGBM", "seg_id0")
    save_feature_importance(id0_res["xgb_model"], "XGBoost", "seg_id0")

    # ---- Save models + preprocessors ----
    # Global model saved too (not persisted anywhere else) -- it's the
    # Tier1+Tier2, ablation-config baseline this script compares segments
    # against, and other code (e.g. the eventual cost framework) may want it.
    global_res["preprocessor"].save(os.path.join(MODELS_DIR, "preprocessor_global_tier1tier2.pkl"))
    global_res["lgb_model"].save_model(os.path.join(MODELS_DIR, "lgb_global_tier1tier2.txt"))
    with open(os.path.join(MODELS_DIR, "xgb_global_tier1tier2.pkl"), "wb") as f:
        pickle.dump(global_res["xgb_model"], f)

    id1_res["preprocessor"].save(os.path.join(MODELS_DIR, "preprocessor_seg_id1.pkl"))
    id0_res["preprocessor"].save(os.path.join(MODELS_DIR, "preprocessor_seg_id0.pkl"))
    id1_res["lgb_model"].save_model(os.path.join(MODELS_DIR, "lgb_seg_id1.txt"))
    id0_res["lgb_model"].save_model(os.path.join(MODELS_DIR, "lgb_seg_id0.txt"))
    with open(os.path.join(MODELS_DIR, "xgb_seg_id1.pkl"), "wb") as f:
        pickle.dump(id1_res["xgb_model"], f)
    with open(os.path.join(MODELS_DIR, "xgb_seg_id0.pkl"), "wb") as f:
        pickle.dump(id0_res["xgb_model"], f)
    print("\nModels + preprocessors saved to models/*_global_tier1tier2 / *_seg_id1 / *_seg_id0.")

    # ---- Save combined metrics JSON ----
    metrics_out = {
        "recency_halflife_days": _RECENCY_HALFLIFE_DAYS,
        "global": {
            "lgb": global_res["lgb_metrics"], "xgb": global_res["xgb_metrics"],
        },
        "segment_id1": {
            "lgb": id1_res["lgb_metrics"], "xgb": id1_res["xgb_metrics"],
            "n_train": id1_res["n_train"], "n_train_fraud": id1_res["n_train_fraud"],
            "train_fraud_rate_pct": id1_res["train_fraud_rate"],
        },
        "segment_id0": {
            "lgb": id0_res["lgb_metrics"], "xgb": id0_res["xgb_metrics"],
            "n_train": id0_res["n_train"], "n_train_fraud": id0_res["n_train_fraud"],
            "train_fraud_rate_pct": id0_res["train_fraud_rate"],
        },
        "segmented_combined": {
            "lgb": combined_lgb_metrics, "xgb": combined_xgb_metrics,
        },
        "fair_per_segment_comparison": fair_comparison,
    }
    metrics_path = os.path.join(MODELS_DIR, "segmented_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Combined metrics saved to {metrics_path}")

    print("\nTest set NOT evaluated here -- touch it exactly once at final reporting.")


if __name__ == "__main__":
    main()
