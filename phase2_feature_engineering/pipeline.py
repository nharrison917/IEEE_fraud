# -*- coding: utf-8 -*-
"""
Phase 2 Tier 1: Phase 1 preprocessing + within-row feature engineering.

Reuses Preprocessor, to_xgb, train_lgb, train_xgb, and evaluate from
phase1_baseline.pipeline unchanged -- only the feature set changes.
Tier 1 features are added to each fold's raw data before the Preprocessor
runs, so the Preprocessor's fit-on-train-only contract still holds and
the new columns get the same missingness/encoding treatment as everything
else.

Outputs are written under phase2_feature_engineering/ and models/*_phase2_*
to avoid overwriting the Phase 1 baseline artifacts.
"""

import os
import json
import pickle
import sys

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
from phase1_baseline.pipeline import Preprocessor, to_xgb, train_lgb, train_xgb, evaluate
from phase2_feature_engineering.feature_engineering import add_tier1_features


def plot_feature_importance(model, model_name, top_n=30):
    """Same as phase1_baseline.pipeline.plot_feature_importance, but writes
    into this directory instead of phase1_baseline/ to avoid overwriting
    the Phase 1 baseline plots."""
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

    print(f"\n  Top {min(15, top_n)} features ({model_name}):")
    for _, row in imp_df.head(15).iterrows():
        print(f"    {row['feature']:<30} {row['importance']:>12.1f}")

    fig = go.Figure(go.Bar(
        x=imp_df["importance"],
        y=imp_df["feature"],
        orientation="h",
    ))
    fig.update_layout(
        title=f"{model_name} (Phase 2 Tier 1) -- Top {top_n} Features by Gain",
        xaxis_title="Gain",
        yaxis=dict(autorange="reversed"),
        height=max(400, top_n * 20),
        margin=dict(l=220),
    )

    slug = model_name.lower().replace(" ", "_")
    out_path = os.path.join(OUT_DIR, f"feature_importance_{slug}_phase2.html")
    fig.write_html(out_path)

    data_path = os.path.join(OUT_DIR, f"feature_importance_{slug}_phase2.json")
    imp_df.to_json(data_path, orient="records", indent=2)

    print(f"  Saved: {out_path}")
    print(f"  Saved: {data_path}")


def print_comparison(phase2_metrics):
    """Print Phase 1 vs Phase 2 val metrics side by side, reading Phase 1's
    saved metrics JSON so the two runs never need to be in the same process."""
    phase1_paths = {
        "LightGBM": os.path.join(MODELS_DIR, "lgb_metrics.json"),
        "XGBoost":  os.path.join(MODELS_DIR, "xgb_metrics.json"),
    }

    print("\n=== Phase 1 Baseline vs Phase 2 Tier 1 (validation set) ===")
    print(f"{'Model':<10} {'Metric':<8} {'Phase1':>8} {'Phase2':>8} {'Delta':>8}")
    print("-" * 46)
    for name, path in phase1_paths.items():
        if not os.path.exists(path):
            print(f"{name:<10}  (no Phase 1 metrics found at {path})")
            continue
        with open(path) as f:
            p1 = json.load(f)["val"]
        p2 = phase2_metrics[name]["val"]
        for metric_key, label in [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]:
            v1, v2 = p1[metric_key], p2[metric_key]
            print(f"{name:<10} {label:<8} {v1:>8.4f} {v2:>8.4f} {v2 - v1:>+8.4f}")


def main():
    print("=== Phase 2 Tier 1 Pipeline (within-row feature engineering) ===\n")

    # Load, split, then add Tier 1 features to each fold independently.
    # Every Tier 1 feature is a deterministic per-row function -- no fitting,
    # so no leakage risk from doing this before or after the split.
    df = load_data()
    train_raw, val_raw, test_raw = make_split(df)

    print("\nAdding Tier 1 within-row features...")
    train_raw = add_tier1_features(train_raw)
    val_raw   = add_tier1_features(val_raw)
    test_raw  = add_tier1_features(test_raw)
    new_cols = [
        "hour_of_day", "day_of_week", "local_hour", "log1p_TransactionAmt",
        "is_round_amount", "P_email_matches_R_email", "P_email_is_free",
        "browser_name", "os_name",
    ]
    print(f"  Added columns: {new_cols}")

    # Preprocess -- fit on train only, same Preprocessor as Phase 1.
    print("\nFitting preprocessor on training fold...")
    prep    = Preprocessor()
    X_train = prep.fit_transform(train_raw)
    y_train = train_raw["isFraud"].values

    X_val = prep.transform(val_raw)
    y_val = val_raw["isFraud"].values

    X_test = prep.transform(test_raw)
    y_test = test_raw["isFraud"].values

    print(f"\n  X_train shape : {X_train.shape}")
    n_str_cat = len(X_train.select_dtypes("category").columns)
    print(f"  String cat cols (LightGBM native)    : {n_str_cat}")
    print(f"  Numeric cat cols (categorical_feature): {len(prep.numeric_cat_present)}")
    print(f"  OHE dummy cols                        : {len(prep.ohe_train_dummies)}")

    prep.save(os.path.join(MODELS_DIR, "preprocessor_phase2.pkl"))
    print("  Preprocessor saved to models/preprocessor_phase2.pkl")

    # LightGBM
    lgb_model = train_lgb(X_train, y_train, X_val, y_val, prep.numeric_cat_present)

    print("\nEvaluating LightGBM...")
    lgb_val_prob   = lgb_model.predict(X_val)
    lgb_train_prob = lgb_model.predict(X_train)
    lgb_metrics = {
        "val":   evaluate(y_val,   lgb_val_prob,   "LightGBM val"),
        "train": evaluate(y_train, lgb_train_prob,  "LightGBM train"),
    }

    lgb_model.save_model(os.path.join(MODELS_DIR, "lgb_phase2.txt"))
    with open(os.path.join(MODELS_DIR, "lgb_phase2_metrics.json"), "w") as f:
        json.dump(lgb_metrics, f, indent=2)

    plot_feature_importance(lgb_model, "LightGBM")

    # XGBoost
    xgb_model, X_tr_xgb, X_vl_xgb = train_xgb(X_train, y_train, X_val, y_val)

    print("\nEvaluating XGBoost...")
    xgb_val_prob   = xgb_model.predict_proba(X_vl_xgb)[:, 1]
    xgb_train_prob = xgb_model.predict_proba(X_tr_xgb)[:, 1]
    xgb_metrics = {
        "val":   evaluate(y_val,   xgb_val_prob,   "XGBoost val"),
        "train": evaluate(y_train, xgb_train_prob,  "XGBoost train"),
    }

    with open(os.path.join(MODELS_DIR, "xgb_phase2.pkl"), "wb") as f:
        pickle.dump(xgb_model, f)
    with open(os.path.join(MODELS_DIR, "xgb_phase2_metrics.json"), "w") as f:
        json.dump(xgb_metrics, f, indent=2)

    plot_feature_importance(xgb_model, "XGBoost")

    # Summary
    print("\n=== Phase 2 Tier 1 Validation Results ===")
    print(f"{'Model':<15}  {'ROC-AUC':>8}  {'PR-AUC':>8}")
    print("-" * 38)
    for name, m in [("LightGBM", lgb_metrics), ("XGBoost", xgb_metrics)]:
        print(f"{name:<15}  {m['val']['roc_auc']:>8.4f}  {m['val']['pr_auc']:>8.4f}")

    print_comparison({"LightGBM": lgb_metrics, "XGBoost": xgb_metrics})

    print("\nTrain-set scores saved to models/*_phase2_metrics.json for overfitting check.")
    print("Test set NOT evaluated here -- touch it exactly once at final reporting.")


if __name__ == "__main__":
    main()
