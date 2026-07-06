# -*- coding: utf-8 -*-
"""
Controlled ablation for Tier 2 candidates: Phase 1 raw vs. Phase 1 + Tier 2
alone vs. the full Phase 1 + Tier 1 + Tier 2 stack -- all trained with
column subsampling disabled (feature_fraction=1.0, colsample_bytree=1.0).
See ablation_check.py's docstring for why subsampling must be disabled for
a trustworthy single-feature-set comparison in this environment.

Tier 2 features are computed on the full dataset before the train/val/test
split (see tier2_features.py) since they depend on each entity's history;
Tier 1 features are still added per-fold since they're row-independent.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import load_data, make_split
from phase1_baseline.pipeline import Preprocessor, train_lgb, train_xgb, evaluate
from phase2_feature_engineering.feature_engineering import add_tier1_features
from phase2_feature_engineering.tier2_features import add_entity_velocity_features

_TIER2_COLS = [
    "combo_prior_txn_count", "combo_time_since_last",
    "combo_amt_zscore", "addr1_changed_from_prev",
    "addr1_change_x_inverse_time", "amt_ratio_vs_prev",
]


def run_config(name, train_raw, val_raw):
    print(f"\n{'='*70}\nConfig: {name}\n{'='*70}")

    prep = Preprocessor()
    X_train = prep.fit_transform(train_raw)
    y_train = train_raw["isFraud"].values
    X_val = prep.transform(val_raw)
    y_val = val_raw["isFraud"].values
    print(f"  Columns: {X_train.shape[1]}")

    lgb_model = train_lgb(
        X_train, y_train, X_val, y_val, prep.numeric_cat_present,
        feature_fraction=1.0,
    )
    lgb_val_prob = lgb_model.predict(X_val)
    lgb_metrics = evaluate(y_val, lgb_val_prob, f"{name} LightGBM val")

    xgb_model, X_tr_xgb, X_vl_xgb = train_xgb(
        X_train, y_train, X_val, y_val, colsample_bytree=1.0,
    )
    xgb_val_prob = xgb_model.predict_proba(X_vl_xgb)[:, 1]
    xgb_metrics = evaluate(y_val, xgb_val_prob, f"{name} XGBoost val")

    print(f"\n  Top 10 LightGBM features:")
    imp = lgb_model.feature_importance(importance_type="gain")
    names = lgb_model.feature_name()
    import pandas as pd
    imp_df = pd.DataFrame({"feature": names, "importance": imp}).sort_values(
        "importance", ascending=False
    ).head(10)
    for _, row in imp_df.iterrows():
        marker = " <-- Tier 2" if row["feature"] in _TIER2_COLS else ""
        print(f"    {row['feature']:<28} {row['importance']:>12.1f}{marker}")

    return {"LightGBM": lgb_metrics, "XGBoost": xgb_metrics}


# Known from prior runs of this script (identical seed/config, so
# reproducible bit-for-bit) -- included here rather than re-training, and
# printed alongside the new v3 (refined entity key) results for a single
# combined comparison table.
_KNOWN_RESULTS = {
    "Phase 1 (raw)": {
        "LightGBM": {"roc_auc": 0.9037, "pr_auc": 0.5435},
        "XGBoost":  {"roc_auc": 0.9190, "pr_auc": 0.5697},
    },
    "Tier 2 v1 (4 feat, card combo key)": {
        "LightGBM": {"roc_auc": 0.9061, "pr_auc": 0.5287},
        "XGBoost":  {"roc_auc": 0.9139, "pr_auc": 0.5724},
    },
    "Tier 1 + Tier 2 v1": {
        "LightGBM": {"roc_auc": 0.9080, "pr_auc": 0.5561},
        "XGBoost":  {"roc_auc": 0.9182, "pr_auc": 0.5705},
    },
    "Tier 2 v2 (6 feat, card combo key)": {
        "LightGBM": {"roc_auc": 0.9061, "pr_auc": 0.5363},
        "XGBoost":  {"roc_auc": 0.9163, "pr_auc": 0.5697},
    },
    "Tier 1 + Tier 2 v2": {
        "LightGBM": {"roc_auc": 0.9051, "pr_auc": 0.5326},
        "XGBoost":  {"roc_auc": 0.9194, "pr_auc": 0.5716},
    },
}


def main():
    print("=== Tier 2 v3 Ablation (feature_fraction=1.0, colsample_bytree=1.0) ===")
    print("Same 6 features as v2, now on the D1-refined entity key (see tier2_features.py).\n")

    df = load_data()
    df = add_entity_velocity_features(df)  # full-dataset, pre-split (causal)

    train_raw, val_raw, test_raw = make_split(df)

    train_tier1_tier2 = add_tier1_features(train_raw)
    val_tier1_tier2 = add_tier1_features(val_raw)

    results = dict(_KNOWN_RESULTS)
    results["Tier 2 v3 (refined entity key)"] = run_config(
        "Tier 2 v3 (refined entity key)", train_raw, val_raw
    )
    results["Tier 1 + Tier 2 v3"] = run_config(
        "Tier 1 + Tier 2 v3", train_tier1_tier2, val_tier1_tier2
    )

    print(f"\n{'='*70}\n=== Comparison (v1/v2 known, v3 measured this run) ===\n{'='*70}")
    print(f"{'Config':<36} {'Model':<10} {'ROC-AUC':>8} {'PR-AUC':>8}")
    print("-" * 66)
    for config_name, model_results in results.items():
        for model_name, m in model_results.items():
            print(f"{config_name:<36} {model_name:<10} {m['roc_auc']:>8.4f} {m['pr_auc']:>8.4f}")


if __name__ == "__main__":
    main()
