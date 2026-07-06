# -*- coding: utf-8 -*-
"""
Investigates why the has_identity=0 segment model (segmented_pipeline.py)
underperformed the global model on its own fair per-segment comparison for
one model/metric combo (XGBoost, both metrics) and split for LightGBM
(PR-AUC up, ROC-AUC down) -- see PLAN.md's Segmented Model results.

Hypothesis: has_identity=0 has fewer fraud examples (5,440 vs. the global
model's 11,988) at the same model complexity (127 leaves / depth 6, tuned
implicitly for the full 354k-row dataset), so the dedicated segment model
overfits more before val performance saturates. Both models' XGBoost runs
in segmented_pipeline.py showed a large train/val gap for this segment
(train AUCPR ~0.92 vs. val AUCPR ~0.235 at the final round) -- consistent
with, though not proof of, overfitting rather than a genuine information
deficit.

This script re-fits the has_identity=0 Preprocessor once, then trains
several progressively more regularized LightGBM/XGBoost configs on the
identical data, comparing each against the global model's own fair
baseline on has_identity=0 val rows (loaded from disk, not retrained, to
save time -- see segmented_pipeline.py for how that baseline was produced).
"""

import os
import sys

import numpy as np
import lightgbm as lgb
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import load_data, make_split
from phase1_baseline.pipeline import Preprocessor, train_lgb, train_xgb, evaluate, to_xgb
from phase2_feature_engineering.feature_engineering import add_tier1_features
from phase2_feature_engineering.tier2_features import add_entity_velocity_features

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

# Known from segmented_pipeline.py's fair per-segment comparison (same seed/
# config, reproducible) -- the bar the tuned id0 segment configs need to beat.
_GLOBAL_ON_ID0 = {
    "LightGBM": {"roc_auc": 0.8609, "pr_auc": 0.2210},
    "XGBoost":  {"roc_auc": 0.8815, "pr_auc": 0.2451},
}
_BASELINE_SEGMENT_ID0 = {
    "LightGBM": {"roc_auc": 0.8489, "pr_auc": 0.2613},
    "XGBoost":  {"roc_auc": 0.8471, "pr_auc": 0.2359},
}


def main():
    print("=== has_identity=0 segment: regularization tuning ===\n")

    df = load_data()
    df["has_identity"] = df["id_01"].notnull().astype(int)
    df = add_entity_velocity_features(df)
    train_raw, val_raw, test_raw = make_split(df)

    train_raw = add_tier1_features(train_raw)
    val_raw   = add_tier1_features(val_raw)

    train_id0 = train_raw[train_raw["has_identity"] == 0].copy()
    val_id0   = val_raw[val_raw["has_identity"] == 0].copy()
    print(f"\nhas_identity=0 -- train: {len(train_id0):,} (fraud={train_id0['isFraud'].sum():,}, "
          f"{train_id0['isFraud'].mean()*100:.2f}%)  val: {len(val_id0):,}")

    prep = Preprocessor()
    X_train = prep.fit_transform(train_id0)
    y_train = train_id0["isFraud"].values
    X_val = prep.transform(val_id0)
    y_val = val_id0["isFraud"].values
    print(f"Columns: {X_train.shape[1]}")

    results = {"Global (on id0 rows)": _GLOBAL_ON_ID0,
               "Segment baseline (127 leaves/depth 6)": _BASELINE_SEGMENT_ID0}

    # ---- LightGBM configs: progressively simpler / more regularized ----
    lgb_configs = {
        "LGB moderate reg (63 leaves, min_child=100, L2=1)": {
            "num_leaves": 63, "min_child_samples": 100, "lambda_l2": 1.0,
        },
        "LGB strong reg (31 leaves, min_child=200, L1=1, L2=5)": {
            "num_leaves": 31, "min_child_samples": 200, "lambda_l1": 1.0, "lambda_l2": 5.0,
        },
    }
    lgb_results = {}
    for name, override in lgb_configs.items():
        print(f"\n{'='*70}\n{name}\n{'='*70}")
        model = train_lgb(X_train, y_train, X_val, y_val, prep.numeric_cat_present,
                           feature_fraction=1.0, params_override=override)
        prob = model.predict(X_val)
        lgb_results[name] = evaluate(y_val, prob, name)

    # ---- XGBoost configs: progressively simpler / more regularized ----
    xgb_configs = {
        "XGB moderate reg (depth=4, min_child_weight=10, L2=5)": {
            "max_depth": 4, "min_child_weight": 10, "reg_lambda": 5,
        },
        "XGB strong reg (depth=3, min_child_weight=20, L2=10)": {
            "max_depth": 3, "min_child_weight": 20, "reg_lambda": 10,
        },
    }
    xgb_results = {}
    for name, override in xgb_configs.items():
        print(f"\n{'='*70}\n{name}\n{'='*70}")
        model, X_tr_xgb, X_vl_xgb = train_xgb(X_train, y_train, X_val, y_val,
                                               colsample_bytree=1.0, params_override=override)
        prob = model.predict_proba(X_vl_xgb)[:, 1]
        xgb_results[name] = evaluate(y_val, prob, name)

    # ---- Comparison table ----
    print(f"\n{'='*70}\n=== has_identity=0 segment: tuning comparison (val) ===\n{'='*70}")
    print(f"{'Config':<52} {'ROC-AUC':>8} {'PR-AUC':>8}")
    print("-" * 70)
    print(f"{'Global (on id0 rows)':<52} {_GLOBAL_ON_ID0['LightGBM']['roc_auc']:>8.4f} "
          f"{_GLOBAL_ON_ID0['LightGBM']['pr_auc']:>8.4f}  <- LightGBM reference")
    print(f"{'Segment baseline (127 leaves)':<52} {_BASELINE_SEGMENT_ID0['LightGBM']['roc_auc']:>8.4f} "
          f"{_BASELINE_SEGMENT_ID0['LightGBM']['pr_auc']:>8.4f}  <- LightGBM baseline")
    for name, m in lgb_results.items():
        print(f"{name:<52} {m['roc_auc']:>8.4f} {m['pr_auc']:>8.4f}")
    print()
    print(f"{'Global (on id0 rows)':<52} {_GLOBAL_ON_ID0['XGBoost']['roc_auc']:>8.4f} "
          f"{_GLOBAL_ON_ID0['XGBoost']['pr_auc']:>8.4f}  <- XGBoost reference")
    print(f"{'Segment baseline (depth 6)':<52} {_BASELINE_SEGMENT_ID0['XGBoost']['roc_auc']:>8.4f} "
          f"{_BASELINE_SEGMENT_ID0['XGBoost']['pr_auc']:>8.4f}  <- XGBoost baseline")
    for name, m in xgb_results.items():
        print(f"{name:<52} {m['roc_auc']:>8.4f} {m['pr_auc']:>8.4f}")


if __name__ == "__main__":
    main()
