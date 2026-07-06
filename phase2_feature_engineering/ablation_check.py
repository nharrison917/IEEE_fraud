# -*- coding: utf-8 -*-
"""
Controlled feature-ablation check: Phase 1 raw features vs. Tier 1 without
has_true_local_hour vs. Tier 1 with it -- all trained with feature/column
subsampling disabled (feature_fraction=1.0, colsample_bytree=1.0).

Why: LightGBM's feature_fraction and XGBoost's colsample_bytree randomly
subsample columns each round, seeded by a fixed seed. That seed only
reproduces the same draw for a fixed column count -- adding or removing a
single column reshuffles the entire random sampling sequence, producing
a materially different model for reasons that have nothing to do with
whether the added column carries information. Row subsampling (bagging_
fraction / subsample) is left untouched since row count is identical
across all three configs here, so it isn't a confound.

This script is diagnostic only -- it exists to answer "does this feature
actually help," not to produce the numbers that go in a final report. The
real reported models keep the tuned 0.8 subsampling for regularization.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import load_data, make_split
from phase1_baseline.pipeline import Preprocessor, to_xgb, train_lgb, train_xgb, evaluate
from phase2_feature_engineering.feature_engineering import add_tier1_features


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

    return {"LightGBM": lgb_metrics, "XGBoost": xgb_metrics}


def main():
    print("=== Controlled Feature Ablation (feature_fraction=1.0, colsample_bytree=1.0) ===")

    df = load_data()
    train_raw, val_raw, test_raw = make_split(df)

    train_tier1 = add_tier1_features(train_raw)
    val_tier1   = add_tier1_features(val_raw)

    train_tier1_no_flag = train_tier1.drop(columns=["has_true_local_hour"])
    val_tier1_no_flag   = val_tier1.drop(columns=["has_true_local_hour"])

    results = {}
    results["Phase 1 (raw)"]              = run_config("Phase 1 (raw)", train_raw, val_raw)
    results["Tier 1 (no local_hour flag)"] = run_config("Tier 1 (no local_hour flag)", train_tier1_no_flag, val_tier1_no_flag)
    results["Tier 1 (with local_hour flag)"] = run_config("Tier 1 (with local_hour flag)", train_tier1, val_tier1)

    print(f"\n{'='*70}\n=== Controlled Comparison (subsampling disabled) ===\n{'='*70}")
    print(f"{'Config':<32} {'Model':<10} {'ROC-AUC':>8} {'PR-AUC':>8}")
    print("-" * 62)
    for config_name, model_results in results.items():
        for model_name, m in model_results.items():
            print(f"{config_name:<32} {model_name:<10} {m['roc_auc']:>8.4f} {m['pr_auc']:>8.4f}")


if __name__ == "__main__":
    main()
