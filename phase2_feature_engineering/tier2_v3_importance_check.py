# -*- coding: utf-8 -*-
"""
Follow-up to tier2_ablation_check.py: re-trains just the best-performing
config (Tier 1 + Tier 2 v3, refined entity key) and prints extended
(top 30) feature importance for both models, to see directly whether the
Tier 2 features earned real split-gain in XGBoost specifically -- the
model where this config scored its best PR-AUC (0.5797) -- versus
LightGBM, where none of the Tier 2 features appeared in any top-10 list
across three feature-set versions.
"""

import os
import sys

import pandas as pd

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


def print_importance(names, imp, model_label, top_n=30):
    imp_df = pd.DataFrame({"feature": names, "importance": imp}).sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)
    imp_df["rank"] = imp_df.index + 1

    print(f"\n  Top {top_n} features ({model_label}):")
    for _, row in imp_df.head(top_n).iterrows():
        marker = " <-- Tier 2" if row["feature"] in _TIER2_COLS else ""
        print(f"    #{row['rank']:<4} {row['feature']:<28} {row['importance']:>14.4f}{marker}")

    print(f"\n  Tier 2 feature ranks (of {len(imp_df)} total features), {model_label}:")
    for col in _TIER2_COLS:
        match = imp_df[imp_df["feature"] == col]
        if len(match):
            r = match.iloc[0]
            print(f"    {col:<28} rank #{int(r['rank']):<5} importance={r['importance']:.4f}")
        else:
            print(f"    {col:<28} not found in model")


def main():
    print("=== Tier 1 + Tier 2 v3: extended feature importance check ===\n")

    df = load_data()
    df = add_entity_velocity_features(df)
    train_raw, val_raw, test_raw = make_split(df)
    train_raw = add_tier1_features(train_raw)
    val_raw = add_tier1_features(val_raw)

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
    evaluate(y_val, lgb_val_prob, "LightGBM val")
    print_importance(
        lgb_model.feature_name(), lgb_model.feature_importance(importance_type="gain"),
        "LightGBM",
    )

    xgb_model, X_tr_xgb, X_vl_xgb = train_xgb(
        X_train, y_train, X_val, y_val, colsample_bytree=1.0,
    )
    xgb_val_prob = xgb_model.predict_proba(X_vl_xgb)[:, 1]
    evaluate(y_val, xgb_val_prob, "XGBoost val")
    print_importance(
        xgb_model.get_booster().feature_names, xgb_model.feature_importances_,
        "XGBoost",
    )


if __name__ == "__main__":
    main()
