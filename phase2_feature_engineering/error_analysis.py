# -*- coding: utf-8 -*-
"""
Error analysis on the current best feature set (Tier 1 + Tier 2 v3, refined
entity key). Data-driven rather than hypothesis-driven -- everything else
this phase started from a guess and then got tested; this instead looks at
which actual fraud cases the models score lowest (the "hardest misses") and
compares them against the frauds scored highest (the "easy catches"), to
see if a pattern falls out that neither of us thought to hypothesize.

Also checks whether LightGBM and XGBoost miss the same fraud cases or
different ones -- if different, that points toward ensembling; if the same,
it points toward a shared blind spot in the feature set itself.

Uses the same feature_fraction=1.0/colsample_bytree=1.0 ablation-mode
training as the rest of Tier 2 work, since the point here is inspecting
which cases are hard for the current best feature set, not producing a
final production model.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import load_data, make_split
from phase1_baseline.pipeline import Preprocessor, train_lgb, train_xgb, evaluate
from phase2_feature_engineering.feature_engineering import add_tier1_features
from phase2_feature_engineering.tier2_features import add_entity_velocity_features

_CONTEXT_COLS = [
    "TransactionAmt", "ProductCD", "has_identity", "DeviceType",
    "card4", "card6", "hour_of_day", "is_round_amount",
    "combo_prior_txn_count", "addr1_changed_from_prev",
    "addr1_change_x_inverse_time", "amt_ratio_vs_prev",
]


def describe_group(df, label):
    print(f"\n  --- {label} (n={len(df)}) ---")
    print(f"  TransactionAmt: mean={df['TransactionAmt'].mean():.2f} "
          f"median={df['TransactionAmt'].median():.2f}")
    print(f"  has_identity rate: {df['has_identity'].mean():.3f}")
    print(f"  is_round_amount rate: {df['is_round_amount'].mean():.3f}")
    print(f"  ProductCD distribution:")
    print(df["ProductCD"].value_counts(normalize=True).round(3).to_string())
    print(f"  DeviceType distribution (of rows with identity):")
    with_id = df[df["has_identity"] == 1]
    if len(with_id):
        print(with_id["DeviceType"].value_counts(normalize=True, dropna=False).round(3).to_string())
    else:
        print("    (no identity rows in this group)")
    print(f"  combo_prior_txn_count: mean={df['combo_prior_txn_count'].mean():.1f} "
          f"median={df['combo_prior_txn_count'].median():.1f}")
    print(f"  addr1_changed_from_prev rate: {df['addr1_changed_from_prev'].mean():.3f}")


def main():
    print("=== Error Analysis: hardest misses vs. easiest catches ===\n")

    df = load_data()
    df = add_entity_velocity_features(df)
    train_raw, val_raw, test_raw = make_split(df)
    train_raw = add_tier1_features(train_raw)
    val_raw = add_tier1_features(val_raw)

    # has_identity is normally added inside Preprocessor.transform(), which
    # runs after this point -- computed directly here since it's needed for
    # the context columns before that happens.
    val_raw["has_identity"] = val_raw["id_01"].notnull().astype(int)

    prep = Preprocessor()
    X_train = prep.fit_transform(train_raw)
    y_train = train_raw["isFraud"].values
    X_val = prep.transform(val_raw)
    y_val = val_raw["isFraud"].values

    lgb_model = train_lgb(
        X_train, y_train, X_val, y_val, prep.numeric_cat_present,
        feature_fraction=1.0,
    )
    lgb_prob = lgb_model.predict(X_val)
    evaluate(y_val, lgb_prob, "LightGBM val")

    xgb_model, X_tr_xgb, X_vl_xgb = train_xgb(
        X_train, y_train, X_val, y_val, colsample_bytree=1.0,
    )
    xgb_prob = xgb_model.predict_proba(X_vl_xgb)[:, 1]
    evaluate(y_val, xgb_prob, "XGBoost val")

    context = val_raw[_CONTEXT_COLS + ["isFraud"]].reset_index(drop=True)
    context["lgb_prob"] = lgb_prob
    context["xgb_prob"] = xgb_prob

    fraud = context[context["isFraud"] == 1].copy()
    print(f"\nTotal fraud cases in val fold: {len(fraud)}")

    for model_col, model_name in [("lgb_prob", "LightGBM"), ("xgb_prob", "XGBoost")]:
        fraud_sorted = fraud.sort_values(model_col)
        n = len(fraud_sorted)
        quartile = n // 4

        hardest = fraud_sorted.iloc[:quartile]
        easiest = fraud_sorted.iloc[-quartile:]

        print(f"\n{'='*70}\n{model_name}: hardest-to-catch vs. easiest-to-catch fraud\n{'='*70}")
        print(f"  Hardest quartile score range: "
              f"{hardest[model_col].min():.4f} - {hardest[model_col].max():.4f}")
        print(f"  Easiest quartile score range: "
              f"{easiest[model_col].min():.4f} - {easiest[model_col].max():.4f}")
        describe_group(hardest, f"{model_name} HARDEST misses (bottom quartile)")
        describe_group(easiest, f"{model_name} EASIEST catches (top quartile)")

    # Overlap: do both models miss the same fraud cases?
    lgb_hardest_idx = set(fraud.sort_values("lgb_prob").iloc[:len(fraud)//4].index)
    xgb_hardest_idx = set(fraud.sort_values("xgb_prob").iloc[:len(fraud)//4].index)
    overlap = lgb_hardest_idx & xgb_hardest_idx
    print(f"\n{'='*70}\n=== Overlap between models' hardest-quartile misses ===\n{'='*70}")
    print(f"  LightGBM hardest quartile: {len(lgb_hardest_idx)} cases")
    print(f"  XGBoost hardest quartile:  {len(xgb_hardest_idx)} cases")
    print(f"  Overlap: {len(overlap)} cases "
          f"({len(overlap)/max(len(lgb_hardest_idx),1)*100:.1f}% of LightGBM's hardest, "
          f"{len(overlap)/max(len(xgb_hardest_idx),1)*100:.1f}% of XGBoost's hardest)")


if __name__ == "__main__":
    main()
