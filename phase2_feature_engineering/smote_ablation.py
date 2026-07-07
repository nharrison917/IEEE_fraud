# -*- coding: utf-8 -*-
"""
SMOTE vs. class-weighting ablation for the two models that actually go to
production under the hybrid architecture (PLAN.md "Segmented model"):
the global (unsegmented) Tier1+Tier2 model, and the has_identity=1 segment
model.

Uses SMOTENC (imbalanced-learn), not plain SMOTE: plain SMOTE linearly
interpolates every column, which would corrupt one-hot dummies, LightGBM's
native categorical columns, and binary engineered flags (has_identity,
is_round_amount, D{n}_was_missing, ...) into meaningless fractional values.
SMOTENC instead takes a majority vote among a synthetic sample's nearest
neighbors for flagged categorical/binary columns, and only interpolates
genuinely continuous columns.

For the has_identity=1 segment specifically: the current production config
combines recency-weighting (sample_weight) with class-weighting
(is_unbalance/scale_pos_weight). To isolate "does SMOTE beat class-
weighting" as a single clean question rather than "does SMOTE beat
class-weighting+recency-weighting" (a confounded one), this script holds
recency-weighting OFF in both arms for that segment and only cites the
known recency-weighted production numbers as context, not as the arm SMOTE
is being compared against.

SMOTE is applied strictly after preprocessing, on the training fold only,
per PLAN.md's stated rule.
"""

import os
import sys

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import load_data, make_split
from phase1_baseline.pipeline import Preprocessor, train_lgb, train_xgb, evaluate, to_xgb
from phase2_feature_engineering.feature_engineering import add_tier1_features
from phase2_feature_engineering.tier2_features import add_entity_velocity_features

# Known from segmented_pipeline.py (session 7) -- the has_identity=1
# segment's actual production config (recency-weighted + class-weighted).
# Cited for context only; NOT the baseline this script's SMOTE arms are
# measured against (see module docstring for why).
_ID1_PRODUCTION_REFERENCE = {
    "LightGBM": {"roc_auc": 0.9314, "pr_auc": 0.7565},
    "XGBoost":  {"roc_auc": 0.9450, "pr_auc": 0.7937},
}

_SMOTE_RATIOS = {"1:10": 0.10, "1:5": 0.20, "1:3": 1 / 3}


def build_categorical_mask(prep, X_xgb):
    """
    Column indices SMOTENC should treat as categorical: LightGBM's native
    high-cardinality string columns, the numeric-categorical columns
    (card2/addr1/id_14/...), the one-hot dummy columns, and -- generically,
    so no engineered binary flag has to be named by hand -- any column with
    at most 2 distinct values in the training data. The last rule catches
    has_identity, is_round_amount, D{n}_was_missing, addr1_changed_from_prev,
    and similar 0/1 flags that would otherwise get linearly interpolated
    into meaningless fractional values by a plain-numeric SMOTE.
    """
    known_cat = set(prep.cat_columns) | set(prep.numeric_cat_present) | set(prep.ohe_train_dummies)
    binary_cols = {c for c in X_xgb.columns if X_xgb[c].nunique(dropna=False) <= 2}
    cat_names = known_cat | binary_cols
    return sorted(i for i, c in enumerate(X_xgb.columns) if c in cat_names)


def apply_smotenc(X_train, y_train, prep, sampling_strategy, seed=42):
    """
    Returns (X_res_lgb, X_res_xgb, y_res). X_res_xgb is fully numeric
    (SMOTENC's native output, on the same int-coded representation
    to_xgb() already produces). X_res_lgb restores category dtype on the
    LightGBM-native categorical columns so LightGBM still gets its
    categorical handling on the resampled data.
    """
    X_xgb = to_xgb(X_train)
    cat_idx = build_categorical_mask(prep, X_xgb)
    print(f"    SMOTENC categorical columns: {len(cat_idx)} / {X_xgb.shape[1]}")

    smote = SMOTENC(categorical_features=cat_idx, sampling_strategy=sampling_strategy,
                     random_state=seed)
    X_res_arr, y_res = smote.fit_resample(X_xgb, y_train)
    X_res_xgb = pd.DataFrame(X_res_arr, columns=X_xgb.columns)

    X_res_lgb = X_res_xgb.copy()
    for col in prep.cat_columns:
        if col not in X_res_lgb.columns:
            continue
        codes = X_res_lgb[col].round().astype(int)
        categories = prep.cat_levels[col]
        mapped = codes.map(lambda c: categories[c] if 0 <= c < len(categories) else np.nan)
        X_res_lgb[col] = pd.Categorical(mapped, categories=categories)

    return X_res_lgb, X_res_xgb, y_res


def run_baseline(name, X_train, y_train, X_val, y_val, prep):
    """Current class-weighting approach (is_unbalance / scale_pos_weight),
    no SMOTE, no recency weight -- the arm SMOTE is being compared against."""
    print(f"\n{'-'*70}\n{name}: baseline (class-weighting, no SMOTE)\n{'-'*70}")
    lgb_model = train_lgb(X_train, y_train, X_val, y_val, prep.numeric_cat_present,
                           feature_fraction=1.0)
    lgb_metrics = evaluate(y_val, lgb_model.predict(X_val), f"{name} baseline LightGBM val")

    xgb_model, X_tr_xgb, X_vl_xgb = train_xgb(X_train, y_train, X_val, y_val, colsample_bytree=1.0)
    xgb_metrics = evaluate(y_val, xgb_model.predict_proba(X_vl_xgb)[:, 1], f"{name} baseline XGBoost val")

    return {"LightGBM": lgb_metrics, "XGBoost": xgb_metrics}


def run_smote(name, ratio_label, sampling_strategy, X_train, y_train, X_val, y_val, prep):
    print(f"\n{'-'*70}\n{name}: SMOTE {ratio_label}\n{'-'*70}")
    print(f"  Original: n={len(y_train):,}  fraud={int(y_train.sum()):,}  "
          f"rate={y_train.mean()*100:.2f}%")

    X_res_lgb, X_res_xgb, y_res = apply_smotenc(X_train, y_train, prep, sampling_strategy)
    print(f"  Resampled: n={len(y_res):,}  fraud={int(y_res.sum()):,}  "
          f"rate={y_res.mean()*100:.2f}%")

    lgb_model = train_lgb(X_res_lgb, y_res, X_val, y_val, prep.numeric_cat_present,
                           feature_fraction=1.0, params_override={"is_unbalance": False})
    lgb_metrics = evaluate(y_val, lgb_model.predict(X_val),
                            f"{name} SMOTE {ratio_label} LightGBM val")

    xgb_model = None
    xgb_metrics = None
    xgb_model, X_tr_xgb_unused, X_vl_xgb = train_xgb(
        X_res_xgb, y_res, X_val, y_val, colsample_bytree=1.0,
        params_override={"scale_pos_weight": 1},
    )
    xgb_metrics = evaluate(y_val, xgb_model.predict_proba(X_vl_xgb)[:, 1],
                            f"{name} SMOTE {ratio_label} XGBoost val")

    return {"LightGBM": lgb_metrics, "XGBoost": xgb_metrics}


def print_comparison(name, baseline, smote_results, production_reference=None):
    print(f"\n{'='*70}\n=== {name}: SMOTE vs. class-weighting comparison (val) ===\n{'='*70}")
    print(f"{'Config':<28} {'Model':<10} {'ROC-AUC':>8} {'PR-AUC':>8}")
    print("-" * 58)
    for model_name in ("LightGBM", "XGBoost"):
        m = baseline[model_name]
        print(f"{'Baseline (class-weight)':<28} {model_name:<10} {m['roc_auc']:>8.4f} {m['pr_auc']:>8.4f}")
        for ratio_label, results in smote_results.items():
            m = results[model_name]
            print(f"{'SMOTE ' + ratio_label:<28} {model_name:<10} {m['roc_auc']:>8.4f} {m['pr_auc']:>8.4f}")
        if production_reference:
            m = production_reference[model_name]
            print(f"{'(FYI: current production)':<28} {model_name:<10} {m['roc_auc']:>8.4f} {m['pr_auc']:>8.4f}")
        print()


def main():
    print("=== SMOTE vs. class-weighting ablation: global + has_identity=1 segment ===\n")

    df = load_data()
    df["has_identity"] = df["id_01"].notnull().astype(int)
    df = add_entity_velocity_features(df)
    train_raw, val_raw, test_raw = make_split(df)
    train_raw = add_tier1_features(train_raw)
    val_raw   = add_tier1_features(val_raw)

    # ---- Global model ----
    print(f"\n{'#'*70}\n# GLOBAL MODEL\n{'#'*70}")
    prep_global = Preprocessor()
    X_train_g = prep_global.fit_transform(train_raw)
    y_train_g = train_raw["isFraud"].values
    X_val_g = prep_global.transform(val_raw)
    y_val_g = val_raw["isFraud"].values

    global_baseline = run_baseline("Global", X_train_g, y_train_g, X_val_g, y_val_g, prep_global)
    global_smote = {}
    for label, ratio in _SMOTE_RATIOS.items():
        global_smote[label] = run_smote("Global", label, ratio, X_train_g, y_train_g,
                                         X_val_g, y_val_g, prep_global)

    # ---- has_identity=1 segment ----
    print(f"\n{'#'*70}\n# HAS_IDENTITY=1 SEGMENT\n{'#'*70}")
    train_id1 = train_raw[train_raw["has_identity"] == 1].copy()
    val_id1   = val_raw[val_raw["has_identity"] == 1].copy()

    prep_id1 = Preprocessor()
    X_train_1 = prep_id1.fit_transform(train_id1)
    y_train_1 = train_id1["isFraud"].values
    X_val_1 = prep_id1.transform(val_id1)
    y_val_1 = val_id1["isFraud"].values

    id1_baseline = run_baseline("has_identity=1", X_train_1, y_train_1, X_val_1, y_val_1, prep_id1)
    id1_smote = {}
    for label, ratio in _SMOTE_RATIOS.items():
        id1_smote[label] = run_smote("has_identity=1", label, ratio, X_train_1, y_train_1,
                                      X_val_1, y_val_1, prep_id1)

    # ---- Comparison tables ----
    print_comparison("Global", global_baseline, global_smote)
    print_comparison("has_identity=1 segment", id1_baseline, id1_smote,
                      production_reference=_ID1_PRODUCTION_REFERENCE)

    print("\nDone. Test set NOT evaluated here -- touch it exactly once at final reporting.")


if __name__ == "__main__":
    main()
