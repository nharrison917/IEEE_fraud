# -*- coding: utf-8 -*-
"""
Randomized hyperparameter search for the two production models (global,
has_identity=1 segment), now that SMOTE 1:10 is the fixed imbalance-
handling approach (see smote_ablation.py / finalize_production_models.py).

Unlike the ablation-mode scripts elsewhere in phase2_feature_engineering/
(feature_fraction=1.0/colsample_bytree=1.0, used specifically to get
trustworthy A/B comparisons between different feature sets or imbalance-
handling schemes), this script is tuning hyperparameters on one fixed,
already-decided feature set and resampling scheme -- so feature/row
subsampling (feature_fraction, bagging_fraction, subsample, colsample_bytree)
are back in the search space as legitimate regularization knobs, not a
confound.

Modest randomized search (not exhaustive grid): N_TRIALS draws per
model/algorithm from a fixed search space, selected on validation PR-AUC
(the project's primary metric). Kept small deliberately -- this is
"tune if time allows" per the project's own stated priority, not the main
event, and an exhaustive grid across 2 models x 2 algorithms would take
hours.
"""

import os
import sys
import random
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import load_data, make_split
from phase1_baseline.pipeline import Preprocessor, train_lgb, train_xgb, evaluate
from phase2_feature_engineering.feature_engineering import add_tier1_features
from phase2_feature_engineering.tier2_features import add_entity_velocity_features
from phase2_feature_engineering.smote_ablation import apply_smotenc

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
N_TRIALS = 6
_SEARCH_SEED = 7  # separate from the model training seed (42)

_LGB_SPACE = {
    "num_leaves":        [31, 63, 127, 255],
    "min_child_samples":  [20, 50, 100, 200],
    "learning_rate":      [0.03, 0.05, 0.08, 0.1],
    "lambda_l1":          [0, 0.5, 1, 2],
    "lambda_l2":          [0, 1, 2, 5],
    "feature_fraction":   [0.6, 0.8, 1.0],
    "bagging_fraction":   [0.6, 0.8, 1.0],
}
_XGB_SPACE = {
    "max_depth":          [3, 4, 5, 6, 8],
    "learning_rate":      [0.03, 0.05, 0.08, 0.1],
    "min_child_weight":   [1, 5, 10, 20],
    "reg_lambda":         [0, 1, 5, 10],
    "reg_alpha":          [0, 0.5, 1, 2],
    "subsample":          [0.6, 0.8, 1.0],
    "colsample_bytree":   [0.6, 0.8, 1.0],
}


def sample_configs(space, n, rng):
    return [{k: rng.choice(v) for k, v in space.items()} for _ in range(n)]


def tune_lgb(name, X_train, y_train, X_val, y_val, numeric_cat_cols, rng):
    configs = sample_configs(_LGB_SPACE, N_TRIALS, rng)
    results = []
    for i, cfg in enumerate(configs):
        print(f"\n  [{name} LightGBM trial {i+1}/{N_TRIALS}] {cfg}")
        override = dict(cfg)
        override["is_unbalance"] = False  # SMOTE already balances the classes
        model = train_lgb(X_train, y_train, X_val, y_val, numeric_cat_cols,
                           feature_fraction=cfg["feature_fraction"],
                           params_override=override)
        metrics = evaluate(y_val, model.predict(X_val), f"{name} LightGBM trial {i+1}")
        results.append({"config": cfg, "metrics": metrics, "model": model})
    best = max(results, key=lambda r: r["metrics"]["pr_auc"])
    return results, best


def tune_xgb(name, X_train, y_train, X_val, y_val, rng):
    configs = sample_configs(_XGB_SPACE, N_TRIALS, rng)
    results = []
    for i, cfg in enumerate(configs):
        print(f"\n  [{name} XGBoost trial {i+1}/{N_TRIALS}] {cfg}")
        override = dict(cfg)
        override["scale_pos_weight"] = 1  # SMOTE already balances the classes
        model, _, X_vl = train_xgb(X_train, y_train, X_val, y_val,
                                    colsample_bytree=cfg["colsample_bytree"],
                                    params_override=override)
        metrics = evaluate(y_val, model.predict_proba(X_vl)[:, 1], f"{name} XGBoost trial {i+1}")
        results.append({"config": cfg, "metrics": metrics, "model": model, "X_vl": X_vl})
    best = max(results, key=lambda r: r["metrics"]["pr_auc"])
    return results, best


def print_results(name, algo, results, best, current_production):
    print(f"\n{'='*70}\n=== {name} {algo}: hyperparameter search results ===\n{'='*70}")
    print(f"(FYI: current production, default hyperparameters) "
          f"ROC-AUC {current_production['roc_auc']:.4f}  PR-AUC {current_production['pr_auc']:.4f}")
    for i, r in enumerate(results):
        marker = "  <-- best" if r is best else ""
        print(f"  Trial {i+1}: ROC-AUC {r['metrics']['roc_auc']:.4f}  "
              f"PR-AUC {r['metrics']['pr_auc']:.4f}  {r['config']}{marker}")


def run_model(name, train_df, val_df, current_production):
    prep = Preprocessor()
    X_train = prep.fit_transform(train_df)
    y_train = train_df["isFraud"].values
    X_val = prep.transform(val_df)
    y_val = val_df["isFraud"].values

    X_res_lgb, X_res_xgb, y_res = apply_smotenc(X_train, y_train, prep, 0.10)
    print(f"  Resampled (SMOTE 1:10): n={len(y_res):,}  fraud={int(y_res.sum()):,}")

    rng_lgb = random.Random(_SEARCH_SEED)
    lgb_results, lgb_best = tune_lgb(name, X_res_lgb, y_res, X_val, y_val,
                                      prep.numeric_cat_present, rng_lgb)
    print_results(name, "LightGBM", lgb_results, lgb_best, current_production["lgb"])

    rng_xgb = random.Random(_SEARCH_SEED)
    xgb_results, xgb_best = tune_xgb(name, X_res_xgb, y_res, X_val, y_val, rng_xgb)
    print_results(name, "XGBoost", xgb_results, xgb_best, current_production["xgb"])

    return {
        "lgb_best_config": lgb_best["config"], "lgb_best_metrics": lgb_best["metrics"],
        "xgb_best_config": xgb_best["config"], "xgb_best_metrics": xgb_best["metrics"],
    }


def main():
    print("=== Hyperparameter tuning on top of SMOTE 1:10 (both production models) ===\n")

    with open(os.path.join(MODELS_DIR, "production_metrics.json")) as f:
        current_production = json.load(f)

    df = load_data()
    df["has_identity"] = df["id_01"].notnull().astype(int)
    df = add_entity_velocity_features(df)
    train_raw, val_raw, test_raw = make_split(df)
    train_raw = add_tier1_features(train_raw)
    val_raw   = add_tier1_features(val_raw)

    print(f"\n{'#'*70}\n# GLOBAL MODEL\n{'#'*70}")
    global_summary = run_model("Global", train_raw, val_raw, current_production["global"])

    train_id1 = train_raw[train_raw["has_identity"] == 1].copy()
    val_id1   = val_raw[val_raw["has_identity"] == 1].copy()
    print(f"\n{'#'*70}\n# HAS_IDENTITY=1 SEGMENT\n{'#'*70}")
    id1_summary = run_model("has_identity=1", train_id1, val_id1, current_production["segment_id1"])

    summary_out = {"global": global_summary, "segment_id1": id1_summary}
    out_path = os.path.join(MODELS_DIR, "hyperparameter_search_results.json")
    with open(out_path, "w") as f:
        json.dump(summary_out, f, indent=2, default=str)
    print(f"\nBest configs + metrics saved to {out_path}")
    print("\nDone. Test set NOT evaluated here -- touch it exactly once at final reporting.")


if __name__ == "__main__":
    main()
