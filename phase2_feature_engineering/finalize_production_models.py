# -*- coding: utf-8 -*-
"""
Retrains and saves the final production models decided in smote_ablation.py
(session 7 continued): SMOTE 1:10 (SMOTENC), no class-weighting, no
recency-weighting, for both the global model and the has_identity=1
segment model. Replaces the earlier class-weighted (global) and
recency-weighted (has_identity=1) artifacts saved by segmented_pipeline.py.

Why the switch: smote_ablation.py showed SMOTE 1:10 beats class-weighting
alone for the global model (LightGBM PR-AUC 0.5289->0.5852, XGBoost
0.5797->0.6072), and beats the has_identity=1 segment's previous
recency-weighted config outright on every metric (recency-weighting,
checked in isolation for the first time in that script, turned out to be
a wash-to-slightly-negative rather than the improvement originally
assumed). See PLAN.md's "SMOTE ablation" section for full numbers and
reasoning.

has_identity=0 is unaffected -- it already uses the global model in the
hybrid architecture, so retraining the global model here IS its production
model too. The has_identity=0-specific segment artifacts remain the
untouched, non-adopted artifacts from segmented_pipeline.py.
"""

import os
import sys
import json
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import load_data, make_split
from phase1_baseline.pipeline import Preprocessor, train_lgb, train_xgb, evaluate
from phase2_feature_engineering.feature_engineering import add_tier1_features
from phase2_feature_engineering.tier2_features import add_entity_velocity_features
from phase2_feature_engineering.smote_ablation import apply_smotenc

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
_SAMPLING_STRATEGY = 0.10  # 1:10 minority:majority -- the winning ratio


def train_and_save(name, train_df, val_df, model_prefix):
    print(f"\n{'='*70}\n{name}\n{'='*70}")

    prep = Preprocessor()
    X_train = prep.fit_transform(train_df)
    y_train = train_df["isFraud"].values
    X_val = prep.transform(val_df)
    y_val = val_df["isFraud"].values
    print(f"  Original: n={len(y_train):,}  fraud={int(y_train.sum()):,}  "
          f"rate={y_train.mean()*100:.2f}%")

    X_res_lgb, X_res_xgb, y_res = apply_smotenc(X_train, y_train, prep, _SAMPLING_STRATEGY)
    print(f"  Resampled (SMOTE 1:10): n={len(y_res):,}  fraud={int(y_res.sum()):,}  "
          f"rate={y_res.mean()*100:.2f}%")

    lgb_model = train_lgb(X_res_lgb, y_res, X_val, y_val, prep.numeric_cat_present,
                           feature_fraction=1.0, params_override={"is_unbalance": False})
    lgb_metrics = evaluate(y_val, lgb_model.predict(X_val), f"{name} LightGBM val")

    xgb_model, _, X_vl_xgb = train_xgb(X_res_xgb, y_res, X_val, y_val, colsample_bytree=1.0,
                                        params_override={"scale_pos_weight": 1})
    xgb_metrics = evaluate(y_val, xgb_model.predict_proba(X_vl_xgb)[:, 1], f"{name} XGBoost val")

    prep.save(os.path.join(MODELS_DIR, f"preprocessor_{model_prefix}.pkl"))
    lgb_model.save_model(os.path.join(MODELS_DIR, f"lgb_{model_prefix}.txt"))
    with open(os.path.join(MODELS_DIR, f"xgb_{model_prefix}.pkl"), "wb") as f:
        pickle.dump(xgb_model, f)
    print(f"  Saved: models/*_{model_prefix}.*")

    return {"lgb": lgb_metrics, "xgb": xgb_metrics}


def main():
    print("=== Finalizing production models: SMOTE 1:10, both models ===\n")

    df = load_data()
    df["has_identity"] = df["id_01"].notnull().astype(int)
    df = add_entity_velocity_features(df)
    train_raw, val_raw, test_raw = make_split(df)
    train_raw = add_tier1_features(train_raw)
    val_raw   = add_tier1_features(val_raw)

    global_metrics = train_and_save("Global (SMOTE 1:10)", train_raw, val_raw,
                                     "global_tier1tier2")

    train_id1 = train_raw[train_raw["has_identity"] == 1].copy()
    val_id1   = val_raw[val_raw["has_identity"] == 1].copy()
    id1_metrics = train_and_save("has_identity=1 segment (SMOTE 1:10)", train_id1, val_id1,
                                  "seg_id1")

    metrics_out = {
        "imbalance_handling": "SMOTE 1:10 (SMOTENC), no class-weighting, no recency-weighting",
        "global": global_metrics,
        "segment_id1": id1_metrics,
    }
    metrics_path = os.path.join(MODELS_DIR, "production_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\nProduction metrics saved to {metrics_path}")
    print("\nTest set NOT evaluated here -- touch it exactly once at final reporting.")


if __name__ == "__main__":
    main()
