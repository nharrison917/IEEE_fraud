# -*- coding: utf-8 -*-
"""
Applies the winning configs from hyperparameter_tuning.py's randomized
search to the production models, overwriting the SMOTE-1:10-default
artifacts saved by finalize_production_models.py.

Three of the four searched combos are adopted (all show a real val PR-AUC
gain, the project's primary metric):
  - Global LightGBM
  - Global XGBoost
  - has_identity=1 segment LightGBM

has_identity=1 segment XGBoost is NOT retrained here -- its best trial
(ROC -0.0005 / PR +0.0042 vs. the SMOTE-1:10 default) was judged too thin
a margin, relative to the "best of 6 trials" selection bias, to be worth
adopting. Its existing saved artifact (models/xgb_seg_id1.pkl, from
finalize_production_models.py) remains the production model.
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

_GLOBAL_LGB_CONFIG = {
    "num_leaves": 63, "min_child_samples": 20, "learning_rate": 0.1,
    "lambda_l1": 0, "lambda_l2": 1, "is_unbalance": False,
}
_GLOBAL_LGB_FEATURE_FRACTION = 0.6
_GLOBAL_LGB_BAGGING_FRACTION = 1.0  # bagging_fraction isn't a train_lgb() param -- pass via override
_GLOBAL_XGB_CONFIG = {
    "max_depth": 8, "learning_rate": 0.05, "min_child_weight": 1,
    "reg_lambda": 1, "reg_alpha": 1, "subsample": 0.6, "scale_pos_weight": 1,
}
_GLOBAL_XGB_COLSAMPLE = 1.0

_ID1_LGB_CONFIG = {
    "num_leaves": 127, "min_child_samples": 20, "learning_rate": 0.05,
    "lambda_l1": 0, "lambda_l2": 0, "is_unbalance": False,
}
_ID1_LGB_FEATURE_FRACTION = 0.8
_ID1_LGB_BAGGING_FRACTION = 0.8


def main():
    print("=== Applying tuned hyperparameters to production models ===\n")

    df = load_data()
    df["has_identity"] = df["id_01"].notnull().astype(int)
    df = add_entity_velocity_features(df)
    train_raw, val_raw, test_raw = make_split(df)
    train_raw = add_tier1_features(train_raw)
    val_raw   = add_tier1_features(val_raw)

    metrics_out = {}

    # ---- Global model ----
    print(f"\n{'='*70}\nGlobal model (tuned)\n{'='*70}")
    prep_g = Preprocessor()
    X_train_g = prep_g.fit_transform(train_raw)
    y_train_g = train_raw["isFraud"].values
    X_val_g = prep_g.transform(val_raw)
    y_val_g = val_raw["isFraud"].values
    X_res_lgb_g, X_res_xgb_g, y_res_g = apply_smotenc(X_train_g, y_train_g, prep_g, 0.10)

    lgb_g_override = dict(_GLOBAL_LGB_CONFIG)
    lgb_g_override["bagging_fraction"] = _GLOBAL_LGB_BAGGING_FRACTION
    lgb_g = train_lgb(X_res_lgb_g, y_res_g, X_val_g, y_val_g, prep_g.numeric_cat_present,
                       feature_fraction=_GLOBAL_LGB_FEATURE_FRACTION, params_override=lgb_g_override)
    lgb_g_metrics = evaluate(y_val_g, lgb_g.predict(X_val_g), "Global tuned LightGBM val")

    xgb_g, _, X_vl_xgb_g = train_xgb(X_res_xgb_g, y_res_g, X_val_g, y_val_g,
                                      colsample_bytree=_GLOBAL_XGB_COLSAMPLE,
                                      params_override=_GLOBAL_XGB_CONFIG)
    xgb_g_metrics = evaluate(y_val_g, xgb_g.predict_proba(X_vl_xgb_g)[:, 1], "Global tuned XGBoost val")

    prep_g.save(os.path.join(MODELS_DIR, "preprocessor_global_tier1tier2.pkl"))
    lgb_g.save_model(os.path.join(MODELS_DIR, "lgb_global_tier1tier2.txt"))
    with open(os.path.join(MODELS_DIR, "xgb_global_tier1tier2.pkl"), "wb") as f:
        pickle.dump(xgb_g, f)
    print("  Saved: models/*_global_tier1tier2.* (both tuned)")
    metrics_out["global"] = {"lgb": lgb_g_metrics, "xgb": xgb_g_metrics}

    # ---- has_identity=1 segment ----
    print(f"\n{'='*70}\nhas_identity=1 segment (LightGBM tuned, XGBoost unchanged)\n{'='*70}")
    train_id1 = train_raw[train_raw["has_identity"] == 1].copy()
    val_id1   = val_raw[val_raw["has_identity"] == 1].copy()

    prep_1 = Preprocessor()
    X_train_1 = prep_1.fit_transform(train_id1)
    y_train_1 = train_id1["isFraud"].values
    X_val_1 = prep_1.transform(val_id1)
    y_val_1 = val_id1["isFraud"].values
    X_res_lgb_1, X_res_xgb_1, y_res_1 = apply_smotenc(X_train_1, y_train_1, prep_1, 0.10)

    lgb_1_override = dict(_ID1_LGB_CONFIG)
    lgb_1_override["bagging_fraction"] = _ID1_LGB_BAGGING_FRACTION
    lgb_1 = train_lgb(X_res_lgb_1, y_res_1, X_val_1, y_val_1, prep_1.numeric_cat_present,
                       feature_fraction=_ID1_LGB_FEATURE_FRACTION, params_override=lgb_1_override)
    lgb_1_metrics = evaluate(y_val_1, lgb_1.predict(X_val_1), "has_identity=1 tuned LightGBM val")

    prep_1.save(os.path.join(MODELS_DIR, "preprocessor_seg_id1.pkl"))
    lgb_1.save_model(os.path.join(MODELS_DIR, "lgb_seg_id1.txt"))
    print("  Saved: models/lgb_seg_id1.txt (tuned) + preprocessor_seg_id1.pkl")
    print("  models/xgb_seg_id1.pkl left unchanged (SMOTE-1:10 default -- tuning margin too thin)")

    # Re-confirm the untouched XGBoost segment metric for the record.
    with open(os.path.join(MODELS_DIR, "xgb_seg_id1.pkl"), "rb") as f:
        xgb_1 = pickle.load(f)
    from phase1_baseline.pipeline import to_xgb
    xgb_1_metrics = evaluate(y_val_1, xgb_1.predict_proba(to_xgb(X_val_1))[:, 1],
                              "has_identity=1 (unchanged) XGBoost val")

    metrics_out["segment_id1"] = {"lgb": lgb_1_metrics, "xgb": xgb_1_metrics}

    metrics_path = os.path.join(MODELS_DIR, "production_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\nUpdated production metrics saved to {metrics_path}")
    print("\nDone. Test set NOT evaluated here -- touch it exactly once at final reporting.")


if __name__ == "__main__":
    main()
