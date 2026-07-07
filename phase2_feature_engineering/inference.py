# -*- coding: utf-8 -*-
"""
Hybrid inference pipeline: routes each transaction to the correct production
model by its has_identity flag and algorithm.

**Routing is algorithm-specific, not a single fixed architecture** (revised
this session -- see PLAN.md "Hybrid architecture re-validated end-to-end..."
for the full finding). The original "hybrid, not full segmentation" decision
was made under the pre-SMOTE, recency-weighted config. Once SMOTE 1:10 +
hyperparameter tuning were adopted for both the global and has_identity=1
segment models, re-running the fair comparison under the *current* config
showed the two algorithms now disagree:

  - XGBoost: the has_identity=1 segment model still beats the global model
    on its own subset (PR-AUC 0.8072 vs 0.8022) -- smaller margin than
    before, but still real. XGBoost keeps hybrid routing:
    has_identity==1 -> seg_id1 model, has_identity==0 -> global model.
  - LightGBM: the global model now *beats* its own has_identity=1 segment
    model on that subset (PR-AUC 0.7868 vs 0.7789) -- the extra training
    data outweighs specialization once both share the same resampling/
    tuning treatment. LightGBM always uses the global model, regardless
    of has_identity.

Each of LightGBM and XGBoost is a separate "model" choice (dashboard model
selector). The routing behavior differs between them by design -- this
isn't an oversight, it's what the val-set evidence supports for each.

**Production default is now a third choice: "ensemble"** (added this
session, after model_divergence_analysis.py showed the two algorithms
agree on only ~68% of fraud cases -- Spearman rank correlation ~0.75, not
~1.0 -- and ensemble_test.py confirmed a weighted blend of their outputs
beats either standalone algorithm on val PR-AUC, not just in theory).
`predict(df, algorithm="ensemble")` runs both algorithms (each through its
own routing above) and returns `_ENSEMBLE_WEIGHT_XGB * xgb_prob +
(1 - _ENSEMBLE_WEIGHT_XGB) * lgb_prob`. The weight (0.70) was chosen by a
0.0-1.0 grid search in steps of 0.05 on val; see
`phase2_feature_engineering/ensemble_weight_sweep.json` -- PR-AUC forms a
broad plateau (0.55-0.85 all within 0.004 of the peak), not a narrow spike,
so this isn't an overfit pick of one lucky weight.

IMPORTANT LIMITATION (documented for the write-up): Tier 2 entity-velocity
features (combo_prior_txn_count, combo_time_since_last, etc.) are causal,
expanding-window aggregates that depend on each entity's full prior
transaction history. They cannot be computed for a single transaction in
isolation -- this module's build_features() must be run on the complete
chronological dataset (or at minimum, the target rows plus their entities'
full prior history) before splitting/scoring, exactly as
finalize_production_models.py does at training time. A true real-time
scoring service would need a persisted per-entity running-state store
(last transaction time, running mean/std, last addr1) rather than
recomputing from raw history on every call -- that engineering step is
out of scope here but worth naming explicitly.
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import load_data, make_split
from phase1_baseline.pipeline import Preprocessor, to_xgb, evaluate
from phase2_feature_engineering.feature_engineering import add_tier1_features
from phase2_feature_engineering.tier2_features import add_entity_velocity_features

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

# Winning weight from ensemble_test.py's 0.0-1.0 grid search (see module
# docstring) -- share of the blend given to XGBoost's prediction.
_ENSEMBLE_WEIGHT_XGB = 0.70


def build_features(df):
    """Run on the full joined dataset, before any split -- entity velocity
    features need each entity's complete chronological history. Mirrors the
    feature order used in finalize_production_models.py / finalize_tuned_models.py."""
    df = df.copy()
    df["has_identity"] = df["id_01"].notnull().astype(int)
    df = add_entity_velocity_features(df)
    df = add_tier1_features(df)
    return df


class HybridModel:
    """Loads both production model pairs once; routes rows by has_identity."""

    def __init__(self, models_dir=MODELS_DIR):
        self.prep_global = Preprocessor.load(os.path.join(models_dir, "preprocessor_global_tier1tier2.pkl"))
        self.lgb_global = lgb.Booster(model_file=os.path.join(models_dir, "lgb_global_tier1tier2.txt"))
        with open(os.path.join(models_dir, "xgb_global_tier1tier2.pkl"), "rb") as f:
            self.xgb_global = pickle.load(f)

        self.prep_id1 = Preprocessor.load(os.path.join(models_dir, "preprocessor_seg_id1.pkl"))
        self.lgb_id1 = lgb.Booster(model_file=os.path.join(models_dir, "lgb_seg_id1.txt"))
        with open(os.path.join(models_dir, "xgb_seg_id1.pkl"), "rb") as f:
            self.xgb_id1 = pickle.load(f)

    def predict(self, df_features, algorithm="ensemble"):
        """
        df_features: dataframe that has already been through build_features()
        (has_identity, Tier 1, and Tier 2 columns all present) -- raw schema,
        pre-Preprocessor.

        algorithm: "lgb", "xgb", or "ensemble" (default -- the recommended
        production choice) -- the dashboard's model-selector axis. Routing
        behavior differs by algorithm (see module docstring):
          - "xgb": has_identity==1 -> seg_id1 model, has_identity==0 ->
            global model (hybrid routing -- segment model wins on its own
            subset for this algorithm).
          - "lgb": global model for every row, regardless of has_identity
            (global model now wins on the has_identity=1 subset too, for
            this algorithm, under the current SMOTE+tuning config).
          - "ensemble": _ENSEMBLE_WEIGHT_XGB * xgb_prob + (1 - that) *
            lgb_prob, where xgb_prob/lgb_prob are each already routed per
            their own rule above. Beats either standalone algorithm on val
            PR-AUC -- see module docstring.

        Returns a pandas Series of fraud probabilities aligned to
        df_features.index.
        """
        if algorithm not in ("lgb", "xgb", "ensemble"):
            raise ValueError(f"algorithm must be 'lgb', 'xgb', or 'ensemble', got {algorithm!r}")

        if algorithm == "ensemble":
            lgb_prob = self.predict(df_features, algorithm="lgb")
            xgb_prob = self.predict(df_features, algorithm="xgb")
            return _ENSEMBLE_WEIGHT_XGB * xgb_prob + (1 - _ENSEMBLE_WEIGHT_XGB) * lgb_prob

        probs = pd.Series(index=df_features.index, dtype=float)

        if algorithm == "lgb":
            # Global model always wins for LightGBM under the current config --
            # no routing. See module docstring.
            X_all = self.prep_global.transform(df_features)
            probs.loc[:] = self.lgb_global.predict(X_all)
            return probs

        # algorithm == "xgb": hybrid routing -- segment model still wins on
        # its own subset for this algorithm.
        mask1 = df_features["has_identity"] == 1
        mask0 = ~mask1

        if mask0.any():
            X0 = self.prep_global.transform(df_features[mask0])
            probs.loc[mask0] = self.xgb_global.predict_proba(to_xgb(X0))[:, 1]

        if mask1.any():
            X1 = self.prep_id1.transform(df_features[mask1])
            probs.loc[mask1] = self.xgb_id1.predict_proba(to_xgb(X1))[:, 1]

        return probs


def _calibration_row(y_true, y_prob, label):
    mean_prob = y_prob.mean()
    actual_rate = y_true.mean()
    ratio = mean_prob / actual_rate if actual_rate > 0 else float("nan")
    print(f"  {label:<32}  mean_pred={mean_prob:.4f}  actual_rate={actual_rate:.4f}  ratio={ratio:.2f}x")
    return {"mean_pred": mean_prob, "actual_rate": actual_rate, "ratio": ratio}


def main():
    print("=== Hybrid inference pipeline: val-set validation (test set untouched) ===\n")

    df = load_data()
    df = build_features(df)
    train_raw, val_raw, test_raw = make_split(df)
    del train_raw, test_raw  # not used here -- inference only needs already-trained models + val to validate

    model = HybridModel()

    results = {}
    for algo in ("lgb", "xgb", "ensemble"):
        routed = {
            "lgb": "global model for all rows (no routing)",
            "xgb": "hybrid routing (seg_id1 for has_identity=1)",
            "ensemble": f"{_ENSEMBLE_WEIGHT_XGB:.2f}*xgb + {1 - _ENSEMBLE_WEIGHT_XGB:.2f}*lgb (each per its own routing above)",
        }[algo]
        print(f"\n--- Algorithm: {algo.upper()} -- {routed} ---")
        probs = model.predict(val_raw, algorithm=algo)

        mask1 = val_raw["has_identity"] == 1
        mask0 = ~mask1

        id1_metrics = evaluate(val_raw.loc[mask1, "isFraud"].values, probs.loc[mask1].values,
                                "  has_identity=1 rows, as scored")
        id0_metrics = evaluate(val_raw.loc[mask0, "isFraud"].values, probs.loc[mask0].values,
                                "  has_identity=0 rows, as scored")
        overall_metrics = evaluate(val_raw["isFraud"].values, probs.values,
                                    "  Full val set, as scored (production number)")

        print("\n  Calibration check (mean predicted prob vs. actual fraud rate):")
        cal_id1 = _calibration_row(val_raw.loc[mask1, "isFraud"].values, probs.loc[mask1].values,
                                    "has_identity=1 rows")
        cal_id0 = _calibration_row(val_raw.loc[mask0, "isFraud"].values, probs.loc[mask0].values,
                                    "has_identity=0 rows")

        results[algo] = {
            "routing": routed,
            "id1_rows_as_scored": id1_metrics,
            "id0_rows_as_scored": id0_metrics,
            "full_val_as_scored": overall_metrics,
            "calibration": {"id1_rows": cal_id1, "id0_rows": cal_id0},
        }

    print("\n=== Summary: production inference metrics, full val set ===")
    print(f"{'Algorithm':<10}  {'ROC-AUC':>8}  {'PR-AUC':>8}  Routing")
    print("-" * 60)
    for algo, r in results.items():
        h = r["full_val_as_scored"]
        print(f"{algo.upper():<10}  {h['roc_auc']:>8.4f}  {h['pr_auc']:>8.4f}  {r['routing']}")

    out_path = os.path.join(MODELS_DIR, "hybrid_val_metrics.json")
    import json
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    print("Test set NOT evaluated here -- touch it exactly once at final reporting.")


if __name__ == "__main__":
    main()