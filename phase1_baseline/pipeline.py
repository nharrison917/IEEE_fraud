# -*- coding: utf-8 -*-
"""
Phase 1 baseline: preprocessing + LightGBM + XGBoost.

Preprocessor is fitted on training data only and applied identically to val
and test.  No feature engineering -- raw preprocessed features only.
Test set is transformed here but NOT evaluated; touch it exactly once at
final reporting.
"""

import os
import re
import json
import pickle
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils import load_data, make_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUT_DIR    = SCRIPT_DIR
os.makedirs(MODELS_DIR, exist_ok=True)

_D_RE = re.compile(r"^D\d+$")

# Numeric columns that encode category codes, not ordinal magnitudes.
# They are filled with -999 for missing and handed to LightGBM via
# categorical_feature.  XGBoost receives them as plain integers
# (documented limitation -- it has no equivalent native mechanism here).
_NUMERIC_CAT = frozenset({
    "card2", "card3", "card5",          # card-number fragments (100-499 codes)
    "addr1",                             # billing region code (308 unique ints)
    "id_13", "id_14",                    # id_14 = timezone offset in minutes
    "id_17", "id_18", "id_19", "id_20",
    "id_21", "id_22", "id_24", "id_25", "id_26",
    "id_32",                             # screen colour depth
})

# String columns with this many unique values or fewer get one-hot encoded.
# Above this count they become LightGBM native categoricals to avoid
# the target-encoding leakage risk documented in the plan.
_OHE_THRESH = 10

# Columns that are identifiers or labels -- never features.
_DROP = frozenset({"TransactionID", "TransactionDT", "isFraud"})


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class Preprocessor:
    """
    Fit on training data, transform all folds.

    LightGBM receives the returned DataFrame directly (auto-detects pandas
    category dtype; numeric categoricals passed via categorical_feature param).

    XGBoost: call to_xgb(X) to convert category dtype columns to integer
    codes before passing to the model.
    """

    def __init__(self):
        self.d_cols              = []
        self.card_medians        = {}   # card1/2/3/5 -> median fill
        self.card_modes          = {}   # card4/6 -> mode fill
        self.sentinel_cols       = []   # numeric cols to fill -999
        self.ohe_columns         = []   # low-card string cols -> one-hot
        self.ohe_train_dummies   = []   # dummy column names from training OHE
        self.cat_columns         = []   # high-card string cols -> category dtype
        self.cat_levels          = {}   # col -> list of known category levels
        self.numeric_cat_present = []   # _NUMERIC_CAT cols found in the data

    # ------------------------------------------------------------------
    def fit(self, train):
        df = train

        # D columns
        self.d_cols = sorted(c for c in df.columns if _D_RE.match(c))

        # Card fills
        for col in ("card1", "card2", "card3", "card5"):
            if col in df.columns:
                self.card_medians[col] = df[col].median()
        for col in ("card4", "card6"):
            if col in df.columns:
                mode = df[col].mode()
                self.card_modes[col] = mode.iloc[0] if len(mode) else "missing"

        # Numeric cols needing -999 fill.
        # Excludes: card cols (median/mode above), _NUMERIC_CAT (filled in
        # their own step), D cols (missingness indicator + fill), and the
        # drop-always set.
        _handled = set(self.card_medians) | set(self.card_modes) | _NUMERIC_CAT | _DROP
        self.sentinel_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
            and c not in _handled
            and not _D_RE.match(c)
            and df[c].isnull().any()
        ]

        # Classify string columns: OHE (low-cardinality) vs native categorical.
        # Uses is_numeric_dtype negation rather than dtype == object so that
        # pandas StringDtype columns (shown as 'str' in some pandas 2.x builds)
        # are caught alongside classic numpy object dtype columns.
        str_cols = [
            c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c])
            and c not in _DROP
        ]
        for col in str_cols:
            # Count unique values including the 'missing' state we will add
            n = df[col].nunique() + int(df[col].isnull().any())
            if n <= _OHE_THRESH:
                self.ohe_columns.append(col)
            else:
                self.cat_columns.append(col)

        # Capture category levels for high-cardinality string cols so val/test
        # get exactly the same category set (unseen values map to NaN, which
        # LightGBM handles natively).
        for col in self.cat_columns:
            levels = list(df[col].fillna("missing").unique())
            if "missing" not in levels:
                levels.append("missing")
            self.cat_levels[col] = levels

        # OHE dry run on training data to capture the exact dummy column names.
        # Val/test will be reindexed to this schema (fill_value=0 for unseen
        # categories).
        ohe_present = [c for c in self.ohe_columns if c in df.columns]
        if ohe_present:
            ohe_df = df[ohe_present].fillna("missing")
            raw_dummies = pd.get_dummies(ohe_df, prefix_sep="__")
            # Sanitize: LightGBM rejects special JSON characters in feature names
            # (e.g. colons from id_23 proxy type values like "IP_PROXY:TRANSPARENT")
            self.ohe_train_dummies = [
                re.sub(r"[^A-Za-z0-9_]", "_", c) for c in raw_dummies.columns
            ]

        self.numeric_cat_present = [c for c in _NUMERIC_CAT if c in df.columns]

    # ------------------------------------------------------------------
    def transform(self, df):
        df = df.copy()

        # 1. has_identity flag -- added before any imputation so it reflects
        #    true absence of identity data, not an imputed value.
        if "id_01" in df.columns:
            df["has_identity"] = df["id_01"].notnull().astype(int)

        # 2. addr2 -> is_us binary.
        #    99% of non-null values are 87 (US country code); near-zero variance
        #    otherwise.  Converting preserves the 1% non-US signal cleanly.
        if "addr2" in df.columns:
            df["is_us"] = (df["addr2"] == 87).astype(int)
            df.drop(columns=["addr2"], inplace=True)

        # 3. D columns: missing indicator (absence = "no prior event") then fill.
        for col in self.d_cols:
            if col in df.columns:
                df[f"{col}_was_missing"] = df[col].isnull().astype(int)
                df[col] = df[col].fillna(-999)

        # 4. Card fills.
        for col, val in self.card_medians.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        for col, val in self.card_modes.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)

        # 5. Sentinel -999 fill for V, id_ numeric, dist, and remaining numerics.
        for col in self.sentinel_cols:
            if col in df.columns:
                df[col] = df[col].fillna(-999)

        # 6. Numeric categorical columns: fill -999 for missing.
        for col in self.numeric_cat_present:
            if col in df.columns:
                df[col] = df[col].fillna(-999)

        # 7. OHE for low-cardinality string columns.
        ohe_present = [c for c in self.ohe_columns if c in df.columns]
        if ohe_present:
            ohe_df = df[ohe_present].fillna("missing")
            dummies = pd.get_dummies(ohe_df, prefix_sep="__")
            dummies.columns = [re.sub(r"[^A-Za-z0-9_]", "_", c) for c in dummies.columns]
            # Reindex to training schema; unseen categories get 0 in all columns.
            dummies = dummies.reindex(columns=self.ohe_train_dummies, fill_value=0)
            df.drop(columns=ohe_present, inplace=True)
            df = pd.concat(
                [df.reset_index(drop=True), dummies.reset_index(drop=True)],
                axis=1,
            )

        # 8. High-cardinality string columns -> pandas category dtype.
        #    LightGBM auto-detects category dtype and handles them natively,
        #    avoiding target-encoding leakage entirely.
        for col in self.cat_columns:
            if col in df.columns:
                df[col] = df[col].fillna("missing")
                # CategoricalDtype.astype converts OOV values to NaN silently;
                # pd.Categorical(...) raises a Pandas4Warning for the same case.
                df[col] = df[col].astype(pd.CategoricalDtype(categories=self.cat_levels[col]))

        # 9. Drop identifiers and label.
        drop_cols = [c for c in _DROP if c in df.columns]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)

        return df

    # ------------------------------------------------------------------
    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)


def to_xgb(X):
    """Convert pandas category columns to integer codes for XGBoost."""
    X = X.copy()
    for col in X.select_dtypes("category").columns:
        X[col] = X[col].cat.codes  # -1 for values not seen in training
    return X


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lgb(X_train, y_train, X_val, y_val, numeric_cat_cols, feature_fraction=0.8,
              sample_weight=None, params_override=None):
    print("\nTraining LightGBM...")

    # Combine category-dtype cols (auto-detected) with the numeric categorical
    # cols that LightGBM needs to be told about explicitly.
    cat_features = (
        list(X_train.select_dtypes("category").columns)
        + [c for c in numeric_cat_cols if c in X_train.columns]
    )

    dtrain = lgb.Dataset(X_train, label=y_train, weight=sample_weight,
                         categorical_feature=cat_features)
    dval   = lgb.Dataset(X_val,   label=y_val,   categorical_feature=cat_features,
                         reference=dtrain)

    params = {
        "objective":          "binary",
        "metric":             "auc",
        "learning_rate":      0.05,
        "num_leaves":         127,
        "min_child_samples":  50,
        "feature_fraction":   feature_fraction,
        "bagging_fraction":   0.8,
        "bagging_freq":       5,
        "is_unbalance":       True,  # equivalent to class_weight='balanced'
        "verbose":            -1,
        "n_jobs":             -1,
        "seed":               42,
    }
    if params_override:
        params.update(params_override)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )

    return model


def train_xgb(X_train, y_train, X_val, y_val, colsample_bytree=0.8, sample_weight=None,
              params_override=None):
    print("\nTraining XGBoost...")

    X_tr = to_xgb(X_train)
    X_vl = to_xgb(X_val)

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())

    xgb_params = {
        "n_estimators":         2000,
        "learning_rate":        0.05,
        "max_depth":            6,
        "subsample":            0.8,
        "colsample_bytree":     colsample_bytree,
        "scale_pos_weight":     neg / pos,
        "eval_metric":          ["auc", "aucpr"],
        "early_stopping_rounds": 50,
        "tree_method":          "hist",
        "n_jobs":               -1,
        "random_state":         42,
        "verbosity":            1,
    }
    if params_override:
        xgb_params.update(params_override)

    model = xgb.XGBClassifier(**xgb_params)

    model.fit(
        X_tr, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_tr, y_train), (X_vl, y_val)],
        verbose=100,
    )

    return model, X_tr, X_vl


# ---------------------------------------------------------------------------
# Evaluation and plots
# ---------------------------------------------------------------------------

def evaluate(y_true, y_prob, label):
    roc   = roc_auc_score(y_true, y_prob)
    prauc = average_precision_score(y_true, y_prob)
    print(f"  {label:<26}  ROC-AUC: {roc:.4f}   PR-AUC: {prauc:.4f}")
    return {"roc_auc": roc, "pr_auc": prauc}


def plot_feature_importance(model, model_name, top_n=30):
    if isinstance(model, lgb.Booster):
        imp   = model.feature_importance(importance_type="gain")
        names = model.feature_name()
    else:
        imp   = model.feature_importances_
        names = model.get_booster().feature_names

    imp_df = (
        pd.DataFrame({"feature": names, "importance": imp})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    fig = go.Figure(go.Bar(
        x=imp_df["importance"],
        y=imp_df["feature"],
        orientation="h",
    ))
    fig.update_layout(
        title=f"{model_name} -- Top {top_n} Features by Gain",
        xaxis_title="Gain",
        yaxis=dict(autorange="reversed"),
        height=max(400, top_n * 20),
        margin=dict(l=220),
    )

    slug = model_name.lower().replace(" ", "_")
    out_path = os.path.join(OUT_DIR, f"feature_importance_{slug}.html")
    fig.write_html(out_path)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Phase 1 Baseline Pipeline ===\n")

    # Load and split
    df = load_data()
    train_raw, val_raw, test_raw = make_split(df)

    # Preprocess -- fit on train only
    print("\nFitting preprocessor on training fold...")
    prep    = Preprocessor()
    X_train = prep.fit_transform(train_raw)
    y_train = train_raw["isFraud"].values

    X_val = prep.transform(val_raw)
    y_val = val_raw["isFraud"].values

    X_test = prep.transform(test_raw)
    y_test = test_raw["isFraud"].values

    print(f"\n  X_train shape : {X_train.shape}")
    n_str_cat = len(X_train.select_dtypes("category").columns)
    print(f"  String cat cols (LightGBM native)    : {n_str_cat}")
    print(f"  Numeric cat cols (categorical_feature): {len(prep.numeric_cat_present)}")
    print(f"  OHE dummy cols                        : {len(prep.ohe_train_dummies)}")
    print(f"  D was_missing cols                    : {len(prep.d_cols)}")

    prep.save(os.path.join(MODELS_DIR, "preprocessor.pkl"))
    print("  Preprocessor saved to models/preprocessor.pkl")

    # LightGBM
    lgb_model = train_lgb(X_train, y_train, X_val, y_val, prep.numeric_cat_present)

    print("\nEvaluating LightGBM...")
    lgb_val_prob   = lgb_model.predict(X_val)
    lgb_train_prob = lgb_model.predict(X_train)
    lgb_metrics = {
        "val":   evaluate(y_val,   lgb_val_prob,   "LightGBM val"),
        "train": evaluate(y_train, lgb_train_prob,  "LightGBM train"),
    }

    lgb_model.save_model(os.path.join(MODELS_DIR, "lgb_baseline.txt"))
    with open(os.path.join(MODELS_DIR, "lgb_metrics.json"), "w") as f:
        json.dump(lgb_metrics, f, indent=2)

    plot_feature_importance(lgb_model, "LightGBM")

    # XGBoost
    xgb_model, X_tr_xgb, X_vl_xgb = train_xgb(X_train, y_train, X_val, y_val)

    print("\nEvaluating XGBoost...")
    xgb_val_prob   = xgb_model.predict_proba(X_vl_xgb)[:, 1]
    xgb_train_prob = xgb_model.predict_proba(X_tr_xgb)[:, 1]
    xgb_metrics = {
        "val":   evaluate(y_val,   xgb_val_prob,   "XGBoost val"),
        "train": evaluate(y_train, xgb_train_prob,  "XGBoost train"),
    }

    with open(os.path.join(MODELS_DIR, "xgb_baseline.pkl"), "wb") as f:
        pickle.dump(xgb_model, f)
    with open(os.path.join(MODELS_DIR, "xgb_metrics.json"), "w") as f:
        json.dump(xgb_metrics, f, indent=2)

    plot_feature_importance(xgb_model, "XGBoost")

    # Summary
    print("\n=== Validation Results ===")
    print(f"{'Model':<15}  {'ROC-AUC':>8}  {'PR-AUC':>8}")
    print("-" * 38)
    for name, m in [("LightGBM", lgb_metrics), ("XGBoost", xgb_metrics)]:
        print(f"{name:<15}  {m['val']['roc_auc']:>8.4f}  {m['val']['pr_auc']:>8.4f}")
    print("\nTrain-set scores saved to models/*_metrics.json for overfitting check.")
    print("Test set NOT evaluated here -- touch it exactly once at final reporting.")


if __name__ == "__main__":
    main()
