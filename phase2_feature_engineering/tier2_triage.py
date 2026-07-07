# -*- coding: utf-8 -*-
"""
Tier 2 triage: before building card-level velocity features, check whether
Vesta's existing C/D columns already cover the same signal.

Builds a small set of candidate features -- transaction count, time since
last transaction, and an amount z-score, all grouped by a card1+card2+
card3+card5 combo entity -- plus an addr1-change flag, and correlates them
against the 14 C columns and 15 D columns. High correlation with an
existing column is evidence the candidate is redundant, not novel signal.
Low correlation means it may be worth adding for real.

Entity key deliberately excludes addr1: folding addr1 into the grouping
key would make an address change look like a brand-new entity (history
resets to zero) rather than a flagged event on a continuous account.
Keeping addr1 out of the key and instead tracking whether it changed from
the entity's immediately preceding transaction preserves account
continuity while still surfacing the address change itself as a signal.

Leakage discipline: every candidate is computed causally -- each row only
uses transactions strictly earlier in time for the same entity, via an
expanding window computed on the full timeline (sorted by entity then
TransactionDT) before any train/val/test split. This is intentionally
different from the Preprocessor's "fit on train only" rule: a deployed
model has access to a card's full prior history, including whatever here
falls inside the "train" window, so computing this before the split is
correct, not a leak. The correlation check itself is restricted to the
train fold only, to avoid using val/test data to make a feature-selection
decision.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import load_data, make_split

_C_COLS = [f"C{i}" for i in range(1, 15)]
_D_COLS = [f"D{i}" for i in range(1, 16)]
_ENTITY_COLS = ["card1", "card2", "card3", "card5"]


def _build_entity_key(df):
    """Vectorized string concat of card1/card2/card3/card5 (NaN filled with
    a sentinel for grouping purposes only -- doesn't touch the real columns
    used elsewhere in the pipeline)."""
    parts = [df[c].fillna(-999).astype(int).astype(str) for c in _ENTITY_COLS]
    key = parts[0]
    for p in parts[1:]:
        key = key + "_" + p
    return key


def add_entity_velocity_candidates(df):
    """Causal (expanding-window) entity-level candidates, grouped by the
    card1+card2+card3+card5 combo. Computed on the full dataset sorted by
    entity then TransactionDT, then restored to the caller's original row
    order via sort_index()."""
    working = df.copy()
    working["_entity_key"] = _build_entity_key(df)

    sorted_df = working.sort_values(["_entity_key", "TransactionDT"])
    grp = sorted_df.groupby("_entity_key")

    # Prior transaction count for this entity (0 for its first transaction).
    prior_count = grp.cumcount()

    # Time since this entity's previous transaction (NaN for the first).
    time_since_last = sorted_df["TransactionDT"] - grp["TransactionDT"].shift(1)

    # Expanding mean/std of TransactionAmt *excluding the current row*, via
    # cumsum/cumcount rather than a per-group .expanding().apply(), which is
    # far too slow at ~590k rows / tens of thousands of entity groups.
    amt = sorted_df["TransactionAmt"]
    cumsum_incl = grp["TransactionAmt"].cumsum()
    cumsum_prior = cumsum_incl - amt
    count = prior_count.replace(0, np.nan)  # first txn has no prior mean
    prior_mean = cumsum_prior / count

    sq = amt ** 2
    sorted_df["_sq"] = sq
    cumsum_sq_incl = sorted_df.groupby("_entity_key")["_sq"].cumsum()
    cumsum_sq_prior = cumsum_sq_incl - sq
    prior_var = (cumsum_sq_prior / count) - prior_mean ** 2
    prior_std = np.sqrt(prior_var.clip(lower=0))  # guard tiny float negatives
    amt_zscore = (amt - prior_mean) / prior_std.replace(0, np.nan)

    # addr1 change vs. this entity's immediately preceding transaction.
    # 0 (not 1) whenever there's nothing to compare against -- the entity's
    # first transaction, or either addr1 value missing -- since prior_count
    # == 0 already flags "no history" elsewhere; this stays a clean binary
    # signal rather than a three-state one.
    prev_addr1 = grp["addr1"].shift(1)
    addr1_changed = (
        (sorted_df["addr1"] != prev_addr1)
        & prev_addr1.notnull()
        & sorted_df["addr1"].notnull()
    ).astype(int)

    new_cols = pd.DataFrame({
        "combo_prior_txn_count": prior_count,
        "combo_time_since_last": time_since_last,
        "combo_amt_zscore": amt_zscore,
        "addr1_changed_from_prev": addr1_changed,
    })
    sorted_df = pd.concat([sorted_df.drop(columns=["_sq", "_entity_key"]), new_cols], axis=1)

    return sorted_df.sort_index()


def main():
    print("=== Tier 2 Triage: card1+card2+card3+card5 combo entity vs. existing C/D columns ===\n")

    df = load_data()
    df = add_entity_velocity_candidates(df)

    train_raw, val_raw, test_raw = make_split(df)

    candidates = [
        "combo_prior_txn_count", "combo_time_since_last",
        "combo_amt_zscore", "addr1_changed_from_prev",
    ]
    print("\nCandidate feature summary (train fold):")
    print(train_raw[candidates].describe())

    print("\nCorrelation of candidates vs. C columns (train fold, Pearson):")
    corr_c = train_raw[candidates + _C_COLS].corr().loc[candidates, _C_COLS]
    print(corr_c.round(3).to_string())

    print("\nCorrelation of candidates vs. D columns (train fold, Pearson):")
    corr_d = train_raw[candidates + _D_COLS].corr().loc[candidates, _D_COLS]
    print(corr_d.round(3).to_string())

    print("\nMax abs correlation per candidate:")
    for c in candidates:
        max_c = corr_c.loc[c].abs().idxmax()
        max_c_val = corr_c.loc[c].abs().max()
        max_d = corr_d.loc[c].abs().idxmax()
        max_d_val = corr_d.loc[c].abs().max()
        print(f"  {c:<26} vs C: {max_c} ({max_c_val:.3f})   vs D: {max_d} ({max_d_val:.3f})")

    print("\nCorrelation with isFraud (train fold) -- for context, not decision-making:")
    # Series.corr() crashes the interpreter outright in this environment
    # (pandas 3.0.3 / numpy 2.4.6 -- reproduced with a bare 5-element
    # Series, unrelated to this script's logic). DataFrame.corr() uses a
    # different internal path and is fine, so build a small frame instead.
    fraud_corr = train_raw[candidates + ["isFraud"]].corr()["isFraud"]
    for c in candidates:
        print(f"  {c:<26} {fraud_corr[c]:.4f}")

    print("\naddr1_changed_from_prev rate (train fold):", train_raw["addr1_changed_from_prev"].mean())
    print("Fraud rate when addr1 changed:", train_raw.loc[train_raw["addr1_changed_from_prev"] == 1, "isFraud"].mean())
    print("Fraud rate when addr1 did not change:", train_raw.loc[train_raw["addr1_changed_from_prev"] == 0, "isFraud"].mean())


if __name__ == "__main__":
    main()
