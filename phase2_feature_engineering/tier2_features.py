# -*- coding: utf-8 -*-
"""
Tier 2: card-level velocity/behavioral features, grouped by a refined
entity key -- card1+card2+card3+card5 plus a D1-adjusted account-start-day
(addr1 deliberately excluded from the key itself -- see
add_entity_velocity_features docstring).

The raw card1+card2+card3+card5 combo alone was validated (uid_validation.py)
to badly conflate distinct real accounts: 84.1% of multi-transaction combos
split into more than one distinct D1-adjusted start-day, and the average
raw combo group spans 3.9 distinct addr1 values (i.e. multiple different
people) vs. 0.94 after adding the start-day split. D1 behaving like "days
since this account was first seen" -- so transaction_day - D1 is roughly
constant per real account even when masked card fragments collide -- is a
general idea from this competition's community, not derived from reading
anyone's submitted code; it was validated independently against this
project's own data (see uid_validation.py) before being used here. D1 is
~99.8% populated, so coverage is not a concern the way it was for id_14.

Unlike Tier 1, these are NOT independent per-row functions: each row's
feature depends on that entity's transaction history, so they must be
computed on the full dataset (sorted by entity, then time) before the
train/val/test split, not per-fold like the Preprocessor. Every feature is
causal (expanding-window, strictly-prior-transactions-only) by construction
-- see tier2_triage.py's docstring for why this is legitimate and where the
leakage risk would come from if done carelessly.

v1 (this refined key) supersedes the card1+card2+card3+card5-only entity
key tested first, which showed no reliable signal (see PLAN.md) -- a result
now understood to be at least partly a consequence of that key's collision
rate rather than these feature concepts being uninformative.
"""

import numpy as np
import pandas as pd

_ENTITY_COLS = ["card1", "card2", "card3", "card5"]
_SECONDS_IN_DAY = 86400


def _build_entity_key(df):
    """Vectorized string concat of card1/card2/card3/card5 plus a D1-
    adjusted account-start-day (NaN filled with a sentinel for grouping
    purposes only -- doesn't touch the real columns used elsewhere in the
    pipeline)."""
    parts = [df[c].fillna(-999).astype(int).astype(str) for c in _ENTITY_COLS]
    key = parts[0]
    for p in parts[1:]:
        key = key + "_" + p

    txn_day = df["TransactionDT"] // _SECONDS_IN_DAY
    start_day = (txn_day - df["D1"]).fillna(-999).astype(int).astype(str)
    return key + "_" + start_day


def add_entity_velocity_features(df):
    """
    Adds 6 causal, entity-level features:
      - combo_prior_txn_count: prior transaction count for this entity
      - combo_time_since_last: seconds since this entity's previous transaction
      - combo_amt_zscore: current TransactionAmt vs. this entity's prior mean/std
      - addr1_changed_from_prev: 1 if addr1 differs from this entity's
        immediately preceding transaction, else 0 (including when there's no
        prior transaction to compare against, or either value is missing)
      - addr1_change_x_inverse_time: addr1_changed_from_prev / (time_since_last
        + 1) -- a continuous "changed, and how fast" score. 0 when addr1
        didn't change; larger when it did change and little time has passed.
        Proxy for "implausible geographic range" -- we don't have real
        coordinates or a travel-speed calculation, but a fast address change
        is the closest available signal to the same idea. Built as an
        explicit interaction rather than left for the trees to discover from
        the two raw features, since a multiplicative relationship like this
        generally needs several axis-aligned splits to approximate.
      - amt_ratio_vs_prev: current TransactionAmt / this entity's immediately
        preceding TransactionAmt. Targets escalating-amount ("card testing")
        patterns specifically -- combo_amt_zscore compares against the
        entity's entire history, which would dilute a sudden jump from the
        single most recent transaction.

    Entity key excludes addr1 by design: folding addr1 into the grouping
    key would make an address change look like a brand-new entity (history
    resets to zero) instead of a flagged event on a continuous account.
    """
    working = df.copy()
    working["_entity_key"] = _build_entity_key(df)

    sorted_df = working.sort_values(["_entity_key", "TransactionDT"])
    grp = sorted_df.groupby("_entity_key")

    prior_count = grp.cumcount()
    time_since_last = sorted_df["TransactionDT"] - grp["TransactionDT"].shift(1)

    amt = sorted_df["TransactionAmt"]
    cumsum_incl = grp["TransactionAmt"].cumsum()
    cumsum_prior = cumsum_incl - amt
    count = prior_count.replace(0, np.nan)
    prior_mean = cumsum_prior / count

    sq = amt ** 2
    sorted_df["_sq"] = sq
    cumsum_sq_incl = sorted_df.groupby("_entity_key")["_sq"].cumsum()
    cumsum_sq_prior = cumsum_sq_incl - sq
    prior_var = (cumsum_sq_prior / count) - prior_mean ** 2
    prior_std = np.sqrt(prior_var.clip(lower=0))
    amt_zscore = (amt - prior_mean) / prior_std.replace(0, np.nan)

    prev_addr1 = grp["addr1"].shift(1)
    addr1_changed = (
        (sorted_df["addr1"] != prev_addr1)
        & prev_addr1.notnull()
        & sorted_df["addr1"].notnull()
    ).astype(int)

    # time_since_last is NaN for an entity's first transaction (no addr1
    # change is possible there either, so addr1_changed is already 0 --
    # fillna(0) here just avoids propagating NaN into the product).
    addr1_change_x_inverse_time = addr1_changed / (time_since_last.fillna(0) + 1)

    prev_amt = grp["TransactionAmt"].shift(1)
    amt_ratio_vs_prev = amt / prev_amt.replace(0, np.nan)

    new_cols = pd.DataFrame({
        "combo_prior_txn_count": prior_count,
        "combo_time_since_last": time_since_last,
        "combo_amt_zscore": amt_zscore,
        "addr1_changed_from_prev": addr1_changed,
        "addr1_change_x_inverse_time": addr1_change_x_inverse_time,
        "amt_ratio_vs_prev": amt_ratio_vs_prev,
    })
    sorted_df = pd.concat([sorted_df.drop(columns=["_sq", "_entity_key"]), new_cols], axis=1)

    return sorted_df.sort_index()
