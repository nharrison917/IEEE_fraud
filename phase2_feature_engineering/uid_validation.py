# -*- coding: utf-8 -*-
"""
Validates a candidate improvement to the Tier 2 entity key before building
anything on top of it: does card1+card2+card3+card5 (the current proxy)
merge multiple distinct real accounts together, and does adjusting for D1
split them apart more cleanly?

Concept under test (general community knowledge about this competition,
not derived from reading anyone's submitted code): D1 behaves like "days
since this account was first seen," so transaction_day - D1 should be
roughly constant for a single real account across all its transactions --
even when the masked card1/2/3/5 fragments collide with a different real
customer. If the current combo entity actually mixes distinct accounts,
splitting it further by this candidate "account start day" should produce
tighter, more internally-consistent sub-groups (e.g. less addr1 diversity
within a sub-group than within the raw combo group).

This script only measures group cohesion -- it does not build or promote
any new feature yet. If the hypothesis doesn't hold up here, there's no
reason to invest further in it.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import load_data
from phase2_feature_engineering.tier2_features import _build_entity_key

_SECONDS_IN_DAY = 86400


def main():
    print("=== Validating D1-adjusted account-start-day as an entity-key refinement ===\n")

    df = load_data()

    print("D1 missingness:", df["D1"].isnull().mean().round(4))

    df["_combo_key"] = _build_entity_key(df)
    df["_txn_day"] = df["TransactionDT"] // _SECONDS_IN_DAY
    df["_start_day"] = df["_txn_day"] - df["D1"]

    combo_sizes = df.groupby("_combo_key").size()
    multi_row_combos = combo_sizes[combo_sizes > 1].index
    print(f"\nCombo entities with >1 transaction: {len(multi_row_combos):,} "
          f"(of {combo_sizes.size:,} total combo entities)")

    multi_df = df[df["_combo_key"].isin(multi_row_combos) & df["_start_day"].notnull()]
    print(f"Rows in multi-transaction combos with D1 present: {len(multi_df):,}")

    distinct_start_days = multi_df.groupby("_combo_key")["_start_day"].nunique()
    print("\nDistinct candidate account-start-days per combo entity (should be 1 if the "
          "combo already uniquely identifies a single real account):")
    print(distinct_start_days.value_counts().sort_index().head(10))
    print(f"  Mean distinct start-days per combo: {distinct_start_days.mean():.2f}")
    print(f"  Combos with >1 distinct start-day (possible collision): "
          f"{(distinct_start_days > 1).sum():,} of {len(distinct_start_days):,} "
          f"({(distinct_start_days > 1).mean()*100:.1f}%)")

    # Cohesion check: for combos that split into >1 start-day group, does
    # addr1 diversity drop within each finer (combo, start_day) sub-group
    # compared to the raw combo group? If the split is meaningful, the
    # finer groups should be more internally consistent.
    split_combos = distinct_start_days[distinct_start_days > 1].index
    sample = multi_df[multi_df["_combo_key"].isin(split_combos)]

    raw_addr1_nunique = sample.groupby("_combo_key")["addr1"].nunique()
    fine_addr1_nunique = sample.groupby(["_combo_key", "_start_day"])["addr1"].nunique()
    fine_addr1_mean_per_combo = fine_addr1_nunique.groupby("_combo_key").mean()

    print(f"\nFor the {len(split_combos):,} combos that split into multiple start-days:")
    print(f"  Mean distinct addr1 per RAW combo group       : {raw_addr1_nunique.mean():.3f}")
    print(f"  Mean distinct addr1 per FINER (combo,start_day) sub-group : "
          f"{fine_addr1_mean_per_combo.mean():.3f}")
    print("  (lower is more internally consistent -- evidence the split separates real accounts)")


if __name__ == "__main__":
    main()
