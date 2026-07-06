# -*- coding: utf-8 -*-
"""
Diagnoses the fraud-rate divergence between the has_identity=1 and
has_identity=0 populations that motivated splitting into a segmented model
(see PLAN.md, session 5/6). Two checks:

1. Per-fold fraud rate by segment -- confirms has_identity=1 runs at a
   substantially higher, and less temporally stable, fraud rate than
   has_identity=0.
2. Within-train chronological quarters -- checks whether the train-to-val
   jump in the has_identity=1 rate is a genuine out-of-sample surprise, or
   whether it's a continuation of a regime shift that already happened
   partway through the training window itself.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import load_data, make_split


def print_segment_rates(df, label):
    print(f"--- {label} ---")
    print(f"  Overall fraud rate: {df['isFraud'].mean()*100:.2f}%  (n={len(df):,})")
    for has_id, seg_label in [(1, "has_identity=1"), (0, "has_identity=0")]:
        sub = df[df["has_identity"] == has_id]
        print(
            f"  {seg_label}: n={len(sub):,} ({len(sub)/len(df)*100:.1f}% of fold)  "
            f"fraud_rate={sub['isFraud'].mean()*100:.2f}%  fraud_count={sub['isFraud'].sum():,}"
        )
    print()


def main():
    print("=== Segment fraud-rate diagnostic: has_identity=1 vs. has_identity=0 ===\n")

    df = load_data()
    df["has_identity"] = df["id_01"].notnull().astype(int)
    train_raw, val_raw, test_raw = make_split(df)

    print("\n--- Per-fold segment fraud rates ---")
    for name, fold in [("Train", train_raw), ("Val", val_raw), ("Test", test_raw)]:
        print_segment_rates(fold, name)

    print("--- Within-train chronological quarters (is the train->val jump new, or already underway?) ---\n")
    train_sorted = train_raw.sort_values("TransactionDT").reset_index(drop=True)
    n = len(train_sorted)
    quarter = n // 4

    for i in range(4):
        start = i * quarter
        end = (i + 1) * quarter if i < 3 else n
        chunk = train_sorted.iloc[start:end]
        id_chunk = chunk[chunk["has_identity"] == 1]
        noid_chunk = chunk[chunk["has_identity"] == 0]
        dt_min, dt_max = chunk["TransactionDT"].min(), chunk["TransactionDT"].max()
        print(f"  Q{i+1} (DT {dt_min:,}-{dt_max:,}): n={len(chunk):,}")
        print(f"    has_identity share: {len(id_chunk)/len(chunk)*100:.1f}%")
        print(
            f"    has_identity=1 fraud rate: {id_chunk['isFraud'].mean()*100:.2f}%  "
            f"(n_fraud={id_chunk['isFraud'].sum()})"
        )
        print(
            f"    has_identity=0 fraud rate: {noid_chunk['isFraud'].mean()*100:.2f}%  "
            f"(n_fraud={noid_chunk['isFraud'].sum()})"
        )

    print("\n  Val fold (for reference):")
    print_segment_rates(val_raw, "Val")


if __name__ == "__main__":
    main()
