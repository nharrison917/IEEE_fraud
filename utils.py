# -*- coding: utf-8 -*-
"""
Shared data loading and splitting utilities.

The split is chronological (sorted by TransactionDT), not random.
This simulates real deployment: train on past, evaluate on future.

Split proportions: 60% train / 20% val / 20% test.
All preprocessing must be fitted on train only and applied to val/test.
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_data(verbose=True):
    """
    Load and join train_transaction.csv and train_identity.csv.
    Returns the combined DataFrame. No preprocessing applied.
    """
    if verbose:
        print("Loading transaction data...")
    txn = pd.read_csv(os.path.join(DATA_DIR, "train_transaction.csv"))

    if verbose:
        print("Loading identity data...")
    idf = pd.read_csv(os.path.join(DATA_DIR, "train_identity.csv"))

    if verbose:
        print("Joining on TransactionID (left join)...")
    df = txn.merge(idf, on="TransactionID", how="left")

    if verbose:
        print(f"  Combined: {df.shape[0]:,} rows x {df.shape[1]} columns")

    return df


def make_split(df, train_pct=0.60, val_pct=0.20, verbose=True):
    """
    Chronological train/val/test split sorted by TransactionDT.

    Returns (train, val, test) DataFrames. Split indices are based on
    row position after sorting — not on time values directly — so each
    fold gets a proportional number of rows regardless of transaction
    density variation across the timeline.

    Parameters
    ----------
    df         : combined DataFrame from load_data()
    train_pct  : fraction of rows for training (default 0.60)
    val_pct    : fraction of rows for validation (default 0.20)
    verbose    : print split statistics
    """
    df_sorted = df.sort_values("TransactionDT").reset_index(drop=True)

    n = len(df_sorted)
    train_end = int(n * train_pct)
    val_end   = int(n * (train_pct + val_pct))

    train = df_sorted.iloc[:train_end].copy()
    val   = df_sorted.iloc[train_end:val_end].copy()
    test  = df_sorted.iloc[val_end:].copy()

    if verbose:
        _print_split_report(train, val, test)

    return train, val, test


def _print_split_report(train, val, test):
    print("\n--- Split Report ---")
    print(f"{'Fold':<8} {'Rows':>8}  {'Fraud':>6}  {'Fraud%':>7}  {'DT min':>12}  {'DT max':>12}")
    print("-" * 62)
    for name, fold in [("Train", train), ("Val", val), ("Test", test)]:
        n       = len(fold)
        n_fraud = fold["isFraud"].sum()
        rate    = n_fraud / n * 100
        dt_min  = fold["TransactionDT"].min()
        dt_max  = fold["TransactionDT"].max()
        print(f"{name:<8} {n:>8,}  {n_fraud:>6,}  {rate:>6.2f}%  {dt_min:>12,}  {dt_max:>12,}")
    print()
    total_fraud = train["isFraud"].sum() + val["isFraud"].sum() + test["isFraud"].sum()
    print(f"Total fraud cases accounted for: {total_fraud:,}")
    print(
        "NOTE: any preprocessing (imputation, scaling, encoding, SMOTE) "
        "must be fitted on train only."
    )
