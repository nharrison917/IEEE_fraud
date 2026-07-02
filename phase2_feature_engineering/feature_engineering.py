# -*- coding: utf-8 -*-
"""
Tier 1 feature engineering: within-row derived features only.

Every feature here is a deterministic function of a single row -- no
cross-row aggregation, no fit/transform split, no leakage risk (see
PLAN.md "Tier 1 -- within-row features (safe, no leakage risk)").

Must run on raw (post-join, pre-preprocessing) dataframes, since it needs
TransactionDT, TransactionAmt, id_14, and the raw email/browser/OS columns
before phase1_baseline.pipeline.Preprocessor drops or encodes them.
"""

import numpy as np
import pandas as pd

# Known free/webmail providers, matched by domain prefix so regional
# variants (hotmail.co.uk, hotmail.de, live.com.mx, ...) fall into the
# same bucket as their .com parent. Built from the actual 59/60 unique
# domain values observed in train_transaction.csv.
_FREE_EMAIL_PREFIXES = (
    "gmail", "yahoo", "hotmail", "live", "outlook", "msn", "aol", "icloud",
    "mac.com", "me.com", "mail.com", "protonmail", "ymail", "rocketmail",
    "web.de", "gmx", "netzero", "juno",
)

# id_31 values are messy: version numbers, "for android"/"for ios" suffixes,
# and a handful of device-model strings (e.g. "Samsung/SCH", "LG/K-200")
# that leaked into this column. Bucket by substring match on known browser
# families; anything unmatched (rare desktop browsers, stray device names)
# falls into "other". "safari" is checked before "mobile" so "mobile safari
# 10.0" resolves to "safari", not "mobile".
_BROWSER_FAMILIES = (
    "chrome", "firefox", "safari", "edge", "opera", "samsung",
    "android", "google", "facebook", "aol", "line", "ie", "mobile",
)

_SECONDS_IN_DAY = 86400


def _map_unique(series, fn):
    """Apply fn to each unique value once, then map back -- avoids running
    fn per-row on columns with hundreds of thousands of rows but only
    tens of unique values."""
    uniques = series.dropna().unique()
    lookup = {v: fn(v) for v in uniques}
    return series.map(lookup)


def _free_email_flag(series):
    def is_free(domain):
        d = str(domain)
        return any(d.startswith(p) for p in _FREE_EMAIL_PREFIXES)
    return _map_unique(series, is_free).fillna(False).astype(int)


def _parse_browser_name(series):
    def parse(value):
        v = str(value).lower()
        for family in _BROWSER_FAMILIES:
            if family in v:
                return family
        return "other"
    return _map_unique(series, parse)


def _parse_os_name(series):
    def parse(value):
        v = str(value)
        if v.startswith("Mac OS X") or v == "Mac":
            return "Mac OS X"
        if v.startswith("Android"):
            return "Android"
        if v.startswith("Windows"):
            return "Windows"
        if v.startswith("iOS"):
            return "iOS"
        if v == "Linux":
            return "Linux"
        return "other"  # covers the "other" and stray "func" values
    return _map_unique(series, parse)


def add_tier1_features(df):
    """
    Add within-row engineered features. Safe to call independently on
    train/val/test -- no fitting involved.

    Raw id_30/id_31/P_emaildomain/R_emaildomain are left in place; the
    Phase 1 Preprocessor still encodes them as before. Parsed columns
    supplement rather than replace them (assumption confirmed with user:
    version-specific signal, e.g. an outdated browser correlating with
    fraud, may be lost if the raw column were dropped).
    """
    df = df.copy()

    # --- Time features -----------------------------------------------
    # TransactionDT is relative seconds from an undisclosed reference point.
    # Only relative periodicity (hour-of-day, day-of-week cycles) is valid --
    # absolute calendar dates cannot be recovered (see PLAN.md EDA notes).
    df["hour_of_day"] = (df["TransactionDT"] % _SECONDS_IN_DAY) // 3600
    df["day_of_week"] = (df["TransactionDT"] // _SECONDS_IN_DAY) % 7

    # local_hour: id_14 is the identity session's timezone offset in minutes.
    # TransactionDT + offset, wrapped to a 24h clock, approximates true local
    # wall-clock hour for the ~24% of rows with an identity record -- this
    # assumption (TransactionDT is UTC) was confirmed valid in a prior
    # session (see PLAN.md Session Resumption Checklist). Falls back to the
    # UTC-relative hour_of_day where id_14 is absent.
    offset_seconds = df["id_14"].fillna(0) * 60
    raw_local_hour = ((df["TransactionDT"] + offset_seconds) % _SECONDS_IN_DAY) // 3600
    df["local_hour"] = np.where(
        df["id_14"].notnull(), raw_local_hour, df["hour_of_day"]
    ).astype(int)

    # --- Amount features -----------------------------------------------
    df["log1p_TransactionAmt"] = np.log1p(df["TransactionAmt"])
    df["is_round_amount"] = (df["TransactionAmt"].round(2) % 1 == 0).astype(int)

    # --- Email features -----------------------------------------------
    both_present = df["P_emaildomain"].notnull() & df["R_emaildomain"].notnull()
    df["P_email_matches_R_email"] = (
        both_present & (df["P_emaildomain"] == df["R_emaildomain"])
    ).astype(int)
    df["P_email_is_free"] = _free_email_flag(df["P_emaildomain"])

    # --- Device features -----------------------------------------------
    df["browser_name"] = _parse_browser_name(df["id_31"])
    df["os_name"] = _parse_os_name(df["id_30"])

    return df
