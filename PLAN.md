# IEEE-CIS Fraud Detection — Project Plan

Sequel to `C:\Projects\credit_card_fraud`. That project was constrained by PCA
anonymization (V1-V28) which eliminated card identifiers and prevented any
relational or sequential feature engineering. This dataset preserves enough
structure to do it properly.

**Portfolio context:** Job application target values fraud analysis experience.
Methodology must be bulletproof — every decision documented and justified.

---

## Environment

- Conda environment: `ieee-fraud`
- Activate: open **Anaconda Prompt** (not PowerShell — conda's PS integration
  is broken on this machine), then `conda activate ieee-fraud`
- VS Code interpreter: select `Python (ieee-fraud)` via Ctrl+Shift+P
- Git remote: https://github.com/nharrison917/IEEE_fraud.git
- **Known bug (session 5):** `pandas.Series.corr()` crashes the Python
  process outright in this env (pandas 3.0.3 / numpy 2.4.6) — no exception,
  no traceback, reproduced on a bare 5-element Series. `DataFrame.corr()`
  is unaffected (different internal code path). Workaround: build a small
  DataFrame and call `.corr()` on that instead of two Series directly.

---

## Data

**Source:** Kaggle IEEE-CIS Fraud Detection competition  
**Files (in `data/`, not tracked in git):**

| File | Size | Use |
|---|---|---|
| `train_transaction.csv` | 652MB | All labeled training data |
| `train_identity.csv` | 26MB | Identity features for training set |
| `test_transaction.csv` | 585MB | Competition test set — no labels, not used |
| `test_identity.csv` | 25MB | Competition test set — not used |

We use only the `train_*` files. The competition test files have no `isFraud`
labels and are useless for supervised learning.

**Key dataset facts (from EDA):**
- 590,540 total transactions after join
- 434 columns (transaction + identity features combined)
- 3.50% fraud rate (20,663 fraud cases)
- 182 days of data (~6 months)
- 24.4% of transactions have identity records
- TransactionDT is seconds from an undisclosed reference point — absolute
  dates and day names cannot be confirmed; only relative periodicity is valid

---

## Split Strategy

**Chronological 60/20/20 — implemented in `utils.py`**

```
load_data()       -> loads and joins train_transaction + train_identity
make_split(df)    -> sorts by TransactionDT, splits by row position
```

| Fold  | Rows    | Fraud  | Fraud% | DT range |
|---|---|---|---|---|
| Train | 354,324 | 11,988 | 3.38%  | 86,400 – 8,745,772 |
| Val   | 118,108 |  4,611 | 3.90%  | 8,745,798 – 12,192,842 |
| Test  | 118,108 |  4,064 | 3.44%  | 12,192,900 – 15,811,131 |

Val fraud rate is slightly elevated (3.90% vs 3.38% train). Minor limitation —
thresholds optimized on val are tuned to a window with marginally more fraud
than average. Noted for write-up.

**Rule:** Everything downstream of the split is fitted on train only and
applied to val/test. The test set is touched exactly once at the end.

---

## EDA Findings (train fold only, `phase1_baseline/`)

### Missingness structure

414 of 436 columns have some missing values. Missingness is **structural,
not random** — it encodes transaction sub-type.

| Group | Cols | Mean missing | Notes |
|---|---|---|---|
| V (Vesta engineered) | 339 | 42.7% | 10 distinct rates — transaction sub-types |
| C (counting) | 14 | 0.0% | Fully present — Vesta velocity features |
| D (timedelta) | 15 | 60.0% | High missing = "no prior event" |
| M (match flags) | 9 | 58.8% | T/F/missing are three distinct states |
| id_ (identity) | 38 | 82.8% | Arrives as a block; 72.5% fully absent |
| card | 6 | 0.5% | Minimal missing |
| addr | 2 | 11.0% | Moderate missing |
| dist | 2 | 77.3% | Splits by sub-type (see below) |

### V column sub-type structure

339 V columns collapse to **10 distinct missingness rates** — transaction
sub-types in Vesta's internal processing systems. Key finding from cross-
correlation analysis:

- **V groups 73.0%, 73.4%, 74.7% missing**: correlation 0.94–0.99 with
  identity presence. When present: 99–100% of rows have identity records.
  These V features are Vesta's identity verification pipeline outputs.

- **V groups 16.2%, 16.5%, 18.0% missing**: correlation -0.64 to -0.72 with
  identity. When present: only 13–14% of rows have identity. These are the
  non-identity transaction path.

- **V group 58.5% missing**: 0% identity overlap when present. 100% addr1/addr2
  presence. 0.98 correlation with M1–M3 match flags. Distinct transaction type,
  likely card-present / domestic.

- **dist1 vs dist2 are on opposite sides**: dist1 present in non-identity
  transactions only; dist2 present in identity transactions only.

`has_identity` flag is nearly interchangeable with V group 73%/73.4% membership.
Adding it as an explicit feature is cleaner than relying on the sentinel pattern.

### What we do NOT know

- V column content: described only as "Vesta engineered rich features."
  Content is proprietary and undisclosed. Do not claim they are velocity
  features in the write-up. They contain strong signal but we cannot
  characterize them further.
- C columns are explicitly described as counting/velocity features.
- D columns are explicitly described as timedelta features.

---

## Preprocessing Pipeline

**Order of operations (all fitted on train, applied to all folds):**

### 1. Add `has_identity` flag
```python
df["has_identity"] = df["id_01"].notnull().astype(int)
```
Add before any imputation so the flag reflects true absence, not imputed values.

### 2. Missing value handling

| Feature group | Strategy | Rationale |
|---|---|---|
| V columns | Fill with -999 | Structural missingness; trees route on sentinel |
| id_ numeric | Fill with -999 | Same |
| id_ categorical | Fill with "missing" | Third distinct state |
| D columns | Add `D{n}_was_missing` binary, then fill -999 | Absence = "no prior event" signal |
| M columns | Fill with "missing" (add as category) | T/F/missing are three states |
| card1 | Median fill for missing; treat as numeric | 11,735 unique values — masked identifier; tree threshold splits are appropriate |
| card2, card3, card5 | Median fill for missing; mark as categorical in LightGBM | 100–499 unique values — numeric codes, not ordinal; LightGBM can group non-contiguous codes |
| card4, card6 | Mode fill for missing; already string categorical | Confirmed categorical (visa/debit etc.) |
| addr1 | -999 fill for missing; mark as categorical in LightGBM | 308 unique integer values — billing region/ZIP code, not ordinal |
| addr2 | Convert to binary `is_us` (value == 87 → 1) | 99% of rows = 87 (US country code); binary preserves the 1% non-US signal |
| dist1, dist2 | -999 fill for missing; keep numeric | Genuinely continuous distance values (0–10,000+) |

### 3. Categorical encoding

| Column(s) | Cardinality | Strategy |
|---|---|---|
| ProductCD | 5 | One-hot |
| card4, card6 | 4 | One-hot |
| DeviceType | 2 | One-hot |
| M1–M9 (after filling) | 3 | One-hot |
| id_12–id_38 categorical | 2–4 | One-hot |
| P_emaildomain, R_emaildomain | 59–60 | Target encoding (fit on train only) |
| id_30 (OS), id_31 (browser) | 75, 130 | Target encoding or group to OS family |
| DeviceInfo | 1,786 | Target encoding |
| id_14 | 25 | Categorical (timezone offsets in minutes: -660 to 720) |
| id_32 | 4 | Categorical (screen color depth: 24-bit, 32-bit etc.) |
| id_13, id_17, id_18, id_19, id_20, id_21, id_22, id_24, id_25, id_26 | 10–492 | Categorical codes — mark as categorical in LightGBM |
| id_01–id_11 (excl. id_02, id_14) | n/a | Numeric confirmed (risk scores / delta fields) |
| id_02 | 81,902 | Numeric (high-cardinality device/session identifier) |

**Target encoding leakage note:** Encoding each training row using the mean
of *all* training rows for that category is circular. Use leave-one-out or
k-fold target encoding on the training set. LightGBM's native categorical
handling avoids this entirely — preferred approach for high-cardinality columns.

### 4. No SMOTE initially — revisited, session 7: SMOTE 1:10 adopted
At 3.5% fraud rate, test `class_weight='balanced'` (LightGBM parameter) first.
Compare against modest SMOTE (1:10) on validation if needed. SMOTE applied to
training fold only, after all preprocessing, never before split.

**Resolved (session 7):** this comparison was run (see "SMOTE ablation and
production config revision" under the Segmented Model section) and SMOTE
1:10 won clearly for both production models. `class_weight`/`is_unbalance`
is no longer used in the production config.

---

## Feature Engineering

### Phase 1 baseline: no feature engineering
Train baseline models on preprocessed raw features only. Establishes a
clean before/after story for the write-up and portfolio.

### Phase 2 enhancement (after baseline results):

**Tier 1 — within-row features (safe, no leakage risk):**
- `hour_of_day`: `(TransactionDT % 86400) // 3600` — relative only (see timezone note below)
- `day_of_week`: `(TransactionDT // 86400) % 7` — relative
- `local_hour` (where id_14 is present): `((TransactionDT + id_14*60) % 86400) // 3600`
  — **if TransactionDT is UTC**, applying the id_14 offset gives true local hour,
  which is far more meaningful for fraud timing patterns (fraud follows local behavior,
  not UTC). Confirm assumption before using. Only available for ~24% of rows (identity
  transactions); fall back to relative hour elsewhere.
- `log1p_TransactionAmt`: right-skewed amounts
- `is_round_amount`: `TransactionAmt % 1 == 0`
- `P_email_matches_R_email`: purchaser and recipient on same domain
- `P_email_is_free`: gmail/yahoo/hotmail vs corporate domain
- Parsed `browser_name` from id_31 (drop version strings)
- Parsed `os_name` from id_30

**Tier 2 — card-level aggregates: see "Phase 2 Tier 2" section below for the
full results. Summary: the first entity key tried (`card1+card2+card3+card5`
alone) showed no reliable value; a refined key incorporating a D1-adjusted
account-start-day fixed a severe entity-collision problem and produced real,
evidenced value for LightGBM. Implemented in `phase2_feature_engineering/
tier2_features.py`.**

---

## Phase 1 — Baseline Detection

### Models
1. **LightGBM** (primary) — handles high-cardinality categoricals natively,
   fast on wide datasets, strong baseline for tabular fraud
2. **XGBoost** (comparison) — cross-check; may differ on encoded categoricals

Random Forest excluded: requires one-hot encoding of all categoricals, making
the comparison unfair against LightGBM's native categorical handling. Noted
in write-up.

Logistic Regression: excluded because structural missingness (-999 sentinels)
and high-cardinality categoricals actively work against it. Documented in
methodology rather than included with a separate preprocessing path.

### Evaluation metrics
- **PR-AUC** (primary): Precision-Recall AUC, more informative than ROC-AUC
  for imbalanced classes. A model can achieve high ROC-AUC while performing
  poorly at operationally relevant precision levels.
- **ROC-AUC** (secondary): Standard for this competition; enables comparison
  to published results.
- **F1 at optimal threshold**: For operational interpretation in Phase 2.

### Pipeline file
`phase1_baseline/pipeline.py` — complete.

Steps:
1. `load_data()` + `make_split()` from utils.py
2. Preprocessing (fitted on train, applied to all folds)
3. LightGBM train + val evaluation
4. XGBoost train + val evaluation
5. Feature importance plots
6. Save best model to `models/`

### Phase 1 results (validation set, no feature engineering)

| Model | Val ROC-AUC | Val PR-AUC | Train PR-AUC | Best round |
|---|---|---|---|---|
| LightGBM (127 leaves) | 0.9125 | 0.5464 | 0.864 | 89 |
| XGBoost | 0.9170 | 0.5714 | 0.827 | 648 |

**XGBoost wins Phase 1** on both metrics despite receiving ordinal integer codes for
categorical columns where LightGBM got native categorical handling. Likely explanation:
the V columns (continuous Vesta outputs) carry most Phase 1 signal and suit XGBoost's
numeric treatment; LightGBM's categorical advantage on card2/addr1 was real but not
enough to overcome the parameter configuration.

**LightGBM early stopping** at 89 rounds is a genuine ceiling, not a parameter
artifact: reducing `num_leaves` to 63 caused earlier stopping (76 rounds) and worse
val PR-AUC (0.531). 127 leaves are needed to partition card2's 400+ unique codes.

**Key feature importance findings:**
- LightGBM: card2 (#1 by large margin), addr1 (#3) — native categoricals dominating
- XGBoost: V258, V70, V294 dominate — continuous Vesta features
- M4__missing in both top-30 lists — validates 3-state M column treatment
- D2_was_missing in XGBoost top-30 — validates D missingness indicator strategy
- is_us in XGBoost top-12 — addr2 binary conversion carried signal
- id_01-id_11 Vesta risk scores absent from both top-30 — signal likely absorbed by V columns

**Train-val gap** is significant (especially PR-AUC). Expected for no-tuning baseline
with high-cardinality categoricals. Feature engineering (Phase 2) should close it.

**Note (session 5):** `models/lgb_metrics.json` had drifted out of sync with this
table — it held results from the earlier `num_leaves=63` experiment referenced
above (val PR-AUC 0.531) rather than the final `num_leaves=127` config already in
`pipeline.py`. Confirmed via `git diff` that `pipeline.py`/`utils.py` were
unchanged since commit `7938f8a`, so this was a stale artifact, not a code
regression. Fixed by re-running `phase1_baseline/pipeline.py`, which reproduced
this table's numbers exactly (val ROC-AUC 0.9125, PR-AUC 0.5464, best round 89).

---

## Phase 2 Tier 1 — Within-Row Feature Engineering (Results)

`phase2_feature_engineering/feature_engineering.py` + `pipeline.py`. Adds 10
within-row features to each fold before the unchanged Phase 1 `Preprocessor`
runs: `hour_of_day`, `day_of_week`, `local_hour`, `has_true_local_hour`,
`log1p_TransactionAmt`, `is_round_amount`, `P_email_matches_R_email`,
`P_email_is_free`, `browser_name`, `os_name`. No fitting involved in any of
these — each is a deterministic per-row function, so they're computed
independently on train/val/test with no leakage risk.

**Assumption confirmed with user:** raw `id_30`/`id_31` are kept alongside the
parsed `os_name`/`browser_name` family columns rather than replaced, so
version-specific signal (e.g. an outdated browser correlating with fraud)
isn't discarded. Free-email prefix list and browser/OS family buckets were
built from the actual unique values in the training data, not guessed.

**`local_hour` coverage is thinner than it looks:** `id_14` (the timezone
offset used to correct `TransactionDT` to local wall-clock time) is populated
for only **13.6%** of all rows, not the ~24% that have any identity record —
most identity rows lack `id_14` specifically. `local_hour` falls back to a
copy of `hour_of_day` for the other 86.4%. `has_true_local_hour` (`id_14`
notnull) was added so the model can separate the two populations rather than
rediscover the distinction indirectly. See ablation results below — it earns
its place.

### Methodology note: feature-count sensitivity in subsampled training

Discovered while testing `has_true_local_hour` in isolation: LightGBM's
`feature_fraction=0.8` and XGBoost's `colsample_bytree=0.8` randomly
subsample columns each round, seeded by a fixed seed. That seed only
reproduces the same draw for a fixed column count — adding or removing even
one column reshuffles the entire subsequent random sampling sequence,
producing a materially different model for reasons unrelated to whether the
added column carries information. Symptom: adding the single
`has_true_local_hour` column moved LightGBM's early-stopping round from 122
to 67 and its val PR-AUC from 0.5579 to 0.5390 in the production config —
looked like the flag actively hurt, which turned out to be an artifact.

**Fix:** `train_lgb`/`train_xgb` (`phase1_baseline/pipeline.py`) now accept
`feature_fraction`/`colsample_bytree` overrides. Setting both to `1.0`
disables column subsampling, which removes this confound (row subsampling —
`bagging_fraction`/`subsample` — is untouched since row count doesn't change
across feature-engineering configs, so it isn't a confound). This ablation
mode (`phase2_feature_engineering/ablation_check.py`) is now the standard
way to test whether a new feature genuinely helps, before trusting a
production-config (0.8 subsampling) comparison. **Adopt this for Tier 2**,
where many more columns will be added at once.

### Controlled ablation results (subsampling disabled — trustworthy)

| Config | Model | ROC-AUC | PR-AUC |
|---|---|---|---|
| Phase 1 (raw) | LightGBM | 0.9037 | 0.5435 |
| Phase 1 (raw) | XGBoost | 0.9190 | 0.5697 |
| Tier 1, no `has_true_local_hour` | LightGBM | 0.9068 | 0.5326 |
| Tier 1, no `has_true_local_hour` | XGBoost | 0.9173 | 0.5737 |
| Tier 1, with `has_true_local_hour` | LightGBM | 0.9072 | **0.5509** |
| Tier 1, with `has_true_local_hour` | XGBoost | 0.9173 | 0.5737 |

Two findings, both clean under this control:
- **`has_true_local_hour` genuinely helps LightGBM** (+0.0183 PR-AUC over
  Tier 1 without it) and is a **complete no-op for XGBoost** — predictions
  are bit-identical with or without it. Plausible read: LightGBM's
  histogram-based splitting benefits from the explicit flag; XGBoost already
  extracts the same information some other way (likely splitting directly
  on `id_14`'s own missingness). Kept the flag.
- **Tier 1's true marginal value over raw Phase 1 is smaller than the
  noisy production-config numbers suggested**: LightGBM +0.0074 PR-AUC,
  XGBoost +0.0040 PR-AUC. Real, but modest — consistent with most
  predictive signal already living in the anonymized V columns.

### Production-config results (validation set, 0.8 subsampling — current shipped model)

| Model | Metric | Phase 1 | Phase 2 Tier 1 | Delta |
|---|---|---|---|---|
| LightGBM | ROC-AUC | 0.9125 | 0.9082 | -0.0043 |
| LightGBM | PR-AUC | 0.5464 | 0.5390 | -0.0074 |
| XGBoost | ROC-AUC | 0.9170 | 0.9180 | +0.0010 |
| XGBoost | PR-AUC | 0.5714 | 0.5778 | +0.0064 |

Take this table with the methodology note above in mind — at this feature-
count scale, single-seed deltas of ±0.01-0.02 PR-AUC are within the noise
band created by subsampling reshuffling, not necessarily real effects. The
ablation table is the trustworthy read on whether Tier 1 helps; this table
is the actual current production checkpoint, included for completeness.

**Train-val gap** widened for LightGBM in the production config (train
PR-AUC 0.864 → ~0.83-0.91 depending on run, val PR-AUC roughly flat) —
consistent with early-stopping round also swinging with the subsampling
noise (89 → 67-122 depending on exact feature set). XGBoost's gap moved
much less. Worth retuning `num_leaves`/`min_child_samples` before drawing
firm conclusions about overfitting from any single production run.

**Feature importance:** none of the 10 new features reached LightGBM's top
15. `P_email_matches_R_email` reached XGBoost's top 12 — the one new feature
with a clearly visible individual contribution across runs.

**Interpretation:** Tier 1 features are a net positive on the metric that
matters (PR-AUC) but a modest one, not a step change — confirmed by the
clean ablation, not just the noisier production numbers. Card-level velocity
aggregates (Tier 2) are the more likely source of a larger gain, if the C
columns turn out not to already cover it.

---

## Phase 2 Tier 2 — Card-Level Aggregates (Results)

`phase2_feature_engineering/tier2_features.py`, `tier2_triage.py`,
`tier2_ablation_check.py`, `uid_validation.py`. Unlike Tier 1, these
features depend on each entity's transaction history, so they're computed
on the full dataset (sorted by entity, then time) before the train/val/test
split — not per-fold like the Preprocessor. Every feature is causal
(expanding-window, strictly-prior-transactions-only); see `tier2_triage.py`'s
docstring for why computing this before the split is legitimate rather than
a leak.

### v1/v2: card1+card2+card3+card5 combo entity — no reliable value

First entity key tried: `card1+card2+card3+card5`, deliberately excluding
`addr1` (folding it into the key would make an address change look like a
brand-new entity — history resets to zero — instead of a flagged event on a
continuous account).

**Triage (correlation against existing C/D columns, train fold):** all
candidates showed low correlation with the 14 C and 15 D columns (max 0.08–0.27)
— not redundant — but also weak correlation with `isFraud` itself (|r| < 0.02
for three of four v1 candidates). One exception: `addr1_changed_from_prev`
correlated -0.056 with `isFraud`, confirmed non-confounded by a 3-way check
excluding each entity's first transaction (fraud rate 4.47% when addr1
unchanged vs. 2.30% when changed, train fold) — the opposite direction from
the "address change = account takeover" hypothesis going in; a more likely
(not certain) explanation is that repeat use of a stolen card doesn't require
re-entering shipping info, so "same address, repeat transactions" looks more
like sustained fraud than "address changed."

**v1 (4 features: prior count, time-since-last, amount z-score,
addr1-changed) and v2 (+ `addr1_change_x_inverse_time`, `amt_ratio_vs_prev`
interaction features) both showed no reliable value** under controlled
ablation (`feature_fraction=1.0`/`colsample_bytree=1.0`, isolating the
feature-count/subsampling confound documented in the Tier 1 section):

| Config | LightGBM ROC/PR | XGBoost ROC/PR |
|---|---|---|
| Phase 1 (raw) | 0.9037 / 0.5435 | 0.9190 / 0.5697 |
| Tier 2 v1 (4 feat) alone | 0.9061 / 0.5287 | 0.9139 / 0.5724 |
| Tier 1 + Tier 2 v1 | 0.9080 / 0.5561 | 0.9182 / 0.5705 |
| Tier 2 v2 (6 feat) alone | 0.9061 / 0.5363 | 0.9163 / 0.5697 |
| Tier 1 + Tier 2 v2 | 0.9051 / 0.5326 | 0.9194 / 0.5716 |

No consistent direction across models or combinations, and **none of the six
features ever cracked either model's top-10 importance** across any
configuration. Read at the time: this doesn't mean the feature *concepts*
are uninformative, it means the entity key itself is probably too noisy to
expose whatever signal exists.

### Diagnosis: the entity key was badly conflated

`uid_validation.py` tested a general community idea about this competition
(not derived from reading anyone's submitted code, validated independently
against this project's own data before use): `D1` behaves like "days since
this account was first seen," so `transaction_day - D1` should be roughly
constant per real account even when masked card fragments collide across
different real customers.

Result: **590,540 rows collapse into only 14,845 distinct
`card1+card2+card3+card5` combos** (~40 transactions per "entity" on
average — implausibly high for genuine individual accounts over 182 days).
Of the 10,765 combos with more than one transaction, **84.1% split into
multiple distinct D1-adjusted start-days**. Cohesion check: raw combo
groups average **3.9 distinct `addr1` values** (several different
households); splitting further by start-day drops that to **0.94** —
essentially one address per sub-group, as expected for a genuine single
account. `D1` is ~99.8% populated, so this isn't an `id_14`-style coverage
problem.

### v3: refined entity key (`card1+card2+card3+card5+start_day`)

Same six feature concepts, same causal/expanding-window construction, now
grouped by the refined key:

| Config | LightGBM ROC/PR | XGBoost ROC/PR |
|---|---|---|
| Tier 2 v3 alone | 0.9060 / 0.5462 | 0.9178 / 0.5693 |
| Tier 1 + Tier 2 v3 | 0.9056 / 0.5289 | **0.9189 / 0.5797** |

First Tier 2 variant to beat the raw baseline for LightGBM alone (0.5462 vs.
0.5435). XGBoost's combined-stack PR-AUC (0.5797) is the best of the entire
Phase 2 investigation.

**Extended feature-importance check (`tier2_v3_importance_check.py`) — the
full picture is more nuanced than "did it crack top-10":**

| Feature | LightGBM rank (of 530) | XGBoost rank (of 530) |
|---|---|---|
| `combo_time_since_last` | #37 | #185 |
| `combo_prior_txn_count` | #38 | #188 |
| `combo_amt_zscore` | #54 | #252 |
| `addr1_change_x_inverse_time` | #55 | #226 |
| `amt_ratio_vs_prev` | #67 | #246 |
| `addr1_changed_from_prev` | #272 | #250 |

**LightGBM genuinely uses these features** — 5 of 6 land in the top 13% of
all features, a real (if modest) contribution that "never cracked top-10"
undersold. `addr1_change_x_inverse_time` (#55) massively outranks the raw
`addr1_changed_from_prev` flag (#272), validating the choice to build the
continuous interaction explicitly rather than leave it for the trees to
reconstruct from the two raw features.

**XGBoost's evidence is weaker despite the better aggregate number** — all
six features rank bottom-half (#185–252), tiny importance values. The 0.5797
PR-AUC is real, but there isn't strong feature-importance evidence that
Tier 2 specifically caused it; could be a genuine thinly-distributed
contribution, or normal run-to-run variance that happens to coincide with
this feature set. Documented as an open question rather than claimed as a
clean win for both models.

### Error analysis (`error_analysis.py`): the blind spot is concentrated, not random

Data-driven rather than hypothesis-driven — compared the hardest-to-catch
fraud (bottom quartile of predicted probability, Tier 1 + Tier 2 v3 models)
against the easiest-to-catch:

| | Hardest misses | Easiest catches |
|---|---|---|
| ProductCD = W | ~73–74% | ~1–3% |
| ProductCD = C | ~17–19% | ~82% |
| `has_identity` rate | ~25–27% | ~98–99% |
| `is_round_amount` rate | ~59–60% | ~19–20% |
| Mean TransactionAmt | ~185–194 | ~90–93 |
| `addr1_changed_from_prev` rate | ~15–16% | ~5.3–5.4% |

Same pattern for both models independently. The model is excellent when
identity/device data is present (98–99% of easy catches have it) and close
to blind when it's absent — consistent with the Phase 1 EDA finding that a
large share of V-columns only exist when identity data is present, so a
non-identity transaction is genuinely missing a chunk of the feature space,
not just lacking a feature we haven't built yet. Two more specific
findings: **higher-dollar fraud is harder to catch** (mean amount ~190 in
misses vs. ~90 in catches — the more expensive mistake), and **round-dollar
amounts are much harder to catch** (60% of misses vs. 20% of catches).
`addr1_changed_from_prev` is higher among misses than catches even though it
predicts *lower* fraud rate overall — coherent, since address-change fraud
structurally resembles a legitimate multi-location shopper, making it
genuinely harder to distinguish on the rare occasions it does happen.

**Model overlap:** 66.5% of each model's hardest-quartile misses are shared
— a real, common blind spot neither model resolves alone. The remaining
~33% are model-specific, which an ensemble could plausibly rescue, but
wouldn't touch the majority shared blind spot.

Signals available specifically for the non-identity population that haven't
had dedicated feature engineering yet: `dist1` (present only in non-identity
transactions) and the V-group that's 16.2–18% missing (correlates -0.64 to
-0.72 with identity — "the non-identity transaction path" per the Phase 1
EDA). Flagged as a possible next angle, not yet pursued.

### Segmented model (session 7): has_identity=1 vs. has_identity=0 — RESULTS

Motivated directly by the error analysis: identity presence is the single
dominant driver of catchability, and a single global model may be
under-serving the harder segment. Mechanism: gradient boosting takes the
splits that reduce loss most; if the identity-present population offers
bigger, easier gains per split (richer signal), the model's limited round
budget gets pulled toward optimizing it, and early stopping triggers on the
*aggregate* metric — which can plateau because the easy segment is already
well-fit, even while the hard segment still has headroom. This is a known
issue with heterogeneous subpopulations in one global model, not specific to
this dataset.

**Segment fraud rates confirm a real divergence** (`segment_analysis.py`,
train fold): `has_identity=1` → 6.73%, `has_identity=0` → 2.12% (3.2x). But
it isn't temporally stable:

| Fold | `has_identity=1` share | `has_identity=1` rate | `has_identity=0` rate |
|---|---|---|---|
| Train | 27.5% | 6.73% | 2.12% |
| Val | 19.6% | 11.02% | 2.17% |
| Test | 20.1% | 9.34% | 1.96% |

`has_identity=0` is stable (~2.0–2.2% throughout). `has_identity=1` nearly
doubles from train to val — initially read as a concerning calibration risk
for a segmented model, since PLAN.md's existing note about val's slightly
elevated aggregate fraud rate (3.38%→3.90%) badly understated how much of
that instability concentrates in this one segment.

**Within-train chronological quarters clarified this rather than deepening the
concern:**

| Quarter | `has_identity` share | `has_identity=1` rate | `has_identity=0` rate |
|---|---|---|---|
| Q1 | 39.7% | 3.60% | 1.99% |
| Q2 | 35.2% | 5.18% | 1.89% |
| Q3 | 17.7% | 13.48% | 2.33% |
| Q4 | 17.3% | 10.14% | 2.18% |
| Val | 19.6% | 11.02% | 2.17% |

There's a regime shift midway through train (Q2→Q3): identity-capture share
roughly halves while the identity-segment fraud rate more than doubles —
plausibly a shift toward more selective, risk-based identity verification,
though that's a plausible read, not a confirmed mechanism. Critically, **val
looks like a continuation of Q3/Q4, not a new surprise** — the apparent
"6.73%→11.02% drift" is an artifact of comparing val against train's
*blended* average, which is diluted by Q1/Q2's outdated, larger-share/
lower-rate regime.

Implemented in `phase2_feature_engineering/segmented_pipeline.py`. Trains
three LightGBM/XGBoost pairs — global (unsegmented), `has_identity=1`
segment, `has_identity=0` segment — all on the identical Tier 1 + Tier 2
feature set, all in ablation mode (`feature_fraction=1.0`/
`colsample_bytree=1.0`) so the comparison isn't confounded by the column-
subsampling reshuffle documented in the Tier 1 section.

**Recency weighting (user decision, session 7):** rather than a hard
Q1/Q2 cutoff, the `has_identity=1` segment's training rows get an
exponential recency weight — `0.5 ** (age_days / 30)`, rescaled to mean
1.0 — so old-regime rows are downweighted, not discarded. Diagnostic
confirms the intended shape (mean weight by chronological quarter of the
segment): Q1 0.432, Q2 0.556, Q3 0.820, Q4 2.193. `has_identity=0` showed
no such regime shift and trains unweighted.

**First pass — pooled combined metric was misleading.** Concatenating
both segments' val predictions into one pooled ROC-AUC/PR-AUC made
segmentation look strictly worse than the global model (LightGBM PR-AUC
0.5268 vs. 0.5289 global; XGBoost 0.5622 vs. 0.5797 global). A calibration
check caught why before this was accepted: `is_unbalance` (LightGBM) and
`scale_pos_weight` (XGBoost) are computed per segment from that segment's
own class ratio, and `has_identity=0`'s ratio (~46:1) is far more extreme
than `has_identity=1`'s (~14:1). Mean predicted probability vs. actual
fraud rate confirms the two segments' models are not on a comparable
scale: `has_identity=1` LightGBM 1.06x actual rate, XGBoost 1.66x —
`has_identity=0` LightGBM 4.50x, XGBoost 5.52x. Pooling two differently-
inflated probability scales into one ranking metric is not a valid
comparison, regardless of how good either segment model actually is.

**Fair comparison (correct methodology):** score the global model and
each segment model separately, restricted to that segment's own val rows,
so no cross-segment pooling occurs and each model is judged purely on its
own internally-consistent scale.

| Segment | Model | Global (on segment rows) ROC/PR | Segment model ROC/PR | Verdict |
|---|---|---|---|---|
| has_identity=1 | LightGBM | 0.9288 / 0.7244 | 0.9314 / 0.7565 | Segment wins both |
| has_identity=1 | XGBoost  | 0.9400 / 0.7852 | 0.9450 / 0.7937 | Segment wins both |
| has_identity=0 | LightGBM | 0.8609 / 0.2210 | 0.8489 / 0.2613 | Mixed (ROC down, PR up) |
| has_identity=0 | XGBoost  | 0.8815 / 0.2451 | 0.8471 / 0.2359 | Global wins both |

`has_identity=1` segmentation is a clean, unambiguous win on both models
and both metrics — consistent with the error-analysis finding that this
population carries richer signal (device/identity V-columns) that a
dedicated model can sharpen without competing for boosting-round budget
against the much larger `has_identity=0` population.

**`has_identity=0` tuning investigation (`segment_id0_tuning.py`):** before
accepting the mixed/negative id0 result, tested whether the segment
model's complexity (127 leaves / depth 6, tuned implicitly for the full
354k-row dataset) was overfitting a segment with fewer fraud examples
(5,440 vs. the global model's 11,988).

- **LightGBM: regularizing further made it worse** in both directions
  tried (63 leaves: ROC 0.8463/PR 0.2440; 31 leaves: ROC 0.8424/PR 0.2013,
  both below the 127-leaf baseline's 0.8489/0.2613) — the original config
  was not overfitting; it's already close to this segment's standalone
  ceiling.
- **XGBoost: regularizing did help**, confirming depth 6 was overfitting
  this segment — depth 4 recovered real ground (ROC 0.8471→0.8634, PR
  0.2359→0.2260) — but even the best-tuned config still falls short of
  the global model's numbers on both metrics (0.8815/0.2451).

No `has_identity=0`-specific configuration, in either direction, beat the
global model. Read: `has_identity=0` doesn't carry enough self-contained
distinguishing signal to benefit from specialization, and pooling it with
`has_identity=1`'s data during training acts as a helpful regularizer/
diversifier a segment-only model can't replicate by tuning alone —
consistent with `error_analysis.py`'s original finding that this
population is a shared, information-limited blind spot rather than a
modeling shortfall.

**Architecture decision revised again (session 8) — see "Inference pipeline"
below.** The hybrid-vs-global comparison above was run under the pre-SMOTE,
recency-weighted config. Wiring up real inference re-ran it under the
current SMOTE 1:10 + tuned config and found the two algorithms now
disagree: XGBoost still prefers the segment model, LightGBM no longer does.
"Hybrid, not full segmentation" is no longer the whole story — it's now
"hybrid for XGBoost, global-only for LightGBM." Left this section's numbers
as-is (they're the historical record of how the decision was reached) but
they should not be read as the current production routing on their own.

**Architecture decision: hybrid, not full segmentation.** Route
`has_identity=1` transactions to the dedicated segment model; keep using
the **global** (unsegmented) Tier 1 + Tier 2 model for `has_identity=0`
transactions, since no segment-specific variant beat it there. The
`has_identity=0` segment model artifacts (`*_seg_id0*`) are kept on disk
for reference but are not the production choice for that population.

### SMOTE ablation and production config revision (session 7 continued)

`smote_ablation.py` tested SMOTE (via `SMOTENC`, not plain SMOTE — see its
docstring for why plain SMOTE would corrupt one-hot/categorical/binary-flag
columns) against the current class-weighting approach for both production
models, at three ratios (1:10, 1:5, 1:3), in ablation mode.

| Model | Config | ROC-AUC | PR-AUC |
|---|---|---|---|
| Global, LightGBM | Baseline (class-weight) | 0.9056 | 0.5289 |
| Global, LightGBM | **SMOTE 1:10** | 0.9178 | **0.5852** |
| Global, LightGBM | SMOTE 1:5 | 0.9161 | 0.5859 |
| Global, LightGBM | SMOTE 1:3 | 0.9153 | 0.5769 |
| Global, XGBoost | Baseline (class-weight) | 0.9189 | 0.5797 |
| Global, XGBoost | **SMOTE 1:10** | 0.9197 | **0.6072** |
| Global, XGBoost | SMOTE 1:5 | 0.9161 | 0.5972 |
| Global, XGBoost | SMOTE 1:3 | 0.9177 | 0.6004 |

SMOTE 1:10 is a real peak, not a monotonic "more oversampling is better"
curve — 1:5 and 1:3 both underperform it on PR-AUC for LightGBM, and are
mixed for XGBoost. The PR-AUC gain over class-weighting alone (LightGBM
+0.056, XGBoost +0.027) is larger than any single feature-engineering gain
found in this project so far.

**Unexpected finding: recency-weighting doesn't hold up under a clean
ablation.** This script's `has_identity=1` baseline arm (class-weighting,
no recency-weight, no SMOTE, otherwise identical config) scored LightGBM
0.9335/0.7616 and XGBoost 0.9434/0.7964 — *better* than the recency-weighted
production numbers reported earlier this session (0.9314/0.7565 LightGBM;
0.9450/0.7937 XGBoost, mixed). Recency-weighting was adopted on a plausible
mechanism (val looks like a continuation of the Q3/Q4 regime) but was never
actually tested with-vs-without, holding everything else fixed, until this
script's baseline arm did so as a side effect. The honest read: it's a wash
to slightly negative for this segment, not the improvement originally
assumed. Plausible explanation: reweighting toward Q4 reduces the
*effective* sample size (Kish's effective-N shrinks whenever weights vary,
even though they're rescaled to mean 1.0) at a cost that isn't clearly
paid back, especially since Q3+Q4 already make up half of this segment's
training rows even unweighted.

SMOTE 1:10 alone (no recency-weight at all) beats the recency-weighted
config outright: `has_identity=1` SMOTE 1:10 scored LightGBM 0.9410/0.7705
and XGBoost 0.9491/0.8072 — better than recency-weighting's 0.9314/0.7565
and 0.9450/0.7937 on every number.

**Revised production config (`finalize_production_models.py`):** both the
global model and the `has_identity=1` segment model now use **SMOTE 1:10
(via SMOTENC), with class-weighting and recency-weighting both dropped**.
Artifacts saved: `models/lgb_global_tier1tier2.txt` +
`models/xgb_global_tier1tier2.pkl` + `models/preprocessor_global_tier1tier2.pkl`
(global), `models/lgb_seg_id1.txt` + `models/xgb_seg_id1.pkl` +
`models/preprocessor_seg_id1.pkl` (has_identity=1 segment) — these
overwrite the earlier class-weighted/recency-weighted artifacts of the same
names. Metrics saved to `models/production_metrics.json`; the earlier
`models/segmented_metrics.json` is left as-is, documenting the segmentation
investigation's history rather than the final production numbers.

**Flagged for later, not yet tested:** whether SMOTE 1:10 combined with
recency-weighting beats SMOTE 1:10 alone — requires deciding how synthetic
SMOTE rows should inherit a recency weight (e.g. from the parent/neighbor
row they were interpolated from), which adds real complexity for an
unconfirmed extra gain. Deferred until/unless there's a specific reason to
revisit it.

### Hyperparameter tuning (session 7 continued)

`hyperparameter_tuning.py` ran a modest randomized search (6 draws per
model/algorithm, not an exhaustive grid — deliberately kept small; this is
"tune if time allows" per the project's stated priority, not the main
event) on top of the now-fixed SMOTE 1:10 resampling, for both production
models. Feature/row subsampling (`feature_fraction`, `bagging_fraction`,
`subsample`, `colsample_bytree`) were back in the search space here —
unlike the ablation-mode scripts elsewhere in this phase, this search
tunes hyperparameters on one fixed, already-decided feature set and
resampling scheme, so subsampling is a legitimate regularization knob here,
not the feature-comparison confound it would be in an A/B test.

| Model | Algorithm | SMOTE-1:10 default | Best of 6 trials | ROC-AUC Δ | PR-AUC Δ |
|---|---|---|---|---|---|
| Global | LightGBM | 0.9178 / 0.5852 | 0.9164 / 0.5974 | −0.0014 | +0.0122 |
| Global | XGBoost | 0.9197 / 0.6072 | 0.9247 / 0.6144 | +0.0050 | +0.0072 |
| has_identity=1 | LightGBM | 0.9410 / 0.7705 | 0.9423 / 0.7789 | +0.0013 | +0.0084 |
| has_identity=1 | XGBoost | 0.9491 / 0.8072 | 0.9486 / 0.8114 | −0.0005 | +0.0042 |

All four trials found a PR-AUC improvement (the project's primary metric).
Caveat worth stating plainly: picking the best of 6 trials by val PR-AUC
carries a mild "winner's curse" — the reported number is somewhat
optimistic relative to true generalization, proportionally more so for a
thin margin than a wide one.

**Decision: adopted 3 of 4** (`finalize_tuned_models.py`) — Global
LightGBM (leaves=63, min_child_samples=20, lr=0.1, L2=1,
feature_fraction=0.6, bagging_fraction=1.0), Global XGBoost (depth=8,
lr=0.05, min_child_weight=1, L1=1, L2=1, subsample=0.6, colsample=1.0),
and has_identity=1 LightGBM (leaves=127, min_child_samples=20, lr=0.05,
feature_fraction=0.8, bagging_fraction=0.8). **has_identity=1 XGBoost kept
at its SMOTE-1:10 default** — its best trial's margin (ROC −0.0005 / PR
+0.0042) was judged too thin, relative to the 6-trial selection bias, to
be worth adopting. These are the current production hyperparameters;
numbers reconfirmed in `models/production_metrics.json`.

### Inference pipeline (session 8): hybrid architecture re-validated end-to-end, and revised

`phase2_feature_engineering/inference.py` wires up the callable pipeline
that was still missing after session 7 — a `HybridModel` class that loads
both production model pairs once (`preprocessor_global_tier1tier2` +
`lgb_global_tier1tier2.txt`/`xgb_global_tier1tier2.pkl`, and
`preprocessor_seg_id1` + `lgb_seg_id1.txt`/`xgb_seg_id1.pkl`) and a
`build_features(df)` helper that reproduces the exact feature order used
at training time (`has_identity` → `add_entity_velocity_features` →
`add_tier1_features`) on the full dataset before any split, since Tier 2's
entity-velocity features need each entity's complete prior history and
cannot be computed for a single transaction in isolation. **Real-time
scoring limitation, worth naming in the write-up:** a production service
would need a persisted per-entity running-state store (last transaction
time, running mean/std, last `addr1`) rather than recomputing from raw
history on every call — out of scope here, but a real gap between this
project's inference code and an actual deployable service.

First run reproduced `production_metrics.json`'s `segment_id1` numbers
exactly (ROC/PR to 4 decimals for both LightGBM and XGBoost) — confirms the
routing and preprocessing wiring is correct, not just plausible.

**Then the first true full-val, routed metric — never actually computed
before this — showed hybrid routing is a net negative for LightGBM under
the current config:**

| Algorithm | Global model, has_identity=1 rows only | Segment model, has_identity=1 rows only | Winner |
|---|---|---|---|
| LightGBM | PR-AUC **0.7868** | PR-AUC 0.7789 | Global (reversed from session 7's finding) |
| XGBoost | PR-AUC 0.8022 | PR-AUC **0.8072** | Segment (same direction as session 7, smaller margin: 0.0085 → 0.0050) |

The session-7 "fair comparison" that justified the hybrid architecture
predates SMOTE 1:10 and hyperparameter tuning — it was never re-run after
those changed both the global and segment models. Once both share the same
resampling/tuning treatment, the global LightGBM model's larger training
set now outweighs the segment model's specialization on this subset —
the same regularizer/diversifier effect already documented for
`has_identity=0`, apparently generalizing to `has_identity=1` for this one
algorithm now that the segment models are less differently-tuned than
before.

**User decision (session 8): algorithm-specific routing, not one fixed
architecture.** `HybridModel.predict(df, algorithm=...)` routes
differently depending on which algorithm is selected:
- `algorithm="xgb"`: hybrid routing kept — `has_identity==1` → seg_id1
  model, `has_identity==0` → global model.
- `algorithm="lgb"`: global model for every row, no routing — the segment
  model is not used in production for LightGBM at all.

**Final production val numbers (`models/hybrid_val_metrics.json`), full val
set, each algorithm scored the way it will actually run in production:**

| Algorithm | Routing | ROC-AUC | PR-AUC |
|---|---|---|---|
| LightGBM | Global model, all rows | 0.9164 | 0.5974 |
| XGBoost | Hybrid (segment model for has_identity=1) | **0.9257** | **0.6168** |

XGBoost with hybrid routing is the best full-val PR-AUC of the entire
project. Calibration check (mean predicted probability vs. actual fraud
rate, current SMOTE 1:10 config) shows both segments landing in the same
0.75–0.90x ballpark for both algorithms — a much closer match than the
4.5–5.5x vs. 1–1.7x mismatch found under the old class-weighted config in
session 7, consistent with SMOTE 1:10 giving both models a more comparable
probability scale and making a single pooled/full-val metric a valid read
now (not just a per-segment one).

Test set NOT touched by any of this — `inference.py`'s `main()` validates
against train+val only, per the "touch test once" rule. The one-time test
confirmation belongs to the Cost-Sensitive Decision Framework's threshold
step below.

**Superseded by the bug fix and ensemble work directly below** — the table
above predates both. See "Model divergence analysis" and "Ensemble" for the
current numbers and the final production default.

### Model divergence analysis (session 8 continued): where LightGBM and XGBoost actually disagree

Prompted by a step-back question: given two production algorithms, are
their strengths different enough for that to matter, and where specifically?
`phase2_feature_engineering/model_divergence_analysis.py` scores the full
val set with each algorithm's actual production routing (not a fresh
ablation retrain) and compares by **rank percentile** rather than raw
probability, since the two algorithms sit on different scales after
algorithm-specific routing.

Spearman rank correlation across the full val set: **0.745** — substantial
agreement, but far from 1.0. Among the 4,611 val fraud cases (top-decile
"catch" vs. bottom-half "miss" thresholds):

| Outcome | Count | % |
|---|---|---|
| Both catch | 3,144 | 68.2% |
| Neither catches (shared blind spot) | 77 | 1.7% |
| XGBoost catches, LightGBM misses | 17 | 0.4% |
| LightGBM catches, XGBoost misses | 7 | 0.2% |

The shared blind spot is 84% `ProductCD=W` with only 14% `has_identity` —
the same non-identity blind spot `error_analysis.py` found originally, now
confirmed genuinely shared rather than one model quietly covering for the
other.

**Traced a concrete bug from the top-divergence list, not just a vague
"models differ."** The 10 legitimate transactions with the largest
model disagreement included exact repeats: `card1=14675` charging **$37.34
three separate times**, `card1=16819` charging **$49.00 twice**. LightGBM
scored these near-zero risk (percentile 0.01–0.07); XGBoost scored them
0.85–0.92. Root cause: `tier2_features.py`'s `combo_amt_zscore` divided by
a prior standard deviation of exactly 0 (an entity whose prior transactions
are all identical amounts — precisely what a recurring subscription looks
like) and produced `NaN` rather than a meaningful value, and the two
algorithms' native missing-value handling diverged sharply on that one
engineered column.

**Fix (`tier2_features.py`):** floor the prior std at 1% of the prior mean
(minimum 1 cent) before dividing, instead of replacing exactly-0 std with
`NaN`. An exact repeat still zscores to 0 (no deviation from a
hyper-consistent history); any deviation from a hyper-consistent history
now produces a large, correctly-signed value instead of erasing the signal.
`combo_prior_txn_count` already tells the model how much history backs this
number, so weak evidence at low counts isn't hidden — it's left for the
model to weight.

**Retrained both production model pairs** (`finalize_production_models.py`
then `finalize_tuned_models.py`) on the corrected feature. Net effect:
XGBoost improved, LightGBM regressed slightly — the bug had been
disproportionately confusing XGBoost, consistent with it being the model
whose false-positive scores on the recurring-payment rows were most
extreme before the fix.

| Model | Metric | Pre-fix | Post-fix | Delta |
|---|---|---|---|---|
| Global LightGBM (tuned) | PR-AUC | 0.5974 | 0.5875 | −0.0099 |
| Global XGBoost (tuned) | PR-AUC | 0.6144 | 0.6174 | +0.0030 |
| has_identity=1 LightGBM (tuned) | PR-AUC | 0.7789 | 0.7751 | −0.0038 |
| has_identity=1 XGBoost (unchanged config) | PR-AUC | 0.8072 | 0.8084 | +0.0012 |

Re-ran the LightGBM global-vs-segment routing check under the fixed
feature before assuming the session-8 routing decision still held: global
still beats segment on PR-AUC (0.7824 vs 0.7751) — same direction as
before the fix, so **no routing change needed**, but this was verified,
not assumed.

Re-running `model_divergence_analysis.py` post-fix confirmed the specific
$37.34/$49.00 repeat-amount false positives are gone from the top-10 most
divergent legitimate transactions, though a few structurally similar
low-history cases remain (expected — the fix makes low-count z-scores more
informative, not silent, so genuinely weak evidence still occasionally
produces a large value). Post-fix agreement breakdown shifted slightly:
both-catch 67.5%, neither-catches 2.0%, XGBoost-only 19 (was 17),
LightGBM-only dropped to 1 (was 7) — LightGBM now contributes almost no
*exclusive* fraud catches, which sets up the ensemble finding below.

Saved: `phase2_feature_engineering/model_divergence.html` +
`model_divergence.json`.

### Ensemble (session 8 continued): blending beats either algorithm alone — new production default

Divergence alone doesn't prove blending helps — `ensemble_test.py` tested
it directly, sweeping a weighted average `w * xgb_prob + (1-w) * lgb_prob`
from `w=0.0` to `w=1.0` in steps of 0.05, on the full val set, using each
algorithm's actual production-routed output (not a fresh retrain).

| Config | ROC-AUC | PR-AUC |
|---|---|---|
| LightGBM alone (`w=0.0`) | 0.9142 | 0.5875 |
| XGBoost alone (`w=1.0`) | 0.9282 | 0.6178 |
| 50/50 raw average (`w=0.5`) | 0.9314 | 0.6235 |
| **Best: `w=0.70`** | **0.9320** | **0.6265** |

+0.0086 PR-AUC over standalone XGBoost (the stronger single model) — the
largest single gain of any change made after the initial feature-engineering
phase. Not a fragile spike: PR-AUC stays within 0.6229–0.6265 across
`w=0.55`–`0.85`, a broad plateau rather than one lucky grid point.

Raw-probability averaging beat rank-percentile averaging across nearly the
whole sweep (best raw PR-AUC 0.6265 vs. best rank PR-AUC 0.6197) — the two
algorithms' calibration is close enough (0.75–0.91x mean-predicted/actual
ratio) that blending magnitude keeps information rank-blending throws away.

The improvement holds in both segments separately, not just in aggregate —
checked via `inference.py`'s per-segment breakdown so the aggregate number
isn't hiding an offsetting loss somewhere:

| Segment | LightGBM alone PR-AUC | XGBoost alone PR-AUC | Ensemble PR-AUC |
|---|---|---|---|
| has_identity=1 | 0.7824 | 0.8084 | **0.8127** |
| has_identity=0 | 0.2978 | 0.3183 | **0.3355** |

**Decision: the ensemble is now the production default.**
`HybridModel.predict(df, algorithm="ensemble")` (default parameter value)
runs both algorithms through their own routing above, then blends with the
fixed weight `_ENSEMBLE_WEIGHT_XGB = 0.70`. `algorithm="lgb"` and
`algorithm="xgb"` remain available as explicit single-model choices for the
dashboard's model selector. Weight sweep chart + underlying data saved to
`phase2_feature_engineering/ensemble_weight_sweep.html` / `.json`.

Note on what this ensemble is *not*: it's a fixed post-hoc blend of two
already-trained algorithms' final outputs, not a per-row conditional
("route this row to LightGBM, that row to XGBoost") and not a stacked
meta-learner. Every row gets scored by both algorithms and blended with the
same weight, unconditionally — simpler than either alternative, and what
was actually tested.

---

## Phase 2 — Cost-Sensitive Decision Framework

Same structure as prior project (`credit_card_fraud/phase2_cost_analysis/`).

### Cost function
Structure carried forward from prior project:
- FP cost: `base_cost + (friction_pct × Amount)`
- FN cost: `Amount` (worst case — no chargeback recovery modeled)

Exact parameter values deferred — to be decided at Phase 2 start.
Currency is unknown; all figures are relative comparisons, not absolute dollars.

### Threshold optimization
- Sweep threshold 0.01–0.99 on **validation set**
- Find cost-minimizing threshold
- Apply to **test set once** to confirm
- Sensitivity analysis across FP cost assumptions

### Deliverable
Interactive Streamlit dashboard with:
- Model selector
- Threshold slider (real-time cost update)
- Operational reality panel (scaled to 1M transactions)
- Sensitivity analysis chart

Target: deployed to Streamlit Community Cloud.
Note: data size (~590k rows) may require pre-computing dashboard data at
model run time rather than loading raw data in the app. Confirm at build time.

### Cost function parameter values (session 9) — payment-processor lens

**Job-application context clarified this session:** the target role is
specifically a **payment processor** (merchant/issuer middleman) — not a
bank/issuer, not a merchant. The source data (Vesta) is itself this kind of
company. This reframes the FP/FN cost meaning relative to the prior project
(`credit_card_fraud`), which used a generic issuer-investigation lens.

**Distribution check before reusing prior values (`amt_stats.py`,
scratchpad):** train-fold `TransactionAmt` (mean $134, median $70, 95th pct
$441, 99th pct $1,092, max $31,937) is broadly comparable in scale to the
prior project's `Amount` (mean $88, median $22, 95th pct $365, 99th pct
$1,018, max $25,691) — tails converge, though this dataset's median runs 3x
higher. Crossover math (friction term overtakes base cost) isn't materially
different in shape either way, so nothing about the scale forces new values
on distribution grounds alone. Also notable: this dataset's fraud amounts
track legit amounts almost exactly (mean/median ratio ~1.04–1.11), unlike
the prior project's card-testing signature (fraud median half of legit) —
consistent with `error_analysis.py`'s earlier finding that fraud here gets
*harder* to catch at higher amounts, not concentrated in small probes.

**Decided values (`phase2_cost_analysis/cost_threshold_analysis.py`):**

```
FP cost = $3 + (3% x Amount)
FN cost = Amount
```

- **FN = Amount** is on firmer ground here than the prior project's "worst
  case, no recovery modeled" hedge: fraud-decisioning vendors in this market
  commonly sell a chargeback-guarantee product — if the vendor approves a
  transaction that turns out fraudulent, the vendor eats the loss, not the
  merchant. This is general industry knowledge about this vendor category,
  **not verified from this project's own data** — stated that way in every
  output, consistent with the project's "don't claim what we don't know"
  discipline (V-column content, etc.).
- **FP = $3 base + 3% friction** reframes the prior project's "$10
  investigation cost" (an issuer's analyst reviewing a flagged transaction
  after the fact) into a processor's own P&L: $3 covers automated
  dispute/support handling for a declined checkout; 3% approximates
  forfeited processing-fee revenue plus a modest allowance for
  merchant-retention risk (processors are chosen/dropped by their
  false-decline rate). Crossover (friction term overtakes base) lands at
  Amount=$100 here vs. $1,000 under the prior issuer framing — intentional,
  since a processor's own revenue is itself fee/percentage-based.

### Threshold optimization results (session 9)

Ran on the production `HybridModel` (`inference.py`), all three algorithm
choices (`lgb`, `xgb`, `ensemble`), 0.01–0.99 threshold sweep on val,
confirmed once on test.

| Algorithm | Val cost-optimal threshold | Val total cost | Test total cost (same threshold) |
|---|---|---|---|
| LightGBM | 0.03 | $280,537 | $305,142 |
| XGBoost | 0.03 | $260,113 | $279,752 |
| **Ensemble** | **0.03** | **$255,922** | $280,056 |

Ensemble wins on validation (as it does on PR-AUC); XGBoost edges it out by
a thin margin on test ($279,752 vs $280,056, ~0.1%) — normal noise at this
margin, not acted on, per the "choose on val, confirm on test" rule
(reported the val-chosen ensemble as production rather than re-picking
based on test).

**Notable finding: the cost-optimal threshold (0.03) is far more aggressive
than a typical real-world fraud tool** — it flags ~15.6% of all validation
transactions (FP=14,706, TP=3,672 of 118,108 rows), vs. e.g. the prior
project's 0.30 or industry-typical 1–5% review/decline rates. This is a
correct, not a buggy, consequence of the chosen cost ratio: average FN cost
(~$134, full Amount) vs. average FP cost (~$3 + 3%×$134 ≈ $7) is roughly
19:1 in favor of recall, so aggressive flagging genuinely minimizes total
dollar cost under this cost function.

**User decision (session 9): document as-is rather than add a capacity
constraint.** The linear, per-transaction cost function has no term for the
*aggregate* false-decline rate — each individual decline is priced
correctly, but a processor declining 1 in 6 legitimate checkouts would in
practice trigger merchant churn through a threshold/nonlinear effect this
model can't see (it only prices friction transaction-by-transaction, not
the reputational cost of a systematically high decline rate). Named
explicitly as a limitation in `cost_report.html` / `results_summary.md`
rather than hidden behind an unjustified capacity constraint (e.g. "cap
flag rate at 5%") that would need its own separate justification (review
team capacity? an industry benchmark?) not yet available. Flagged as a
candidate refinement if that justification later emerges.

**Sensitivity analysis (validation, ensemble):**

| FP base cost (friction fixed 3%) | Opt. threshold | Total cost |
|---|---|---|
| $1 | 0.02 | $215,445 |
| $3 | 0.03 | $255,922 |
| $5 | 0.04 | $278,336 |
| $10 | 0.06 | $314,678 |

| Friction rate (base fixed $3) | Opt. threshold | Total cost |
|---|---|---|
| 1% | 0.02 | $188,053 |
| 3% | 0.03 | $255,922 |
| 5% | 0.05 | $291,105 |
| 10% | 0.07 | $349,713 |

Threshold stays low (0.02–0.07) across a 10x range of both parameters —
the recall-favoring conclusion is not a fragile artifact of the exact $3/3%
choice, though the *degree* of aggressiveness (flag rate) scales with the
assumption.

**Operational interpretation (test set, ensemble, threshold 0.03, scaled to
1,000,000 transactions):** flags ~156,611 as fraud, of which ~26,061 are
genuine fraud caught and ~130,550 are false alarms. Estimated false-decline
cost ~$1,076,150 vs. ~$1,295,034 fraud prevented, net saving ~$2,793,024 vs.
no detection.

**Outputs:** `phase2_cost_analysis/cost_threshold_analysis.py` (full
reproducible pipeline), `cost_results.html` (interactive Plotly: cost
curves by algorithm, both sensitivity axes, cost breakdown at test,
errors-by-amount scatter), `cost_report.html` (stakeholder report),
`results_summary.md` (written summary), `models/cost_dashboard_data.json`
(feeds the planned Streamlit dashboard below — full val sweep + sensitivity
+ test results, all pre-computed so the dashboard doesn't need to reload
raw data or models).

**Next action:** build the Streamlit dashboard (model selector, threshold
slider, operational reality panel, sensitivity chart) reading from
`models/cost_dashboard_data.json`.

### Flagged for later: amount-weighted training (not yet pursued)

Raised (session 7): rather than only tuning a decision *threshold* against
the cost function, could training itself be made cost-aware by weighting
each fraud row's loss by its dollar amount (e.g. `log1p(Amount)` or
`sqrt(Amount)`, rescaled to mean 1 like the recency weights, to avoid one
large transaction dominating the loss the way unnormalized weights would)?

This is a real, named technique (cost-proportional/amount-weighted
training) and a genuinely different lever than threshold tuning — it
would push the tree structure itself toward separating high-value fraud,
not just any fraud. Two considerations before trying it:

- **Evaluation isn't ROC-AUC/PR-AUC** — those metrics are blind to dollar
  amounts entirely, so amount-weighted training can only be fairly judged
  against a dollar-denominated metric (e.g. total fraud-dollars caught,
  or expected cost) once the cost function below is actually defined.
- **Trades away the dashboard's real-time adjustability** — the
  threshold-sweep approach lets the cost-sensitivity analysis run against
  a single fixed trained model with no retraining; baking a specific cost
  assumption into training weights means any change to that assumption
  requires retraining, not just re-sweeping a slider.

Recommendation (not yet implemented): keep threshold optimization as the
primary mechanism, and treat amount-weighted training as a separate
ablation to test once the cost function's dollar metric exists to judge it
against.

---

## Git Workflow

- Branch `main` tracks production-ready work
- Feature work on branches, merged via PR on GitHub (no local merges)
- Never commit autonomously — confirm before every commit
- No Co-Authored-By trailers
- Commit by concern, not by session

**Current state (end of session 9):**
- main: Phase 1, Phase 2 Tier 1, **and** Phase 2 Tier 2/segmentation/
  ensemble all merged. PR #2 (`feature/phase2-feature-engineering`)
  merged 2026-07-06; **PR #3 (`feature/phase2-tier2-and-segmentation`)
  merged 2026-07-07** (merge commit, not squash — `gh pr merge 3 --merge`,
  matching PR #2's method). `models/production_metrics.json` and
  `models/hybrid_val_metrics.json` hold the final post-fix, post-ensemble
  numbers — see "Model divergence analysis" and "Ensemble" sections above.
- **`feature/phase2-cost-sensitivity`**: rebased onto the updated main
  right after PR #3 merged (clean, since the branch was still local-only
  and unpushed at that point — no force-push needed). 3 commits ahead of
  main, **still local-only, not yet pushed**:
  1. PLAN.md sync (carried over from session 8, was never actually part
     of PR #3 — it was cost-sensitivity's own first commit even before
     this session started).
  2. Cost-Sensitive Decision Framework: payment-processor cost lens (see
     "Cost function parameter values" section above),
     `phase2_cost_analysis/cost_threshold_analysis.py`, val threshold
     sweep + sensitivity analysis + one-time test confirmation across all
     three `HybridModel` algorithm choices. Ensemble wins on val
     (cost-optimal threshold 0.03); the aggressive resulting flag rate
     (~15.6%) documented as a named limitation rather than masked with an
     unjustified capacity constraint (user decision).
  3. `app.py` (Streamlit dashboard reading `models/cost_dashboard_data.json`)
     — model selector, live threshold/cost-parameter sliders (linear
     decomposition trick, generalized to two parameters across all three
     algorithms), sensitivity charts, operational reality panel. Verified
     end-to-end with a Playwright driver (no `chromium-cli` available in
     this environment) rather than just launched — captured as a project
     skill at `.claude/skills/run-ieee-fraud/` (unignored from the user's
     global `.claude/`-blocking gitignore via explicit negation rules in
     this project's `.gitignore`).
- **Phase 2's Cost-Sensitive Decision Framework is now complete** —
  cost function decided, threshold optimization + sensitivity analysis
  run, test set confirmed once, dashboard built and verified.
- **Next action:** not yet decided. Candidates: push this branch and open
  its own PR against main; write up Phase 2 findings for the portfolio
  README; deploy the dashboard to Streamlit Community Cloud (original
  PLAN.md target, "Note: data size may require pre-computing dashboard
  data at model run time" — already satisfied, since `app.py` reads the
  pre-computed `cost_dashboard_data.json` rather than loading raw data).

---

## Session Resumption Checklist

When starting a new session:
1. Open Anaconda Prompt → `conda activate ieee-fraud`
2. Open VS Code in `C:\Projects\IEEE_fraud`
3. Select Python interpreter: `Python (ieee-fraud)`
4. Read this file and `utils.py` to re-establish context
5. Check `git status`, `git branch`, and `git log --oneline` — work
   continues on `feature/phase2-cost-sensitivity`, now rebased onto main
   (PR #3 merged 2026-07-07, so main includes Tier 2/segmentation/ensemble
   too). 3 commits ahead of main, **still local-only, not yet pushed** —
   confirm with the user before pushing / opening a PR.
6. **Phase 2's Cost-Sensitive Decision Framework is done**: cost function
   decided (session 9 — payment-processor lens, see "Cost function
   parameter values" section), threshold-sweep + sensitivity + one-time
   test confirmation done (`phase2_cost_analysis/cost_threshold_analysis.py`),
   Streamlit dashboard built and verified (`app.py`, plus the
   `.claude/skills/run-ieee-fraud/` project skill for re-verifying it).
   **Next action not yet decided** — see "Current state" above for
   candidates (push + PR, README write-up, Streamlit Cloud deploy).
7. TransactionDT timezone: single reference point confirmed — id_14-adjusted
   `local_hour` is valid. Used in Phase 2 Tier 1's `local_hour` feature.
8. `pandas.Series.corr()` crashes this environment outright — see Environment
   section. Use `DataFrame.corr()` instead.

---

## Key Decisions Log

| Decision | Choice | Reason |
|---|---|---|
| Train/val/test | 60/20/20 chronological | Simulates deployment; prevents temporal leakage |
| Missing values | Sentinel -999 / "missing" | Structural missingness; trees handle natively |
| SMOTE | Class weights first | 3.5% rate less extreme; test before adding complexity |
| Feature engineering | Deferred past baseline | Before/after story for portfolio |
| RF excluded | Yes | Unfair comparison without native categoricals |
| LR excluded | Yes | Feature structure works against it; document rather than include |
| Primary metric | PR-AUC | More informative than ROC-AUC for imbalanced classes |
| Phase 2 cost function | Deferred | Carry prior structure; confirm values at Phase 2 start |
| addr2 treatment | Binary `is_us` | Preserves non-US signal; near-zero variance otherwise |
| TransactionDT timezone | Single reference point confirmed | id_14-adjusted local_hour is valid for Phase 2 |
| High-card string encoding | LightGBM native categorical | Avoids target-encoding leakage; OHE threshold = 10 unique values |
| Tier 1 browser/OS columns | Keep raw id_30/id_31 alongside parsed browser_name/os_name | Confirmed with user: avoids discarding version-specific signal; trees tolerate redundant features |
| Tier 1 outputs | Written to phase2_feature_engineering/ + models/*_phase2_* | Avoids overwriting Phase 1 baseline artifacts, keeps before/after comparable |
| has_true_local_hour flag | Keep | Controlled ablation: +0.018 PR-AUC for LightGBM, exact no-op for XGBoost — real gain, no downside |
| Feature-count comparisons | Use ablation mode (subsampling=1.0) to validate any single feature, not the production config | Fixed seed + column subsampling reshuffles the whole random draw when column count changes; production-config deltas below ~0.02 PR-AUC aren't trustworthy on their own |
| Tier 2 grouping key | `card1+card2+card3+card5+D1-adjusted start_day` (not the card-fragment combo alone) | Raw combo conflates ~14 real accounts per entity on average (84.1% of multi-txn combos split into >1 D1-adjusted start-day); validated via addr1-diversity cohesion check before trusting it |
| Tier 2 addr1 handling | Own feature (`addr1_changed_from_prev` + `addr1_change_x_inverse_time`), not folded into the entity key | Folding it into the key would make an address change look like a new entity (history resets) instead of a flagged event on a continuous account |
| Tier 2 XGBoost result | Documented as an open question, not a confirmed win | Best aggregate PR-AUC of the investigation (0.5797), but all 6 features rank bottom-half in importance — can't rule out the improvement being incidental to this run |
| Error analysis | Data-driven pass added alongside hypothesis-driven feature engineering | Found the identity-presence blind spot directly, rather than requiring it to be guessed at in advance |
| Segmented model | Build `has_identity=1`/`has_identity=0` as separate models rather than one global model | Segment fraud rates diverge 3.2x (train fold); error analysis showed identity presence is the dominant driver of catchability |
| Segmented model recency handling | Downweight/drop early train quarters for the `has_identity=1` segment only | Within-train quarters show a regime shift (Q2→Q3) in this segment specifically that val/test continue; `has_identity=0` showed no such drift and needs no adjustment. **Superseded (session 7 continued):** a clean with/without ablation (run as a side effect of `smote_ablation.py`) showed recency-weighting is a wash-to-slightly-negative, not the improvement its mechanism argument assumed — dropped from the production config |
| Segmented model recency method | Exponential decay (half-life 30 days), not a hard Q1/Q2 cutoff | User decision (session 7): avoids discarding Q1/Q2 signal entirely while still emphasizing the Q3/Q4-like regime val/test continue. **Superseded** — see recency handling row above |
| Segmented model evaluation | Fair per-segment comparison (global vs. segment model, each scored only on its own segment's rows), not a pooled combined metric | Pooling two independently class-weighted segment models' raw probabilities into one ranking metric compares scores on different scales — confirmed via calibration check: `has_identity=0`'s mean predicted probability was 4.5–5.5x its true rate vs. ~1–1.7x for `has_identity=1` |
| Segmented model final architecture | Hybrid: dedicated model for `has_identity=1`, global model for `has_identity=0` (session 7 version) | Fair comparison shows `has_identity=1` segmentation wins outright on both models/metrics; `has_identity=0` tuning (both more and less model complexity) never beat the global model there. **Revised session 8** — see next row |
| Inference routing (final, session 8) | Algorithm-specific: XGBoost keeps hybrid routing (seg_id1 for has_identity=1); LightGBM uses the global model for all rows, no routing | Re-running the fair comparison under the current SMOTE 1:10 + tuned config (never done before wiring up real inference) showed LightGBM's global model now beats its own has_identity=1 segment model (PR-AUC 0.7868 vs 0.7789) — reversed from session 7's finding, which predates SMOTE/tuning. XGBoost's segment model still wins, margin shrunk (0.0085→0.0050). User decision: route each algorithm by what the current evidence shows, not a single fixed rule |
| Imbalance handling (production) | SMOTE 1:10 (via `SMOTENC`), replacing `is_unbalance`/`scale_pos_weight` class-weighting and recency-weighting for both production models | Ablation (session 7 continued) showed SMOTE 1:10 beats class-weighting alone on PR-AUC by a wide margin (LightGBM +0.056, XGBoost +0.027) and beats the has_identity=1 segment's recency-weighted config outright on every metric; more aggressive ratios (1:5, 1:3) underperform 1:10 |
| SMOTE implementation | `SMOTENC`, not plain `SMOTE` | Plain SMOTE linearly interpolates every column, corrupting one-hot dummies, native categorical columns, and binary engineered flags into meaningless fractional values; `SMOTENC` majority-votes those columns instead of interpolating them |
| Hyperparameter tuning scope | Modest randomized search (6 trials/model/algorithm), not an exhaustive grid | Tuning is "if time allows" per project priority, not the main event; adopted 3 of 4 winning configs (Global LightGBM/XGBoost, has_identity=1 LightGBM), kept has_identity=1 XGBoost at its SMOTE-1:10 default since its best trial's margin was too thin relative to the 6-trial selection bias to trust |
| Git branch structure (session 6) | New branch `feature/phase2-tier2-and-segmentation`, separate from Tier 1's `feature/phase2-feature-engineering` (PR #2) | Confirmed with user: keeps PR #2 scoped to Tier 1 and independently mergeable, rather than growing into an unrelated, harder-to-review PR |
| `combo_amt_zscore` zero-variance fix (session 8) | Floor prior std at 1% of prior mean (min 1 cent) instead of dividing by a raw 0 and getting `NaN` | A step-back question about the two algorithms' relative strengths led to `model_divergence_analysis.py`, which found LightGBM and XGBoost scoring identical-repeat-amount legitimate transactions (recurring-payment pattern) 0.01-0.07 vs. 0.85-0.92 percentile risk — traced to this feature's `NaN` output for any entity with zero historical variance, handled very differently by each algorithm's native missing-value routing |
| Production model ensemble (session 8) | Weighted blend `0.70 * xgb_prob + 0.30 * lgb_prob` (each already per its own routing) is now `inference.py`'s default (`algorithm="ensemble"`) | `ensemble_test.py`'s 21-point grid search (w=0.0-1.0, step 0.05) found this beats standalone XGBoost by +0.0086 PR-AUC, holds in both `has_identity` segments separately, and sits in a broad plateau (w=0.55-0.85 all within 0.004) rather than a fragile single-point spike. Motivated by `model_divergence_analysis.py` showing the two algorithms agree on only 67-68% of fraud cases (Spearman ρ=0.745) |
| Cost function lens (session 9) | Payment-processor P&L, not issuer-investigation or merchant-lost-margin | Job-application target is specifically a payment processor (merchant/issuer middleman); the source data (Vesta) is itself this kind of company. Changes both FP meaning (processor's forfeited fee revenue + retention risk, not analyst time or merchant margin) and strengthens FN=Amount's justification (chargeback-guarantee liability model common for this vendor category, vs. prior project's "worst case, no recovery" hedge) |
| Cost function parameter values (session 9) | FP = $3 + 3%×Amount; FN = Amount | Amount-distribution check (`amt_stats.py`) showed this dataset's scale is broadly comparable to the prior project's, so no distribution-driven reason to change magnitudes; values instead chosen to fit the processor-P&L reframing (see lens decision above) rather than reused blindly from the prior project's $10+1% issuer framing |
| Aggressive cost-optimal threshold (session 9) | Documented as a named limitation, not masked by an ad hoc capacity constraint | The cost-minimizing threshold (0.03) flags ~15.6% of validation transactions — mathematically correct given the ~19:1 FN:FP cost ratio, but far more aggressive than real-world decline rates. A capacity constraint (e.g. cap flag rate at 5%) would need its own justification (review-team capacity? industry benchmark?) not yet available, so the linear cost function's blind spot (no term for the *aggregate* false-decline rate, only its sum) is named explicitly in the write-up instead. Revisit if a defensible capacity justification emerges |
