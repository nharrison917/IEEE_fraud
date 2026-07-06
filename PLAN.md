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

### 4. No SMOTE initially
At 3.5% fraud rate, test `class_weight='balanced'` (LightGBM parameter) first.
Compare against modest SMOTE (1:10) on validation if needed. SMOTE applied to
training fold only, after all preprocessing, never before split.

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

---

## Git Workflow

- Branch `main` tracks production-ready work
- Feature work on branches, merged via PR on GitHub (no local merges)
- Never commit autonomously — confirm before every commit
- No Co-Authored-By trailers
- Commit by concern, not by session

**Current state (end of session 5):**
- main: Phase 1 complete and merged (PR #1)
- Branch `feature/phase2-feature-engineering`: 3 commits already made (stale
  metrics fix, Tier 1 feature engineering, `.gitattributes` housekeeping).
  Since then: added `has_true_local_hour`, discovered and fixed the
  feature-count/subsampling confound, built `ablation_check.py`, validated
  the flag via controlled ablation, and scoped Tier 2 (see sections above).
  Not yet committed.
- **Next action:** commit the `has_true_local_hour` + ablation-methodology
  work as one unit, then push and open the PR. After merge: start Tier 2
  card-level aggregates (grouping keys and feature types scoped above),
  starting with the C/D-column correlation triage before building anything.
  Cost-Sensitive Decision Framework is explicitly on hold until feature
  engineering is judged to have run its course.

---

## Session Resumption Checklist

When starting a new session:
1. Open Anaconda Prompt → `conda activate ieee-fraud`
2. Open VS Code in `C:\Projects\IEEE_fraud`
3. Select Python interpreter: `Python (ieee-fraud)`
4. Read this file and `utils.py` to re-establish context
5. Check `git status` and `git log --oneline` to see current state
6. **Next action:** confirm commit split and open the Phase 2 Tier 1 PR (see
   "Current state" above). After merge, decide Tier 2 vs Cost-Sensitive
   Decision Framework as the next phase.
7. TransactionDT timezone: single reference point confirmed — id_14-adjusted
   `local_hour` is valid. Used in Phase 2 Tier 1's `local_hour` feature.

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
