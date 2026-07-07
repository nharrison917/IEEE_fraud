# Cost-Sensitive Fraud Detection -- Results Summary

## Assumptions -- Payment-Processor Lens

> Job-application target for this project is a payment processor (merchant/issuer middleman),
> not a bank or a merchant. Costs below reflect the processor's own P&L, not the merchant's
> lost margin or an issuer's investigation cost. All figures are illustrative -- TransactionAmt's
> currency/units are not confirmed by Kaggle.

| Item | Value | Rationale |
|---|---|---|
| Base FP cost | $3.00 | Automated dispute/support handling for a declined checkout |
| Friction penalty | 3% of Amount | Forfeited processing-fee revenue + retention-risk allowance |
| Full FP cost | $3.00 + (3% x Amount) | -- |
| FN cost | Full Amount | Chargeback-guarantee liability model (general industry knowledge for this vendor category, not verified from this project's data) |

---

## Algorithm Comparison -- Test Set

| Algorithm | Threshold Type | Threshold | Total Cost | FP Cost | FN Cost | Recall | F1 |
|---|---|---|---|---|---|---|---|
| LightGBM | Cost-Optimal (val) | 0.03 | $305,142 | $127,629 | $177,513 | 0.7160 | 0.2605 |
| LightGBM | 0.5 Default | 0.50 | $492,510 | $2,739 | $489,771 | 0.3054 | 0.4343 |
| XGBoost | Cost-Optimal (val) | 0.03 | $279,752 | $121,848 | $157,904 | 0.7537 | 0.2780 |
| XGBoost | 0.5 Default | 0.50 | $484,327 | $2,330 | $481,997 | 0.3263 | 0.4588 |
| Ensemble (0.70 XGB / 0.30 LGB) **** | Cost-Optimal (val) | 0.03 | $280,056 | $127,102 | $152,954 | 0.7574 | 0.2729 |
| Ensemble (0.70 XGB / 0.30 LGB) | 0.5 Default | 0.50 | $487,345 | $2,067 | $485,278 | 0.3209 | 0.4562 |

---

## Operational Interpretation (scaled to 1,000,000 transactions)

At a cost-optimal threshold of **0.03**, Ensemble (0.70 XGB / 0.30 LGB) flags approximately
**156,611** transactions as fraud, of which approximately **26,061** are genuine
fraud caught and **130,550** are false alarms (declined legitimate checkouts). Estimated
fee-loss/friction cost of false alarms is **$1,076,150**, offset against approximately
**$1,295,034** in fraud prevented, for a net saving of approximately **$2,793,024**
compared to no fraud detection.

---

## Sensitivity Analysis

### (a) FP base cost (friction rate fixed at 3%)

| FP Base Cost | Optimal Threshold | Total Cost (validation) |
|---|---|---|
| $1 | 0.02 | $215,445.12 |
| $3 | 0.03 | $255,922.03 |
| $5 | 0.04 | $278,336.49 |
| $10 | 0.06 | $314,678.19 |

### (b) Friction rate (base FP cost fixed at $3.00)

| Friction Rate | Optimal Threshold | Total Cost (validation) |
|---|---|---|
| 1% | 0.02 | $188,052.75 |
| 3% | 0.03 | $255,922.03 |
| 5% | 0.05 | $291,104.77 |
| 10% | 0.07 | $349,713.30 |

---

## Limitations

1. **Cost lens is a modeling choice**: payment-processor framing chosen to match the job-application
   target; a merchant or issuer would see a materially different cost structure.
2. **Chargeback-guarantee assumption**: general industry knowledge about this vendor category, not
   verified from this project's own data.
3. **Currency/units unconfirmed**: TransactionAmt's scale is not officially documented by Kaggle.
4. **Real-time scoring gap**: Tier 2 entity-velocity features require full prior history, computed
   offline here -- see `phase2_feature_engineering/inference.py` docstring.
5. **Retention-risk cost** is folded into the 3% friction rate as an allowance, not derived from an
   independent churn model.
6. **The cost function has no term for the AGGREGATE false-decline rate, only its sum**: the pure
   cost-minimizing threshold (0.03) flags approximately **15.6%** of all validation transactions --
   each individual false decline is priced correctly, but a processor declining 1 in 6 legitimate
   checkouts would in practice trigger merchant churn through a threshold/nonlinear effect this
   linear, per-transaction cost function cannot see. Documented here deliberately (rather than
   papered over with an unjustified capacity constraint) so the gap between "cost-optimal in
   theory" and "operationally deployable" stays visible.
