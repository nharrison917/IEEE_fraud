# IEEE Fraud Detection — Project Instructions for Claude

## Project context
See PLAN.md for full methodology, decisions log, and session resumption checklist.
Read PLAN.md and utils.py at the start of every session before writing any code.

## Output standards for this project

All analysis scripts should produce outputs in two forms:

1. **Interactive HTML** (Plotly) — for the portfolio and visual inspection by the user.
2. **Machine-readable terminal output** — printed to stdout so Claude can read and
   interpret results directly using the Bash tool. This enables real back-and-forth
   analysis without relying on Claude rendering HTML.

Concretely:
- Feature importance: print the top-N table to stdout (name + value), not just save HTML.
- Model metrics: print ROC-AUC and PR-AUC for both train and val folds.
- Any comparison table (e.g. model A vs B) should be printed in plain text.
- When saving plots, also save the underlying data as JSON or CSV in the same directory
  so it can be read back for discussion.

The goal is that Claude can act as a genuine thinking partner on results, not just a
code writer — and that requires Claude to be able to see the numbers.

## Git workflow
- Never commit or push autonomously — confirm before every commit.
- No Co-Authored-By trailers.
- Feature branches merged via PR on GitHub; no local merges to main.
- Commit by concern, not by session.

## Data
- data/ is not tracked in git.
- Use only train_transaction.csv and train_identity.csv (test files have no labels).

## Environment
- Conda env: ieee-fraud
- Activate in Anaconda Prompt (not PowerShell — conda PS integration is broken).
