---
name: run-ieee-fraud
description: Build, run, and smoke-test the IEEE_fraud Streamlit cost-sensitivity dashboard (app.py). Use when asked to start the dashboard, launch app.py, screenshot the dashboard, verify the Streamlit app still works, or drive its sliders.
---

This is a single Streamlit app (`app.py`, project root) reading
`models/cost_dashboard_data.json`. It's driven via a Playwright driver at
`.claude/skills/run-ieee-fraud/driver.py` (headless Chromium) -- there is no
`chromium-cli` in this environment, so a small committed Python driver
stands in for it. All paths below are relative to the repo root
(`C:\Projects\IEEE_fraud`).

Environment: Windows, Git Bash as the shell, conda env `ieee-fraud`.
`conda activate` does not work reliably in PowerShell on this machine
(known issue), and plain `conda` is not on Git Bash's PATH at all --
**always use the full path**, `/c/Users/nharr/anaconda3/Scripts/conda run
-n ieee-fraud <cmd>`, as this skill does throughout. (Bare `conda run ...`
fails with `conda: command not found` -- hit and fixed while verifying
this skill; see Gotchas.)

## Prerequisites

Playwright + headless Chromium, installed once into the `ieee-fraud` env
(not tracked in `env/environment.yml` -- it's a test-only tool, not a project
dependency):

```bash
/c/Users/nharr/anaconda3/Scripts/conda run -n ieee-fraud python -m pip install playwright
/c/Users/nharr/anaconda3/Scripts/conda run -n ieee-fraud python -m playwright install chromium
```

`streamlit` itself is already in `env/environment.yml` (verified installed:
1.58.0).

## Setup / Build

None -- no compile step. `models/cost_dashboard_data.json` must already
exist (produced by `phase2_cost_analysis/cost_threshold_analysis.py`); if
missing, regenerate it first:

```bash
/c/Users/nharr/anaconda3/Scripts/conda run --no-capture-output -n ieee-fraud python phase2_cost_analysis/cost_threshold_analysis.py
```

(That script reloads and rebuilds features across the full 590k-row
dataset -- takes a few minutes. Not needed just to launch the dashboard if
the JSON is already present.)

## Run (agent path)

1. Launch the server headless on a fixed port, in the background, and wait
   for it to actually serve (don't fixed-`sleep` -- poll the health
   endpoint):

```bash
/c/Users/nharr/anaconda3/Scripts/conda run --no-capture-output -n ieee-fraud streamlit run app.py --server.headless true --server.port 8765 > streamlit.log 2>&1 &
disown
timeout 30 bash -c 'until curl -sf http://localhost:8765/_stcore/health >/dev/null 2>&1; do sleep 1; done' && echo "SERVER UP"
```

2. Run the driver against it:

```bash
/c/Users/nharr/anaconda3/Scripts/conda run -n ieee-fraud python .claude/skills/run-ieee-fraud/driver.py 8765
```

It navigates the app, screenshots every major section, drags the
Classification Threshold slider (proving the cost curves recompute live,
not static images), and exits 1 if the browser console logged any error or
an uncaught page exception fired. Screenshots land in
`.claude/skills/run-ieee-fraud/screenshots/` (`01_top.png` through
`06_after_slider_drag.png`, overwritten each run) -- **actually look at
them**, don't just check the exit code.

3. Stop the server cleanly. `curl` health-checking a plain HTTP endpoint
   doesn't need cleanup, but the streamlit process does -- **`kill $!` does
   NOT work here**: `conda run` wraps the real `streamlit.exe` in a child
   process, so the PID bash captured is the wrapper, not the server. Find
   the real PID by port instead:

```bash
PID=$(netstat -ano | grep ":8765" | grep LISTENING | awk '{print $5}' | head -1)
taskkill //PID "$PID" //F
```

## Run (human path)

```bash
/c/Users/nharr/anaconda3/Scripts/conda run -n ieee-fraud streamlit run app.py
```

Opens the default browser automatically at `http://localhost:8501`.
`Ctrl+C` in the terminal to stop. No special flags needed for normal
interactive use -- `--server.headless` / `--server.port 8765` above are
only for the automated driver.

## Test

No automated test suite for the dashboard itself -- `driver.py`'s
zero-console-errors + screenshot check *is* the test.

---

## Gotchas

- **Streamlit Community Cloud prioritizes `environment.yml` over `requirements.txt`**
  if both exist at repo root (`uv.lock` > `Pipfile` > `environment.yml` >
  `requirements.txt` > `pyproject.toml`). This project's local conda dev environment
  lives at `env/environment.yml` specifically to stay out of Cloud's root-level scan --
  `requirements.txt` + `runtime.txt` at the actual repo root are what Cloud deploys use.
  Hit this for real: a first deploy attempt sat at conda's "Solving environment:" for the
  full dev stack (xgboost, lightgbm, imbalanced-learn, scikit-learn) for 10+ minutes
  before the cause was traced to environment.yml being at root and outranking
  requirements.txt. If `env/environment.yml` ever moves back to root, re-verify this.
- **`get_by_text()` for section names is ambiguous -- use
  `get_by_role("heading", ...)` instead.** The sidebar's own caption
  ("Settings that affect the Threshold Explorer, Sensitivity Analysis, and
  Operational Reality sections below") repeats the section names verbatim.
  `page.get_by_text("Sensitivity Analysis")` matches that sidebar caption
  (already in view) instead of the actual `st.header("Sensitivity
  Analysis")` further down the page, so `scroll_into_view_if_needed()`
  becomes a silent no-op -- the screenshot looks identical to the previous
  one and nothing errors. Streamlit renders `st.header()`/`st.subheader()`
  as real `<h2>`/`<h3>` tags, so `get_by_role("heading", name=...)` is
  unambiguous. (Confirmed by hitting this exact bug once before fixing
  `driver.py`.)
- **`st.expander(..., expanded=True)` content (Limitations section) is not
  a `heading` role** -- its label renders as a `<summary>`/button, so
  `get_by_role("heading", name="Limitations")` times out. Since it's
  expanded by default, scrolling to the bottom of the page
  (`page.keyboard.press("End")`) is enough to capture it; no special
  targeting needed.
- **`kill $!` after backgrounding `conda run ... streamlit run ...`
  doesn't stop the server.** `conda run` forks a wrapper process; `$!` is
  that wrapper's PID, not the actual `streamlit.exe`/Python process bound
  to the port. `kill` on the wrapper exits silently successfully but the
  server keeps answering `curl` afterward. Find the PID actually
  `LISTENING` on the port via `netstat -ano` and `taskkill //PID <pid> //F`
  instead (see step 3 above).
- **A bare `curl http://localhost:8765/` proves the static shell loaded,
  not that the app script ran without error.** Streamlit's actual script
  execution happens over a websocket once a browser session connects, not
  on the initial HTML GET -- a page-level Python exception in `app.py`
  would only surface at that point. `driver.py`'s `console`/`pageerror`
  listeners catch it; a plain `curl` smoke check would not.

## Troubleshooting

- **`/usr/bin/bash: line N: conda: command not found`, server never comes
  up, health-check loop times out (`timeout 30 ...` exits 124)**: bare
  `conda` isn't on Git Bash's PATH on this machine. Use the full path,
  `/c/Users/nharr/anaconda3/Scripts/conda run -n ieee-fraud <cmd>`, exactly
  as written in this file -- don't shorten it. Hit while verifying this
  skill: the first draft used bare `conda run` and silently failed to
  launch, backgrounded, with no visible error until `cat streamlit.log`.
- **`ModuleNotFoundError: No module named 'playwright'`**: not installed
  in the `ieee-fraud` env yet -- run the Prerequisites step above.
- **`playwright._impl._errors.TimeoutError` on
  `get_by_role("heading", name=...)`**: either the server isn't actually up
  yet (check the health-check step ran and printed `SERVER UP` before
  launching the driver) or the section name doesn't match an `st.header()`
  string exactly -- check `app.py` for the literal heading text.
