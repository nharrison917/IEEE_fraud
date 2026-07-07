# -*- coding: utf-8 -*-
"""
Driver for smoke-testing app.py (the Streamlit cost-sensitivity dashboard).

Assumes the server is already running (see SKILL.md "Run (agent path)").
Navigates every major section, screenshots each, drags the Classification
Threshold slider to prove live recompute works, and fails (exit 1) if the
browser console logs any error or an uncaught page exception fires.

Usage:
    conda run -n ieee-fraud python driver.py [port]   # port defaults to 8765

Screenshots land in .claude/skills/run-ieee-fraud/screenshots/, overwritten
each run.
"""

import sys
import os
from playwright.sync_api import sync_playwright

PORT = sys.argv[1] if len(sys.argv) > 1 else "8765"
URL = f"http://localhost:{PORT}"
SHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")

console_errors = []

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1600, "height": 1200})
    page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
    page.on("pageerror", lambda exc: console_errors.append(f"pageerror: {exc}"))

    print(f"Navigating to {URL} ...")
    page.goto(URL, wait_until="domcontentloaded", timeout=30000)
    page.wait_for_selector("text=IEEE-CIS Fraud Detection", timeout=30000)
    page.wait_for_timeout(2000)  # let first Plotly charts finish rendering
    page.screenshot(path=f"{SHOT_DIR}/01_top.png")
    print("Saved 01_top.png (header, Bottom Line, top of Threshold Explorer)")

    # NOTE: don't use get_by_text() for section names here -- the sidebar's
    # own caption text repeats "Sensitivity Analysis" / "Operational Reality"
    # verbatim, so get_by_text() matches the sidebar and scroll_into_view_if_needed()
    # becomes a no-op (already "in view"). get_by_role("heading", ...) is
    # unambiguous because Streamlit renders st.header()/st.subheader() as
    # actual <h2>/<h3> tags, but sidebar captions are plain text, not headings.
    for i, name in enumerate(["Algorithm Comparison", "Sensitivity Analysis", "Operational Reality"], start=2):
        heading = page.get_by_role("heading", name=name).first
        heading.scroll_into_view_if_needed(timeout=15000)
        page.wait_for_timeout(1500)
        fname = f"{i:02d}_{name.lower().replace(' ', '_')}.png"
        page.screenshot(path=f"{SHOT_DIR}/{fname}")
        print(f"Saved {fname}")

    # Limitations lives inside an st.expander -- its label renders as a
    # <summary>/button, not a heading role, so get_by_role("heading", ...)
    # will time out on it. It's expanded=True in app.py, so scrolling to the
    # end of the page is enough to capture its content.
    page.keyboard.press("End")
    page.wait_for_timeout(1000)
    page.screenshot(path=f"{SHOT_DIR}/05_limitations.png")
    print("Saved 05_limitations.png (bottom of page, Limitations expander)")

    # Drag the Classification Threshold slider (first stSlider in the sidebar)
    # to prove the cost-curve recompute is actually live, not a static image.
    print("Dragging Classification Threshold slider...")
    page.mouse.wheel(0, -100000)  # scroll all the way back up first
    page.wait_for_timeout(500)
    slider_handle = page.locator('div[data-testid="stSlider"]').first.locator('div[role="slider"]').first
    box = slider_handle.bounding_box()
    if box is None:
        console_errors.append("driver: could not find slider handle bounding box")
    else:
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x + 80, y, steps=5)
        page.mouse.up()
        page.wait_for_timeout(2500)
        page.screenshot(path=f"{SHOT_DIR}/06_after_slider_drag.png")
        print("Saved 06_after_slider_drag.png -- check the 'Selected: <t>' vline "
              "appears distinct from 'Cost-optimal at current sliders' to confirm live recompute")

    browser.close()

print("\n--- Console errors / page exceptions ---")
if console_errors:
    for e in console_errors:
        print("ERROR:", e)
    sys.exit(1)
print("None. Smoke test passed.")
