---
name: portable-html-report
description: Package a self-contained Learning2Rank storytelling report
version: 1.0.0
metadata:
  hermes:
    tags: [html, accessibility, charts, offline]
    category: AI
---

# Portable HTML report

## When to use
Use last, after evidence, narrative, and diagrams are ready.

## Procedure
1. Build the single-column editorial reading path from `DIRECTORS_SCRIPT.md`.
2. Embed bounded report data and use repository-relative assets only.
3. Use inline SVG or a pinned/vendored Canvas-capable chart implementation; do not use Perspective and do not fetch data at runtime.
4. Add semantic tables for every interactive chart and visible source/caveat context near the claim.
5. Support responsive, keyboard, print, light/dark, and offline rendering.
6. Run repository HTML/report QA and write the build log.

## Output
Write `report.html`, `report_data.json`, `report_sources.json`, `chart_manifest.json`, `report_build_log.md`, and the two model-flow SVGs required by the Director's Script.
