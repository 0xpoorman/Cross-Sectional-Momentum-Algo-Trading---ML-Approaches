---
name: ml-evidence-audit
description: Reconcile Learning2Rank artifacts into sourced evidence
version: 1.0.0
metadata:
  hermes:
    tags: [mlops, ranking, audit, mlflow, optuna]
    category: MLOps
---

# ML evidence audit

## When to use
Use before writing any post-study claim or chart.

## Procedure
1. Read `DIRECTORS_SCRIPT.md` and enumerate source paths and metric definitions.
2. Discover completed versus partial Optuna studies from manifests and trial states.
3. Read packaged JSON/CSV artifacts before MLflow; use MLflow/SQLite to reconcile chronology and parameters.
4. Join runs using dataset fingerprint, feature set, horizon, tail mode, model family, seed, and strategy parameters.
5. Produce bounded `report_data.json`, `report_sources.json`, `chart_manifest.json`, and `report_build_log.md`.

## Guardrails
- Do not rerun optimization or invent missing test metrics.
- Do not call an Optuna Sharpe objective return.
- Keep validation, untouched test, and post-selection diagnostics separate.
