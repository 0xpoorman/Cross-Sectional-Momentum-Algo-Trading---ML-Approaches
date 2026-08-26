# Cross-Sectional Momentum — ML Approaches

An evidence-first study of cross-sectional momentum using neural LambdaRank and
LightGBM LambdaMART. The repository preserves the initial Phase 1 reproduction,
its ranking and portfolio evidence, and the reporting brief used to tell the
larger research story without hiding weak or conflicting results.

[![Phase 1 codebase architecture](first%20phase_LMART_LRank/assets/phase1-codebase-dark.svg)](first%20phase_LMART_LRank/assets/phase1-codebase.html)

<p align="center">
  <strong><a href="first%20phase_LMART_LRank/assets/phase1-codebase-dark.svg">Open the full-size architecture map</a></strong>
  ·
  <a href="first%20phase_LMART_LRank/assets/phase1-codebase.html">Interactive Archify version</a>
</p>

## How the Phase 1 code fits together

`recreate_first_trials.py` is the entry point. It loads and engineers the market
frame through the feature helpers, applies a purged chronological split and
cross-sectional normalization, then dispatches into two genuinely different
training paths:

- LambdaMART builds grouped relevance data and trains LightGBM directly.
- LambdaRank delegates the PyTorch network and two-tail ranking objective to
  `helpers/model.py`.

Both branches return to the same ranking metrics, fixed three-bar backtest,
artifact writer, isolated MLflow tracker, and portable comparison report. Click
the diagram above to inspect those relationships at full resolution.

## Start here

| Resource | Purpose |
|---|---|
| [Phase 1 package](first%20phase_LMART_LRank/) | Reproduce the original LambdaMART/LambdaRank comparison |
| [Phase 1 evidence report](report_phases/phase_1/report.html) | Read the standalone technical evidence report |
| [Director's Script](DIRECTORS_SCRIPT.md) | Review the production brief for the full research narrative |
| [Hermes report bundle](hermes_report/) | Inspect the report builder, skills, sources, and generated output |

## Phase 1 contract

- 11 SPDR sector ETFs with SPY kept outside the ranked universe
- Ten OHLCV-derived features and cross-sectional z-scoring per signal date
- Chronological 70/15/15 split with a three-bar purge
- Next-open to open-three-bars-later labels
- Top-two long and bottom-two short, non-overlapping three-bar cohorts
- Zero transaction costs and no stop or take-profit logic in the frozen baseline

See the [Phase 1 README](first%20phase_LMART_LRank/README.md) for installation,
execution, MLflow, test, and interpretation details.
