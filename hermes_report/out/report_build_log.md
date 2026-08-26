# Report build log — l2r-post-study-report

Build date: 2026 (local build from repository artifacts)
Builder: Hermes agent, following `hermes_report/skills/*` and DIRECTORS_SCRIPT.md

## What was built

- `out/report.html` — self-contained editorial report; all charts inline SVG with semantic data-table fallbacks in `<details>` drawers. No external fetches, no CDN, no Perspective, no runtime data loading.
- `out/report_data.json` — bounded normalized dataset (equity curves downsampled ~weekly).
- `out/report_sources.json` — 13 source records covering every chart and quantitative claim.
- `out/chart_manifest.json` — 13 chart entries with section, question, source ids, comparison basis, supported claim.
- `out/assets/model-flow-lambdamart.svg` and `out/assets/model-flow-lambdarank.svg` — editorial model-flow diagrams with `<title>`/`<desc>` text alternatives.
- `build_report.py` — deterministic rebuild script.

## Discovery results (reconciled, not copied)

- Three COMPLETED Optuna studies: LRank_sectors_both_broad_smooth_1 (100 trials: 67 COMPLETE / 33 PRUNED), LMart_sectors_both_architecture_1 (200: 49/151), TreeRank_sectors_both_architecture_1 (200: 87/113). Total completed-study evidence = 500 trials. Four other LRank study directories exist but were not declared complete in the brief; they are excluded from the headline and listed under exclusions.
- MLflow `artifacts/mlflow.db`: 23 FINISHED runs in Learning2Rank_sectors. `LRank_sectors_1` appears three times; deduplicated by run UUID, noted here per rules.
- Packaged runs joined by run name + parameter fingerprint + data fingerprint (8ca13be0…d6c09) across packages and MLflow params.
- Stable selections reconciled against the brief's anchors: neural trial 99 (obj 1.004) vs best single trial 79 (1.052); legacy LightGBM stable 194 (0.877) vs best 132 (1.070); mixed-tree trial 159 is both stable and best single (1.040). All three favor long_only and no MACD; mixed-tree winner uses LightGBM. No conflicts found.
- Test window for all compared equity curves: entries 2025-01-27 → 2025-12-11, equity through 2025-12-30. Early baselines use their own longer window (2024-11-11 start) and are only compared within that panel, never mixed with test-window runs.

## Warnings and gaps

1. **Strategy 2 (ATR trailing exits): no completed package found.** Repository-wide search for ATR/trailing/strategy-2 artifacts returned nothing matching the selected configurations. Section 10 presents an explicit gap callout rather than numbers. UNRESOLVED.
2. **No exact untouched-test package exists for the selected Optuna configurations** (neural trial 99 params, tree trial 159 params). Per the brief's instruction, the report says so implicitly by presenting packaged long-only/test-window runs as evidence of behavior, not as confirmation of the selected trials. The "next steps" section makes exact reruns action #2.
3. **Early baselines lack realized-Sharpe logging** (LMart_sectors_1, LRank_sectors_1); those panels compare total return only, flagged in the figure caption.
4. **XGBoost complete-trial count is small** (7 of 87 in the mixed study); backend comparison reported as observed means with counts, no inferential language.
5. **Neural dropout**: selected trial records dropout 0.05 but the builder instantiates dropout only between hidden layers; with one hidden layer it is inactive. Shown as crossed-out/requested-but-inactive chip in the SVG and param table, per acceptance tests.
6. **LightGBM truncation (10)** appears only on tree diagrams/tables, never as a neural or XGBoost knob.

## Exclusions

- Pruned trials (33/151/113) are excluded from clean trial exports; sample counts shown on charts.
- Partial/incomplete LRank studies (`LRank_sectors_long_only_*`, `LRank_sectors_architecture_1`) omitted from the 500-trial headline.
- Packaged `report.html` files used only as visual reference, never numeric sources.
- SPY derived only from its own file, never inferred from the ranked universe.
- Absolute local paths appear only in this log and source manifests; published HTML uses run names only.

## QA checklist (acceptance tests)

- [x] Every visible number traces to report_sources.json (source id line under each figure).
- [x] Completed vs partial studies separated (partial omitted + logged).
- [x] Stable vs best-single trials distinguished (ring/star encodings, dumbbell chart, tables).
- [x] Walk-forward objective labeled Sharpe throughout; never called return.
- [x] No test metric influenced Optuna selection (test_rows_used=0 in manifests; stated in sources S4).
- [x] Model-flow figures match effective implementation incl. inactive dropout and capped leaves.
- [x] Truncation shown only where it belongs.
- [x] Strategy 2 labeled post-selection gap.
- [x] Compared equity curves share date range and assumptions within each figure.
- [x] SPY visible wherever strategy performance discussed; neutral dashed styling.
- [x] No raw JSON or machine-local absolute paths in published HTML.
- [x] Desktop/mobile responsive, keyboard focus-visible styles, light/dark via prefers-color-scheme, print stylesheet, fully offline/static-hostable (single file + two asset SVGs also embedded inline).
- [x] Charts fall back to semantic tables inside each figure's details drawer.


## Revision 2 — Director's Amendments (DIRECTOR.md)

All P0 corrections applied:

1. **Illustrative vs selected runs**: evidence-status banner added at top; LRank_sectors_14 / LMart_sectors_7 labeled "illustrative packaged runs" in every chart legend, caption, and manifest entry; no package is called "best/selected/winner" unless fingerprint-matched (none currently is).
2. **Stability units fixed**: dumbbell chart now plots raw fold-level mean Sharpe for stable trial and cluster on one common axis; penalized objectives listed separately (table + metric dictionary defining mean_sharpe, fold_std, stability_penalty=0.25, objective = mean_sharpe − 0.25 × fold_std). No penalized objective is labeled "Sharpe."
3. **Counts corrected**: headline now "500 attempted, 203 completed, 297 pruned" with per-study unique completed-trial counts by tail mode (62/5, 43/6, 82/5). Fold-observation counts labeled as repeated fold×seed measurements nested within trials.
4. **Timeline dates derived from artifacts**: MLflow run start times + manifest mtimes give 2026-08-22 (runs 1–6) and 2026-08-23 (runs 7–14, all three studies). No hard-coded 2025 dates remain.
5. **Equity evidence rebuilt**: two separate common-window panels (baseline window vs test window), each series rebased to 0% at the common start on a percentage scale, drawdown in the data table, SPY annotated as external and non-risk-matched, realized-only note retained. No cross-window overlay.
6. **Terminology**: "portfolio" removed except one explicit sentence stating SPY is not a risk-matched portfolio; "rank-to-position action", "top-two selection rule", "equal-notional long-only strategy" used throughout (HTML + both SVGs).
7. **Ridge baseline**: new section, accurately described (date-standardized target + global L2), labeled designed-but-not-yet-evidenced; no result inferred from config defaults.
8. **Dropout**: "Effective dropout: none" wording used; implementation detail moved to a methods note drawer.
9. **Qualified claims**: backend (imbalanced allocation), MACD (associational, not ablation), simplicity (selected not causal), regime (hypothesis pending defined regime variable), short-side (attributed to short selections under the fixed exit rule; duration/exit-type mechanical relation noted), adaptive-allocation caveat added to tail distributions.
10. **Arithmetic verification**: equity endpoints re-checked against backtest summaries at rebuild (all four runs match: −0.55%, −4.11%, +0.69%, +4.05%).

## Revision 3 — Cluster and stability chart correction

1. Replaced the individual-trial “Where the winners sit” scatter with a true
   cluster map sourced from all three `optuna_clusters.csv` files. One bubble is
   one cluster; area is unique completed trials; fill is tail mode; the ring is
   matched to `selected_cluster` in `optuna_selection.json`.
2. Added a visible selected-cluster table and retained all cluster definitions,
   trial counts, across-trial dispersion, and stability scores in the detail
   table.
3. Rebuilt the stability comparison so its horizontal axis contains raw mean
   walk-forward Sharpe only: selected-cluster mean, stable-trial mean, and
   best-single-trial mean.
4. Moved fold dispersion, the 0.25× dispersion penalty, and the resulting
   objective into a separate visible decomposition table. No objective appears
   on a Sharpe axis.
5. Collapsed mixed-tree trial 159 to one “stable = best” mark and one table row.
