# Director's Script: Learning to Rank in a Small Financial Universe

## Commission

Create one polished, self-contained HTML report for a senior audience. The report
must pull together the saved Optuna studies, MLflow runs, packaged model runs,
ranking diagnostics, strategy results, and reproducibility evidence in this
repository. It should tell an honest progression story: the starting point, the
few changes that materially altered the evidence, the best stable results, and
what did not generalize.

This document is a production brief. It is not permission to invent results,
rerun optimization, alter models, or optimize the report after seeing test data.

## Hermes execution contract

This brief is intended to be executed by Hermes using the bundled skills in
`hermes_report/skills/`. Install the skills and bundle before asking Hermes to
build the report. The bundle is deliberately limited to evidence extraction,
model-diagram reconciliation, and editorial HTML storytelling:

1. `ml-evidence-audit` — read-only discovery of Optuna, MLflow, packaged runs,
   manifests, and metric definitions; records provenance and conflicts.
2. `ranking-storytelling` — turns reconciled LambdaRank/LambdaMART evidence into
   an answer-first senior narrative without inventing results.
3. `model-diagram-reconciliation` — derives the LambdaRank and LambdaMART
   diagrams from the selected trial's effective parameters and implementation.
4. `portable-html-report` — packages bounded JSON, SVG diagrams, semantic
   fallbacks, and a self-contained responsive HTML report.

Use the bundle name `l2r-post-study-report`. The intended Hermes sequence is:

```text
mkdir -p .hermes/skills
cp -R hermes_report/skills/* .hermes/skills/
hermes skills trust .
hermes skills list
hermes bundles create l2r-post-study-report \
  --skill ml-evidence-audit \
  --skill ranking-storytelling \
  --skill model-diagram-reconciliation \
  --skill portable-html-report \
  -d "Build the Learning2Rank post-study report"
hermes chat --toolsets skills -q "/l2r-post-study-report Read DIRECTORS_SCRIPT.md and build the report."
```

Hermes loads repo-local skills from `.hermes/skills/` after the project is
trusted. If repo-local skills are unavailable in a particular Hermes profile,
copy each skill directory into `~/.hermes/skills/` instead.
Do not install or invoke Perspective for this report. Use the portable HTML
renderer already present in the repository and local, vendored or inline
Canvas/SVG-compatible charts only.

## Audience and editorial objective

Write for a senior technical, investment, or data-science leader who understands
models but does not want a source-code tour. The reader should leave knowing:

1. Why cross-sectional learning to rank was attempted on 11 daily sector ETFs.
2. How the experiment moved from two-tail baselines to controlled walk-forward
   searches across neural and boosted-tree rankers.
3. Which findings were stable and which were isolated best trials.
4. Why ranking quality, Sharpe, and absolute return did not move together.
5. What was learned from the short side, volatility sensitivity, model capacity,
   and exit-policy diagnostics.
6. Which claims are untouched-test evidence versus validation selection versus
   post-selection strategy diagnostics.

Tone: candid, precise, measured, and useful. Do not write a victory lap or an
apology. Avoid claims of causality, alpha, production readiness, or superiority
unless the saved evidence directly supports them.

Working title: **Learning to Rank Sector Returns: What Improved, What Broke, and
What the Search Actually Taught Us**.

## Required deliverables

Produce:

- `report.html`: responsive, self-contained, and suitable for GitHub Pages.
- `report_data.json`: the normalized, bounded dataset used by the report.
- `report_sources.json`: source file, field, metric-definition, and extraction
  metadata for every chart and quantitative statement.
- `chart_manifest.json`: chart id, section, question, source, fields, filters,
  comparison basis, and supported claim.
- `report_build_log.md`: warnings, exclusions, unresolved joins, and QA results.
- `assets/model-flow-lambdamart.svg`: senior-level boosted-tree flow.
- `assets/model-flow-lambdarank.svg`: senior-level neural ranking flow.

Do not use Torchview. Do not draw individual tensor operations or every tree.
The two model-flow figures must be editorial diagrams, not runtime graphs.

The neural diagram may use the attached rough NN image as a composition
reference, but it must show the reconciled best-trial configuration: input
feature groups, cross-sectional normalization, effective hidden width/depth,
activation, normalization, dropout if active, scalar score, grouped date list,
and LambdaRank pairwise/NDCG weighting. The boosted-tree diagram must show the
feature matrix, date-grouped ranking objective, effective LightGBM/XGBoost
backend, depth/leaves, truncation level where applicable, score ordering, and
tail portfolio mapping. Parameters must be read from the selected artifacts;
never copy them from the rough image or from a prior HTML report.

The report is an editorial HTML story, not a dashboard wall. Use a restrained
sequence of sections, small multiples, distribution views, fold heatmaps,
model-flow diagrams, and compact evidence tables. Every quantitative statement
must resolve to `report_sources.json`. No Perspective, no runtime graph dump,
no Torchview, and no external data fetch at render time.

## Canonical source locations

Repository root:

`/Users/grokking/Documents/Learning2Rank`

### Raw datasets

- Ranked universe:
  `/Users/grokking/Documents/Learning2Rank/database/datasets/spdr_sectors_2018_2025.csv`
- SPY benchmark:
  `/Users/grokking/Documents/Learning2Rank/database/datasets/spdr_sectors_2018_2025_spy.csv`

Treat the sector file and SPY file as separate sources. Do not infer SPY from the
ranked universe.

### MLflow

- Tracking database:
  `/Users/grokking/Documents/Learning2Rank/artifacts/mlflow.db`
- Optional legacy artifact store:
  `/Users/grokking/Documents/Learning2Rank/mlruns`

Use the SQLite database as the run registry. Relevant tables include `runs`,
`experiments`, `params`, `latest_metrics`, `metrics`, and `tags`. At the time of
this brief, the database contains 23 finished runs in `Learning2Rank_sectors`.
Do not assume that count remains fixed; discover it at build time.

### Completed Optuna evidence

Neural LambdaRank, 100 trials:

`/Users/grokking/Documents/Learning2Rank/artifacts/optuna/LRank_sectors_both_broad_smooth_1`

Legacy LightGBM-only LambdaMART, 200 trials:

`/Users/grokking/Documents/Learning2Rank/artifacts/optuna/LMart_sectors_both_architecture_1`

Mixed LightGBM/XGBoost tree study, 200 trials:

`/Users/grokking/Documents/Learning2Rank/artifacts/optuna/TreeRank_sectors_both_architecture_1`

Within each completed study, prefer these files:

- `study_manifest.json`: study contract and selection rule.
- `optuna_trials_clean.csv`: one row per completed trial.
- `optuna_fold_scores.csv`: fold/seed evidence.
- `optuna_clusters.csv`: stability clusters.
- `optuna_parameter_importance.csv`: study-level importance, not SHAP.
- `optuna_selection.json`: stable-cluster selection and best single trial.
- `study.db`: audit and recovery source; use only when CSV/JSON artifacts are
  incomplete or inconsistent.

Potentially running or incomplete studies must not be merged into the completed
500-trial headline. Detect completion from artifacts and trial states. Label
partial studies separately or omit them with a build-log note.

### Packaged model iterations

LambdaMART packages:

`/Users/grokking/Documents/Learning2Rank/artifacts/lambdamart/LMart_sectors_*`

LambdaRank packages:

`/Users/grokking/Documents/Learning2Rank/artifacts/lambdarank/LRank_sectors_*`

For each package, discover and join:

- `run_config.json`
- `ranking_metrics.json`
- `backtest_summary.json`
- `training_history.csv`
- `feature_importance.csv`
- `closed_lots.csv`
- `equity_curve.csv`
- `strategy_trade_summary.csv`
- `strategy_trade_summary_by_asset.csv`
- `dataset_manifest.json`
- `artifact_manifest.json`
- `report.html`, only as a visual reference, never as a numeric source

Use `run_config.json` and MLflow parameters to match an Optuna selection to a
subsequent packaged run. Never match only by iteration number or model name.
Compare all material parameters, feature set, tail mode, dataset fingerprint,
seed, strategy mode, and label horizon. If no exact final run exists, say
“selected on walk-forward validation; untouched-test package not yet available.”

## Evidence hierarchy

Apply this order whenever sources disagree:

1. Packaged immutable CSV/JSON artifact for the exact run.
2. Optuna study manifest and clean trial/fold exports for selection evidence.
3. MLflow database for run chronology, parameters, and logged metrics.
4. Raw Optuna SQLite database for reconciliation.
5. Existing HTML reports only for layout/reference checks.

Record conflicts. Do not average conflicting values.

## Metric contract

Keep these evidence classes visibly separate:

- **Walk-forward selection:** Optuna fold means, fold dispersion, stability-
  penalized objective, cluster membership, and best single trial.
- **Validation diagnostics:** ranking metrics used during ordinary model runs.
- **Untouched test:** final ranking and realized strategy metrics from a packaged
  run that exactly matches the selected configuration.
- **Post-selection diagnostics:** ATR trailing-stop or other strategy changes run
  after model selection. Never present these as model-selection evidence.

Define the principal metrics in plain language:

- NDCG: ordering quality with more weight near the actionable tail.
- MAP/MRR/ERR: complementary top/bottom retrieval diagnostics.
- Realized strategy return: closed-lot P&L only; terminal open lots are excluded.
- Realized Sharpe: annualized zero-risk-free Sharpe of the realized equity path.
- Stability objective: fold mean minus the configured penalty times dispersion.
- Stable trial: best member of the selected multi-trial stability cluster.
- Best single trial: highest individual objective, shown as more selection-risky.

Do not call the Optuna objective “return.” It is Sharpe in the completed studies.
Do not compare a value near `1.0` from Optuna with a percentage return.

## Verified anchor facts to reconcile, not blindly copy

Recompute these from source and flag any difference:

- Completed study evidence currently totals 500 trials: 100 neural, 200
  LightGBM-only, and 200 mixed-backend tree trials.
- Neural stable selection: trial 99, long-only, no MACD, one 128-unit hidden
  layer, LeakyReLU, no normalization, NAdam, sigma 1.25.
- Neural best single trial: trial 79, also long-only and no MACD, but Adam and a
  different slope/dropout/regularization configuration.
- Mixed-tree stable and best selection: trial 159, LightGBM, long-only, no MACD,
  depth 2, effective leaves 3, truncation 10, minimum child samples 60.
- Legacy LightGBM stable selection: trial 194; best single trial: 132.
- All three completed Optuna selections favored long-only.
- The selected mixed-tree trial used LightGBM rather than XGBoost.

These are narrative anchors only after source reconciliation. They do not prove
that shorts are universally unprofitable or that MACD has no value in other data.

## Report spine

### 1. Title

Use the working title or a shorter equivalent. Add one quiet scope line:
11 U.S. sector ETFs, daily data, SPY benchmark, 2018–2025 source period.

### 2. Executive Summary

Use three or four short bullets:

- The experiment became more informative when it shifted from isolated runs to
  purged walk-forward studies and stability-based selection.
- Long-only configurations dominated the completed searches; the short side was
  the persistent source of fragility, especially in low-volatility conditions.
- Simpler effective capacity won: one neural hidden layer and a shallow
  LightGBM configuration, despite larger requested knobs in some trials.
- The strongest lesson is methodological: ranking metrics, realized Sharpe, and
  absolute return answer different questions and must remain separated.

Only retain bullets supported by the final extracted evidence.

### 3. The starting point was plausible but unstable

Show the earliest comparable LambdaMART and LambdaRank packaged runs. State their
tail mode, model assumptions, ranking metrics, strategy return, Sharpe, and SPY
comparison. Avoid selecting the worst baseline merely to dramatize improvement.

Visual: paired baseline equity curves against SPY, with identical date scales.

Adjacent interpretation: the baseline established a functioning pipeline, not a
credible investment result. Explain any early HTML, packaging, or strategy bugs
only if they altered numbers; implementation mishaps belong in a small timeline,
not the main financial comparison.

### 4. Discipline changed the quality of the evidence

Tell the progression through a horizontal experiment timeline:

- reproducible iteration packages and MLflow logging;
- symmetric long/short metrics beyond NDCG;
- smooth percentile relevance and elite weighting;
- explicit tail-mode experiments;
- modern neural activation/normalization/optimizer controls;
- LightGBM versus XGBoost as a model-family knob;
- purged expanding folds, two seeds, stability penalty, untouched test;
- Strategy 2 ATR trailing exits, clearly labeled post-selection.

Visual: milestone timeline keyed to MLflow start times and artifact creation
times. Distinguish technical fixes from conceptual experiments.

### 5. The search consistently preferred one tail

Visual A: distribution of fold-level Sharpe by model family and tail mode. Use a
box/violin plus jittered trial means; show sample counts.

Visual B: an actual **cluster map** built from `optuna_clusters.csv`, not another
trial scatter. One bubble represents one hyperparameter cluster. Plot cluster
mean Sharpe against mean fold dispersion; use bubble area for the number of
unique completed trials, fill for tail mode, and color for study/model family.
Ring the selected cluster by matching its defining fields to
`optuna_selection.json`. Put the complete cluster definition and
`across_trial_std` in the table, not in an overloaded legend.

Interpretation: explain whether long-only dominance is broad across several
multi-trial clusters or concentrated in singleton/small clusters. A stable trial
is a member chosen from a selected cluster; it is not itself a cluster. Do not
infer that a long-only result validates the ranking model independently of the
market regime.

### 6. Stable selection traded peak Sharpe for a repeatable region

Use two explicitly separated evidence layers:

1. A same-unit dot/dumbbell chart containing only raw mean walk-forward Sharpe:
   selected-cluster mean, stable-trial mean, and best-single-trial mean.
2. A visible decomposition table for stable and best-single candidates:
   raw mean Sharpe, fold standard deviation, `0.25 × fold_std` penalty, and the
   resulting objective.

Never place an objective marker on an axis labeled Sharpe, even if a caption says
the quantities differ. Do not use fold extrema as endpoints of a dumbbell whose
other marks are study-level means. Include the selection rule beside the visual
and explain why the stable choice may have lower raw mean Sharpe or objective
than the best isolated trial.

### 7. The winning models were simpler than the search space

Place the two model-flow SVGs here, full width, one after the other. Follow each
with a compact “requested versus effective” parameter table.

#### LambdaMART / tree-ranker flow figure

Mimic the attached leaf-wise expansion reference, but use an executive pipeline
rather than a generic tree tutorial. Left to right:

1. **Daily cross-section**: 11 sector ETFs at one timestamp.
2. **Available signals**: returns, volatility/range/ATR, volume, VWAP; visually
   mute MACD because the selected stable trial excluded it.
3. **Cross-sectional standardization**: z-score within the timestamp.
4. **Integer relevance**: smooth percentile grades 0–10; long-only actionable
   tail highlighted.
5. **Leaf-wise boosted trees**: show three small sequential trees, each correcting
   the prior ensemble. Do not draw hundreds of trees.
6. **Actual selected knobs around the tree block**:
   LightGBM; depth 2; requested leaves 31; effective leaves 3; minimum child
   samples 60; learning rate approximately 0.0562; feature fraction 1.0; bagging
   fraction 1.0; max bin 31; L1 approximately 4.874; L2 approximately 0.000286;
   LambdaRank truncation 10; no MACD.
7. **Additive ranking score**: one scalar score per ETF.
8. **Portfolio action**: rank within date, take top two in long-only mode.
9. **Separate exit-policy box**: fixed 6% take-profit/5% stop baseline; Strategy 2
   ATR trail is a later diagnostic, not part of model training.

Use leaf-wise visual language: compact branching trees, highlighted expandable
leaf, and additive arrows. Do not imply that `num_leaves=31` was effective when
depth 2 capped effective leaves at 3. Show the cap explicitly; it is an important
senior-level lesson about requested versus realized complexity.

For XGBoost, add a small rejected-alternative callout: tested as a backend knob;
no LightGBM truncation parameter; native top-k pair construction; not selected by
the completed mixed-backend study. Do not draw a second full tree diagram unless
the evidence warrants it.

#### Neural LambdaRank flow figure

Mimic the attached input-hidden-output network image, but simplify it to grouped
blocks. Left to right:

1. **Daily cross-section**: 11 ETFs × selected features.
2. **Selected signals**: ten features; MACD muted/excluded.
3. **Cross-sectional z-score**.
4. **One hidden layer**: 128 units represented by 5–7 circles, not 128 circles.
5. **LeakyReLU**: slope approximately 0.0604 for the stable trial.
6. **No normalization**: show a crossed-out normalization chip, not an omitted
   fact.
7. **Dropout note**: the selected configuration records dropout 0.05, but the
   current builder inserts dropout only between hidden layers. With one hidden
   layer, dropout is not instantiated. Label this “requested knob; inactive in
   effective graph.” Do not draw an active dropout layer.
8. **Linear scalar output**: no sigmoid or softmax.
9. **Grouped LambdaRank loss**: within-date pairs only, NDCG-weighted, sigma 1.25,
   long-only tail for the selected trial.
10. **Optimizer and controls**: NAdam; learning rate approximately 0.000717;
    weight decay approximately `1.02e-8`; gradient clipping 0.1.
11. **Portfolio action and exit-policy box**, matching the tree diagram.

Use restrained arrows and broad blocks. The point is information flow and the
effective selected architecture, not neural-network decoration.

### 8. Ranking quality and strategy outcomes diverged

Visual: scatter of final packaged test NDCG versus realized Sharpe, with return
encoded by color and model family by shape. Add SPY Sharpe as a horizontal dashed
reference, but do not assign SPY an NDCG coordinate.

Use only packaged runs with compatible test windows and strategy assumptions.
If assumptions differ, facet or filter; never place incompatible points together.

Adjacent interpretation: a model can rank somewhat better without generating a
better realized path, particularly when exits and the short side dominate P&L.

### 9. The short side carried disproportionate fragility

Visual A: long versus short closed-lot expected return and stop rate by packaged
run, limited to comparable two-tail runs.

Visual B: asset-level stop concentration for XLE, XLK, and all other sectors.
Use counts and dollars; do not rely on percentages alone.

Visual C: trade-speed comparison using entry-to-exit signed ROC and bars held,
split by side and profitable/loss outcome.

Phrase as association. Do not say mean reversion caused losses unless a separate
causal design exists.

### 10. Volatility became both a signal and an execution question

Visual: rank/importance trajectory for ATR, volatility, VWAP deviation, returns,
volume, and volatility-adjusted MACD across iterations. Use rank positions rather
than raw importance when model families use incomparable scales.

Then show Strategy 2 separately:

- fixed stop versus ATR trailing for the exact selected model;
- long multiplier and wider short multiplier;
- realized return, Sharpe, drawdown, stop count, bars held, and asset
  concentration;
- explicit badge: **post-selection diagnostic; not included in model selection**.

Do not include smoke-test values as final evidence. Use only completed one-off
packages matching the stable selected model configurations.

### 11. What the search did not establish

State plainly:

- one small universe and one historical period cannot establish broad efficacy;
- repeated experimentation raises selection risk despite walk-forward folds;
- SPY is a benchmark, not a risk-matched portfolio;
- no transaction costs, slippage, borrow costs, or capacity model should be
  assumed unless the artifacts explicitly contain them;
- long-only superiority may reflect the sample's market regime;
- Strategy 2 is post-selection until evaluated under a predeclared protocol;
- feature importance is not causal attribution;
- the selected trials require exact untouched-test reruns before final claims.

### 12. Recommended next steps

Keep to four actions:

1. Freeze the selected stable neural and tree configurations.
2. Run Strategy 1 and Strategy 2 once each under identical untouched-test data,
   with no further tuning.
3. Add costs and a risk-matched long-only benchmark before publication claims.
4. Repeat the frozen protocol on a separately named dataset before changing
   features or labels again.

### 13. Further questions

Ask only questions that could change the conclusion:

- Does the long-only result survive a different asset universe or period?
- Does a wider short ATR trail improve robustness without merely increasing loss
  tolerance?
- Are top-decile label weights helping rank quality, strategy outcomes, or only
  validation selection?
- Does the no-MACD selection persist when the feature family is evaluated on a
  new dataset?

## Chart production rules

- Every chart needs an adjacent paragraph containing takeaway, reading guidance,
  implication, and caveat.
- Use full-width charts for trial distributions, equity paths, and fold heatmaps.
- Use consistent color semantics: neural, LightGBM, and XGBoost each receive one
  stable family color; long-only and two-tail use line style or secondary tone.
- SPY is always a neutral dashed benchmark, never a model-family color.
- Stable trial uses a ring; best single trial uses a star; baseline uses a square.
- Show sample counts and missing-data exclusions.
- Never connect unordered Optuna trial numbers as if they were a time series.
- Do not use dual axes unless one axis is explicitly normalized and the visual
  remains unambiguous.
- Prefer distributions, fold heatmaps, scatterplots, dumbbells, and rank-change
  charts over decorative gauges.
- Provide a compact data table or semantic fallback for every interactive chart.

## HTML experience

- Single-column editorial reading path, not a dashboard wall.
- Sticky mini table of contents on desktop; collapsible on mobile.
- Executive Summary must be the first section after the title.
- Optional detail drawers for exact parameters, fold scores, and source metadata.
- No raw JSON in the visible page.
- No local-file fetches at runtime. Embed the bounded report data or use relative
  packaged assets so GitHub Pages works over HTTPS.
- Avoid external CDNs where practical. If a library is used, vendor and pin it.
- Support light/dark system appearance and print-friendly rendering.
- All SVG model diagrams must have text alternatives and remain readable at
  768-pixel width.
- Do not expose absolute local paths in the published HTML; retain them only in
  the local source manifest and replace with repository-relative display names.

## Join and reconciliation rules

1. Normalize run names but preserve original names.
2. Join MLflow to package directories by run name plus parameter fingerprint.
3. Join Optuna selection to packages by complete material-parameter comparison,
   not trial number.
4. Verify dataset fingerprint, date bounds, horizon, tail mode, feature list,
   model family, and strategy assumptions before comparing runs.
5. Deduplicate repeated `LRank_sectors_1` MLflow entries by run UUID; never drop
   duplicates silently.
6. Treat missing final packages as missing evidence, not zero performance.
7. Record all exclusions and reasons in `report_build_log.md`.

## Macro-observation protocol

Generate observations only after the normalized dataset exists. For every
observation, save:

- claim;
- supporting chart/table ids;
- source ids;
- comparison population and denominator;
- effect size or concentration;
- confidence label: verified, suggestive, or unresolved;
- nearest plausible alternative explanation;
- publication-safe wording.

Candidate observations to test:

- long-only advantage appears across model families;
- stable-cluster selection narrows the gap to best single trials;
- simpler effective capacity is associated with stronger stable outcomes;
- MACD exclusion is common among stable selections;
- fold dispersion remains material despite favorable average Sharpe;
- short-side stop concentration explains a meaningful share of strategy drag;
- feature-rank instability exceeds metric movement across some iterations;
- neural and tree rankers converge on similar portfolio choices despite different
  internal mechanisms.

Reject any candidate that the extracted evidence does not support.

## Acceptance tests

The production is complete only when:

- every visible number traces to `report_sources.json`;
- completed and partial Optuna studies are separated;
- stable and best-single trials are never conflated;
- walk-forward objective is labeled Sharpe, not return;
- no test metric influenced Optuna selection;
- model-flow figures match effective implementation, including inactive neural
  dropout and capped LightGBM leaves;
- LightGBM truncation is never shown as an XGBoost or neural knob;
- Strategy 2 is labeled post-selection;
- all compared equity curves share date range and strategy assumptions;
- SPY remains visible wherever strategy performance is discussed;
- no raw JSON or machine-local absolute path appears in published HTML;
- desktop, mobile, keyboard, print, and offline/static-host checks pass;
- the report remains understandable when charts fall back to their semantic
  tables.

## Final editorial test

A senior reader should be able to skim the title, Executive Summary, section
headings, and the two model-flow diagrams and accurately repeat this story:

> A disciplined search improved the quality of evidence more than it proved a
> winning strategy. Across neural and tree rankers, the completed studies favored
> long-only and relatively simple effective models. The short side, regime
> sensitivity, and the gap between ranking metrics and realized outcomes remain
> the central unresolved issues. The work is valuable because those limitations
> are measured, reproducible, and explicit.
