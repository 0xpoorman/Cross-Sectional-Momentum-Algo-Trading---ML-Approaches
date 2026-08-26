# First Phase — LambdaMART and LambdaRank

This is a self-contained reproduction of the original first comparison. It is
intentionally separate from the later Optuna and production-style experiments.

## Frozen Phase 1 contract

- 11 SPDR sector ETFs; SPY is not ranked.
- Ten original OHLCV-derived features; no MACD.
- Cross-sectional z-score per signal date.
- Chronological 70/15/15 split with a three-bar purge.
- Forward label: next open to the open three bars later.
- Top two long and bottom two short assets.
- Exactly three-bar holding period using non-overlapping cohorts.
- No take-profit, no stop-loss, no trailing exit, and zero transaction costs.
- LambdaMART uses full date groups; `lambdarank_truncation_level` is not set.
- LambdaRank uses ReLU + LayerNorm and a two-tail NDCG-weighted pairwise loss.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python recreate_first_trials.py \
  --dataset /absolute/path/to/spdr_sectors_2018_2025.csv \
  --output-root runs/phase1 \
  --report first_trials_report.html \
  --force
```

The default dataset path is the parent Learning2Rank repository's sector CSV,
so the command can be shortened when this directory remains inside that repo.

Outputs are isolated under `runs/phase1/`:

```text
runs/phase1/
  contract.json
  mlflow.db
  mlartifacts/
  lmart/{summary.json,training_history.csv,backtest.csv,test_scores.csv,model.txt}
  lrank/{summary.json,training_history.csv,backtest.csv,test_scores.csv,model.pt}
```

The high-level comparison is the separate `first_trials_report.html` in this
directory. It embeds only summary data and does not require MLflow to render.
The primary repository-level codebase map is the readable dark-purple SVG at
`assets/phase1-codebase-dark.svg`. The Archify specification and interactive
companion remain in `assets/phase1-codebase.architecture.json` and
`assets/phase1-codebase.html`. Together they show how the entry point, helper
modules, key functions, two training branches, evaluation, backtest, MLflow,
and report outputs interact. The older experiment-flow diagram remains
available as `assets/phase1-architecture.html`.

To regenerate the diagram after editing the JSON (Archify is installed by the
skill installer into the local Codex skills directory):

```bash
node /Users/grokking/.codex/skills/archify/bin/archify.mjs validate architecture \
  assets/phase1-codebase.architecture.json --repo-root .. --quality showcase --json
node /Users/grokking/.codex/skills/archify/bin/archify.mjs deliver architecture \
  assets/phase1-codebase.architecture.json assets/phase1-codebase.html \
  --repo-root .. --quality showcase --json
```

The checked reproduction produced MLflow runs `592df6fd444a40b2a5fa4bcec0b489c8`
(LambdaMART) and `97fd8f075fea4c3c81ddfb5cc0ea03c5` (LambdaRank). The neural
run reproduces the important shape of the supplied Phase 1 evidence: training
loss falls from about 0.727 to 0.695, validation loss remains near 0.694, and
NDCG stays around 0.39–0.40. LightGBM early stopping means its saved history
may contain fewer than the screenshot's 45 points; this is an approximate
reproduction, not a claim of byte-identical historical state.

## MLflow

```bash
source .venv/bin/activate
mlflow ui --backend-store-uri sqlite:///$(pwd)/runs/phase1/mlflow.db --port 5001
```

Open `http://127.0.0.1:5001` and select `Legacy_First_Trials_FixedH3`.

## Tests

```bash
python -m unittest discover -s . -p 'test_*.py' -v
```

The run is expected to be weak. Pairwise accuracy near 50% is chance-like;
NDCG around 0.39–0.41 can coexist with a validation loss near `ln(2)`.

## Publishing as one GitHub repository

This directory is deliberately self-contained: copy it to a new repository,
then run the install command above. Generated model binaries, MLflow's local
database, and artifact store are ignored; compact summaries, histories,
backtests, the report, and the Archify diagram can be committed as evidence.
No remote or commit is created automatically.
