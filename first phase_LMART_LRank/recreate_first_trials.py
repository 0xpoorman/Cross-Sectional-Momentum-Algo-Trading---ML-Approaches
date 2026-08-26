#!/usr/bin/env python3
"""Recreate isolated Phase 1 LambdaMART and LambdaRank MLflow runs.

This runner deliberately does not import the repository's mutable model configs,
does not use WFO/Optuna, and never writes to the current artifacts/mlflow.db.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
SRC_ROOT = PACKAGE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from first_phase.helpers.features import (  # noqa: E402
    engineer_features,
    filter_complete_dates,
    load_frame,
    select_ranked_universe,
)
from first_phase.helpers.model import fit_ranker  # noqa: E402
from first_phase.helpers.preprocessing import cross_sectional_zscore  # noqa: E402
from first_phase.helpers.metrics import evaluate_rankings  # noqa: E402
from first_phase.helpers.splits import chronological_split  # noqa: E402


FEATURES = [
    "ret_1", "ret_5", "ret_20", "volatility_20", "range_1",
    "atr_14", "close_location", "volume_z_20", "volume_change_1",
    "vwap_dev_20",
]


@dataclass(frozen=True)
class TrialContract:
    """Immutable contract for the historical first-trial comparison.

    The contract is deliberately narrower than the modern production pipeline:
    ten original OHLCV-derived features, one purged chronological split, a
    three-bar open-to-open holding period, zero trading costs, and no intratrade
    stop or take-profit. Keeping it explicit prevents current defaults from
    silently changing the historical experiment.
    """
    dataset: str
    horizon: int = 3
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    min_symbols: int = 11
    top_n: int = 2
    bottom_n: int = 2
    seed: int = 7
    strategy: str = "fixed_horizon"
    walk_forward_optimization: bool = False
    stop_loss: None = None
    take_profit: None = None
    transaction_cost_bps: float = 0.0


def parse_args() -> argparse.Namespace:
    """Parse reproducibility controls without exposing modern search knobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path,
        default=REPO_ROOT / "database/datasets/spdr_sectors_2018_2025.csv",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=PACKAGE_ROOT / "runs/phase1",
    )
    parser.add_argument("--force", action="store_true", help="Replace only the isolated output root")
    parser.add_argument("--lrank-epochs", type=int, default=200)
    parser.add_argument("--lmart-rounds", type=int, default=100)
    parser.add_argument(
        "--report",
        type=Path,
        default=PACKAGE_ROOT / "first_trials_report.html",
        help="Separate high-level HTML report path",
    )
    return parser.parse_args()


def prepare(contract: TrialContract) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Load, engineer, purge, and z-score the exact Phase 1 data contract."""
    indexed = select_ranked_universe(load_frame(contract.dataset))
    frame = filter_complete_dates(
        engineer_features(indexed, contract.horizon), contract.min_symbols
    )
    # Recreate trial 1: the original ten features, before MACD was added.
    frame = frame.dropna(subset=FEATURES).copy()
    train, validation, test = chronological_split(
        frame, contract.horizon, contract.train_fraction, contract.validation_fraction
    )

    def flatten(part: pd.DataFrame) -> pd.DataFrame:
        return cross_sectional_zscore(part.reset_index(), FEATURES)

    fingerprint = hashlib.sha256(Path(contract.dataset).read_bytes()).hexdigest()
    return flatten(train), flatten(validation), flatten(test), fingerprint


def score_metrics(frame: pd.DataFrame, scores: np.ndarray, contract: TrialContract) -> dict[str, float]:
    """Evaluate a score vector against grouped tail-ranking metrics."""
    scored = frame[["signal_datetime", "symbol", "forward_return"]].copy()
    scored["score"] = np.asarray(scores, dtype=float)
    return evaluate_rankings(
        scored, top_size=contract.top_n, bottom_size=contract.bottom_n
    )["summary"]


def fixed_horizon_backtest(
    frame: pd.DataFrame, scores: np.ndarray, contract: TrialContract
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Backtest non-overlapping three-bar cohorts with no stop logic.

    A cohort is opened at the next available open and closed at the open three
    bars later. Cohorts are skipped while the preceding cohort is held, so the
    result cannot accidentally overlap capital. Long and short legs are equal
    notional; transaction costs remain zero under the Phase 1 contract.
    """
    scored = frame[[
        "signal_datetime", "entry_datetime", "exit_datetime", "symbol", "forward_return"
    ]].copy()
    scored["score"] = np.asarray(scores, dtype=float)
    rows: list[dict[str, Any]] = []
    groups = list(scored.groupby("signal_datetime", sort=True))
    cost = contract.transaction_cost_bps / 10_000.0
    for group_number, (timestamp, group) in enumerate(groups):
        # Full-capital cohorts cannot overlap. A new portfolio is formed every h bars.
        if group_number % contract.horizon:
            continue
        ordered = group.sort_values(["score", "symbol"], ascending=[False, True], kind="stable")
        longs = ordered.head(contract.top_n)
        shorts = ordered.tail(contract.bottom_n)
        long_return = float(longs["forward_return"].mean())
        short_asset_return = float(shorts["forward_return"].mean())
        gross = 0.5 * long_return - 0.5 * short_asset_return
        # One round trip on one unit of gross notional.
        net = gross - 2.0 * cost
        rows.append({
            "signal_datetime": timestamp,
            "entry_datetime": group["entry_datetime"].min(),
            "exit_datetime": group["exit_datetime"].max(),
            "long_symbols": ",".join(longs["symbol"]),
            "short_symbols": ",".join(shorts["symbol"]),
            "long_return": long_return,
            "short_asset_return": short_asset_return,
            "gross_return": gross,
            "net_return": net,
        })
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("backtest produced no non-overlapping cohorts")
    result["equity"] = (1.0 + result["net_return"]).cumprod()
    result["drawdown"] = result["equity"] / result["equity"].cummax() - 1.0
    returns = result["net_return"]
    periods_per_year = 252.0 / contract.horizon
    volatility = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    total_return = float(result["equity"].iloc[-1] - 1.0)
    metrics = {
        "cohort_count": float(len(result)),
        "total_return": total_return,
        "annualized_return": float(
            result["equity"].iloc[-1] ** (periods_per_year / len(result)) - 1.0
        ),
        "annualized_volatility": volatility * math.sqrt(periods_per_year),
        "sharpe_zero_rf": (
            float(returns.mean() / volatility * math.sqrt(periods_per_year))
            if volatility > 0.0 else 0.0
        ),
        "max_drawdown": float(result["drawdown"].min()),
        "hit_rate": float((returns > 0.0).mean()),
        "mean_cohort_return": float(returns.mean()),
        "annualization_periods": periods_per_year,
    }
    return result, metrics


def train_lrank(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame,
    contract: TrialContract, epochs: int,
) -> tuple[Any, dict[str, Any]]:
    """Train the historical neural ranker with ReLU and LayerNorm.

    The scalar network score is optimized using the repository's grouped
    LambdaRank-style objective. No sigmoid or softmax is applied because scores
    are used only to order the eleven assets within each date.
    """
    # No ridge fields are present: this is the neural LambdaRank path only.
    config = SimpleNamespace(
        model_family="neural", preprocessing="cross_sectional_zscore", device="cpu",
        seed=contract.seed, hidden_dim=64, hidden_layers=2, dropout=0.3,
        activation="ReLU",
        leaky_relu_negative_slope=0.01,
        normalization="LayerNorm", learning_rate=1e-3, weight_decay=1e-5,
        optimizer="AdamW", max_gradient_norm=0.0, sigma=1.0, epochs=epochs,
        patience=12, min_epochs=13, checkpoint_start_epoch=1,
        # Phase 1 predates elite-positive weighting. None is important: using
        # 1.0 with the modern helper creates a zero-width percentile interval
        # and silently removes every LambdaRank pair.
        elite_positive_quantile=None, elite_label_gain=0.0, torch_num_threads=1,
    ) 
    fit = fit_ranker(
        train, validation, FEATURES, config,
        top_size=contract.top_n, bottom_size=contract.bottom_n,
    )
    scores = {
        "train": fit.predict(train),
        "validation": fit.predict(validation),
        "test": fit.predict(test),
    }
    return fit, {"config": vars(config), "scores": scores, "history": fit.history,
                "diagnostics": fit.diagnostics}


def percentile_relevance(returns: np.ndarray) -> np.ndarray:
    """Convert each date's continuous returns to eleven LightGBM grades."""
    ranks = pd.Series(returns).rank(method="average").to_numpy(float) - 1.0
    return np.rint(ranks * 10.0 / max(1, len(returns) - 1)).astype(np.int32)


def lmart_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build date-contiguous features, relevance labels, and group sizes."""
    ordered = frame.sort_values(["signal_datetime", "symbol"], kind="stable")
    labels, groups = [], []
    for _, group in ordered.groupby("signal_datetime", sort=True):
        groups.append(len(group))
        labels.extend(percentile_relevance(group["forward_return"].to_numpy(float)))
    return ordered[FEATURES], np.asarray(labels, np.int32), np.asarray(groups, np.int32)


def train_lmart(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame,
    contract: TrialContract, rounds: int,
) -> tuple[Any, dict[str, Any]]:
    """Train Phase 1 LightGBM LambdaMART without truncation.

    `lambdarank_truncation_level` is intentionally absent. Every date group is
    available to the objective; only evaluation uses NDCG@2 to reflect the two
    long and two short portfolio slots.
    """
    train_x, train_y, train_groups = lmart_matrix(train)
    valid_x, valid_y, valid_groups = lmart_matrix(validation)
    label_gain = np.expm1(np.linspace(0.0, np.log1p(15.0), 11)).tolist()
    params = {
        "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [2],
        "label_gain": label_gain, "learning_rate": 0.05, "max_depth": 3,
        "num_leaves": 7, "min_child_samples": 20, "seed": contract.seed,
        "deterministic": True, "force_col_wise": True, "verbosity": -1,
        # Intentionally no lambdarank_truncation_level.
    }
    train_set = lgb.Dataset(train_x, label=train_y, group=train_groups, feature_name=FEATURES)
    valid_set = lgb.Dataset(valid_x, label=valid_y, group=valid_groups,
                            reference=train_set, feature_name=FEATURES)
    history: dict[str, dict[str, list[float]]] = {}
    model = lgb.train(
        params, train_set, num_boost_round=rounds,
        valid_sets=[train_set, valid_set], valid_names=["train", "validation"],
        callbacks=[lgb.early_stopping(10), lgb.record_evaluation(history)],
    )
    best = int(model.best_iteration or model.current_iteration())
    scores = {
        "train": model.predict(train[FEATURES], num_iteration=best),
        "validation": model.predict(validation[FEATURES], num_iteration=best),
        "test": model.predict(test[FEATURES], num_iteration=best),
    }
    flat_history = []
    count = max(len(values) for dataset in history.values() for values in dataset.values())
    for index in range(count):
        row: dict[str, float] = {"iteration": float(index + 1)}
        for dataset, metrics in history.items():
            for name, values in metrics.items():
                if index < len(values):
                    row[f"{dataset}_{name}"] = float(values[index])
        flat_history.append(row)
    return model, {"config": params, "scores": scores, "history": flat_history,
                "diagnostics": {"best_iteration": best, "truncation": "none"}}


def run_model(
    method: str, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame,
    contract: TrialContract, output_dir: Path, fingerprint: str,
    lrank_epochs: int, lmart_rounds: int,
) -> dict[str, Any]:
    """Train one model, score all splits, and write auditable run artifacts."""
    if method == "lrank":
        model, result = train_lrank(train, validation, test, contract, lrank_epochs)
    else:
        model, result = train_lmart(train, validation, test, contract, lmart_rounds)
    ranking = {
        split: score_metrics(frame, np.asarray(result["scores"][split]), contract)
        for split, frame in (("train", train), ("validation", validation), ("test", test))
    }
    backtest, backtest_metrics = fixed_horizon_backtest(
        test, np.asarray(result["scores"]["test"]), contract
    )
    summary = {
        "method": "LambdaRank" if method == "lrank" else "LambdaMART",
        "contract": asdict(contract), "dataset_sha256": fingerprint,
        "features": FEATURES, "model_config": result["config"],
        "diagnostics": result["diagnostics"], "ranking": ranking,
        "backtest": backtest_metrics,
        "overfit_pairwise_accuracy_gap": (
            ranking["train"]["pairwise_accuracy"] - ranking["test"]["pairwise_accuracy"]
        ),
    }
    output_dir.mkdir(parents=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    pd.DataFrame(result["history"]).to_csv(output_dir / "training_history.csv", index=False)
    backtest.to_csv(output_dir / "backtest.csv", index=False)
    scored = test[["signal_datetime", "symbol", "forward_return"]].copy()
    scored["score"] = np.asarray(result["scores"]["test"])
    scored.to_csv(output_dir / "test_scores.csv", index=False)
    if method == "lrank":
        torch.save({"state_dict": model.model.state_dict(), "features": FEATURES}, output_dir / "model.pt")
    else:
        model.save_model(output_dir / "model.txt")
    return summary


def log_run(method: str, summary: dict[str, Any], output_dir: Path) -> str:
    """Log one isolated run and its artifacts to the Phase 1 MLflow database."""
    with mlflow.start_run(run_name=f"trial_1_{method}_fixed_h3") as run:
        mlflow.set_tags({
            "reproduction": "legacy_first_trial", "method": method,
            "isolation": "separate_tracking_database", "wfo": "false",
        })
        params = {
            "dataset": summary["contract"]["dataset"],
            "dataset_sha256": summary["dataset_sha256"],
            "horizon": 3, "strategy": "fixed_horizon", "purged": True,
            "walk_forward_optimization": False, "stop_loss": "none",
            "take_profit": "none", "ridge_regression": False,
            "features": ",".join(FEATURES),
            "lambdarank_truncation_level": (
                "not_set_full_list" if method == "lmart" else "not_applicable"
            ),
        }
        mlflow.log_params(params)
        metrics: dict[str, float] = {}
        for split, values in summary["ranking"].items():
            for key in ("pairwise_accuracy", "macro_tail_ndcg@2_2", "spearman"):
                mlflow_key = key.replace("@", "_at_")
                metrics[f"ranking/{split}/{mlflow_key}"] = float(values[key])
        metrics.update({f"backtest/{key}": float(value) for key, value in summary["backtest"].items()})
        metrics["diagnostics/overfit_pairwise_accuracy_gap"] = float(
            summary["overfit_pairwise_accuracy_gap"]
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(output_dir), artifact_path="trial_outputs")
        return run.info.run_id


def fmt_pct(value: float) -> str:
    """Format a fractional metric as a percentage for the report."""
    return f"{100.0 * value:.2f}%"


def fmt_num(value: float) -> str:
    """Format a scalar report metric with three decimal places."""
    return f"{value:.3f}"


def write_report(summaries: list[dict[str, Any]], run_ids: dict[str, str], path: Path) -> None:
    """Write the separate high-level HTML comparison report.

    The report embeds only bounded summaries, so it remains portable and does
    not need MLflow or a server at read time. Detailed training histories and
    per-date scores remain beside the report in the run directories.
    """
    cards, rows = [], []
    for summary in summaries:
        key = "lrank" if summary["method"] == "LambdaRank" else "lmart"
        test = summary["ranking"]["test"]
        bt = summary["backtest"]
        cards.append(
            f'<article><h2>{html.escape(summary["method"])}</h2>'
            f'<div class="big">{fmt_num(bt["sharpe_zero_rf"])}</div><p>3-day Sharpe</p>'
            f'<dl><dt>Test accuracy</dt><dd>{fmt_pct(test["pairwise_accuracy"])}</dd>'
            f'<dt>Test NDCG</dt><dd>{fmt_num(test["macro_tail_ndcg@2_2"])}</dd>'
            f'<dt>Total return</dt><dd>{fmt_pct(bt["total_return"])}</dd>'
            f'<dt>Max drawdown</dt><dd>{fmt_pct(bt["max_drawdown"])}</dd></dl>'
            f'<small>MLflow run: {html.escape(run_ids[key])}</small></article>'
        )
        rows.append(
            f'<tr><td>{summary["method"]}</td>'
            f'<td>{fmt_pct(summary["ranking"]["train"]["pairwise_accuracy"])}</td>'
            f'<td>{fmt_pct(summary["ranking"]["validation"]["pairwise_accuracy"])}</td>'
            f'<td>{fmt_pct(test["pairwise_accuracy"])}</td>'
            f'<td>{fmt_pct(summary["overfit_pairwise_accuracy_gap"])}</td>'
            f'<td>{fmt_num(bt["sharpe_zero_rf"])}</td></tr>'
        )
    payload = json.dumps({"runs": summaries, "mlflow_run_ids": run_ids}, default=str).replace("</", "<\\/")
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>First trial reproduction</title>
<style>:root{{--ink:#18221c;--muted:#667268;--paper:#f4f0e7;--card:#fffdf8;--line:#d8d0bf;--blue:#146b8c}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1050px;margin:auto;padding:3rem 1rem}}h1{{font-size:clamp(2rem,5vw,4rem);line-height:1;margin:.3rem 0}}.eyebrow{{color:var(--blue);text-transform:uppercase;letter-spacing:.12em;font-size:.75rem}}.lede{{max-width:780px;color:var(--muted)}}.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:1rem;margin:2rem 0}}article,.panel{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:1.3rem}}.big{{font:2.2rem ui-monospace,monospace;color:var(--blue)}}dl{{display:grid;grid-template-columns:1fr auto;gap:.45rem}}dt{{color:var(--muted)}}dd{{margin:0;font-family:ui-monospace,monospace}}table{{width:100%;border-collapse:collapse}}th,td{{padding:.7rem;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}code{{background:#e8e2d5;padding:.1rem .3rem;border-radius:4px}}@media(max-width:650px){{.grid{{grid-template-columns:1fr}}}}</style></head>
<body><main><div class="eyebrow">Isolated legacy reconstruction</div><h1>First ranking trials, replayed</h1>
<p class="lede">Same SPDR-sector dataset and one purged 70/15/15 chronological split. No walk-forward optimization. Signals hold for exactly three bars from next open to exit open; there are no stop-loss or take-profit exits. Sharpe uses non-overlapping cohorts and 252/3 annualization.</p>
<div class="grid">{''.join(cards)}</div><section class="panel"><h2>Overfitting diagnostic</h2><table><thead><tr><th>Model</th><th>Train accuracy</th><th>Validation</th><th>Test</th><th>Train–test gap</th><th>Sharpe</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>“Accuracy” is within-date pairwise ordering accuracy. Near 50% is chance-like. The gap is descriptive evidence of overfit; the untouched test set did not select either model.</p></section>
<section class="panel" style="margin-top:1rem"><h2>Reproduction contract</h2><ul><li>LambdaMART: LightGBM <code>lambdarank</code>, full lists; truncation is not configured.</li><li>LambdaRank: neural two-tail NDCG-weighted pairwise objective with ReLU and LayerNorm; no ridge baseline.</li><li>Feature set: the original ten OHLCV-derived features, no MACD.</li><li>Labels: next-open to open three bars later; no take-profit or stop-loss.</li><li>Costs: zero, matching the early research simplification.</li><li>Detailed runs and the separate report live under this package directory.</li><li><a href="assets/phase1-architecture.html">Open the labeled Archify architecture diagram</a>.</li></ul></section>
<script type="application/json" id="reproduction-data">{payload}</script></main></body></html>"""
    path.write_text(document, encoding="utf-8")


def main() -> None:
    """Execute both historical models into an isolated, collision-safe package."""
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"isolated output exists; pass --force to replace it: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    contract = TrialContract(dataset=str(args.dataset.resolve()))
    train, validation, test, fingerprint = prepare(contract)
    split = {
        "train_dates": int(train["signal_datetime"].nunique()),
        "validation_dates": int(validation["signal_datetime"].nunique()),
        "test_dates": int(test["signal_datetime"].nunique()),
    }
    (output_root / "contract.json").write_text(
        json.dumps({"contract": asdict(contract), "features": FEATURES, "split": split,
                    "dataset_sha256": fingerprint}, indent=2) + "\n"
    )

    tracking_uri = f"sqlite:///{output_root / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "Legacy_First_Trials_FixedH3"
    client = mlflow.MlflowClient()
    existing_experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = (
        existing_experiment.experiment_id
        if existing_experiment is not None
        else client.create_experiment(
            experiment_name, artifact_location=(output_root / "mlartifacts").as_uri()
        )
    )
    mlflow.set_experiment(experiment_name)
    summaries, run_ids = [], {}
    for method in ("lmart", "lrank"):
        model_dir = output_root / method
        summary = run_model(
            method, train, validation, test, contract, model_dir, fingerprint,
            args.lrank_epochs, args.lmart_rounds,
        )
        summaries.append(summary)
        run_ids[method] = log_run(method, summary, model_dir)
    report = args.report.resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    write_report(summaries, run_ids, report)
    for run_id in run_ids.values():
        client.log_artifact(run_id, str(report), artifact_path="comparison")
    print(json.dumps({
        "tracking_uri": tracking_uri, "experiment_id": experiment_id,
        "experiment_name": experiment_name, "run_ids": run_ids,
        "report": str(report), "split": split,
    }, indent=2))


if __name__ == "__main__":
    main()
