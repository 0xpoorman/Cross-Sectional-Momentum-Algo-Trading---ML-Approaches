from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


def tail_relevance(forward_returns: np.ndarray, top_size: int = 2, bottom_size: int = 2) -> np.ndarray:
    if top_size < 1 or bottom_size < 0:
        raise ValueError("top_size must be positive and bottom_size cannot be negative")
    if top_size + bottom_size >= len(forward_returns):
        raise ValueError("top_size + bottom_size must be smaller than the group size")
    order = np.argsort(forward_returns, kind="stable")
    relevance = np.ones(len(forward_returns), dtype=float)
    if bottom_size:
        relevance[order[:bottom_size]] = 0.0
    relevance[order[-top_size:]] = 2.0
    return relevance


def ndcg(scores: np.ndarray, relevance: np.ndarray, k: int) -> float:
    k = min(max(1, k), len(scores))
    discount = 1.0 / np.log2(np.arange(k) + 2.0)
    ranked = relevance[np.argsort(-scores, kind="stable")[:k]]
    ideal = np.sort(relevance)[::-1][:k]
    numerator = np.sum((2.0**ranked - 1.0) * discount)
    denominator = np.sum((2.0**ideal - 1.0) * discount)
    return float(numerator / denominator) if denominator > 0 else 0.0


def average_precision_at_k(scores: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """Average precision for binary tail membership within the first k ranks."""
    k = min(max(1, k), len(scores))
    relevant = np.asarray(relevant, dtype=bool)
    denominator = min(int(relevant.sum()), k)
    if denominator == 0:
        return 0.0
    ranked_hits = relevant[np.argsort(-scores, kind="stable")[:k]].astype(float)
    precision = np.cumsum(ranked_hits) / np.arange(1.0, k + 1.0)
    return float(np.sum(precision * ranked_hits) / denominator)


def reciprocal_rank_at_k(scores: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """Reciprocal rank of the first binary-relevant result within k."""
    k = min(max(1, k), len(scores))
    ranked = np.asarray(relevant, dtype=bool)[np.argsort(-scores, kind="stable")[:k]]
    hits = np.flatnonzero(ranked)
    return float(1.0 / (hits[0] + 1.0)) if len(hits) else 0.0


def expected_reciprocal_rank_at_k(
    scores: np.ndarray, relevance: np.ndarray, k: int
) -> float:
    """ERR using graded relevance as a rank-dependent stopping probability."""
    k = min(max(1, k), len(scores))
    relevance = np.asarray(relevance, dtype=float)
    maximum = float(relevance.max(initial=0.0))
    if maximum <= 0.0:
        return 0.0
    ranked = relevance[np.argsort(-scores, kind="stable")[:k]]
    stop_probabilities = (2.0**ranked - 1.0) / (2.0**maximum)
    continuation = 1.0
    result = 0.0
    for rank, stop_probability in enumerate(stop_probabilities, start=1):
        result += continuation * float(stop_probability) / rank
        continuation *= 1.0 - float(stop_probability)
    return result


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def spearman_correlation(scores: np.ndarray, values: np.ndarray) -> float:
    if len(scores) < 2:
        return 0.0
    rs = _rankdata(scores)
    rv = _rankdata(values)
    return pearson_correlation(rs, rv)


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denom = np.sqrt(np.sum(a_centered**2) * np.sum(b_centered**2))
    return float(np.sum(a_centered * b_centered) / denom) if denom else 0.0


def kendall_tau_b(scores: np.ndarray, values: np.ndarray) -> float:
    concordant = discordant = ties_x = ties_y = 0
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            dx = np.sign(scores[i] - scores[j])
            dy = np.sign(values[i] - values[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return float((concordant - discordant) / denominator) if denominator else 0.0


def pairwise_accuracy(scores: np.ndarray, values: np.ndarray) -> tuple[float, int]:
    correct = total = 0
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            target = np.sign(values[i] - values[j])
            if target == 0:
                continue
            total += 1
            predicted = np.sign(scores[i] - scores[j])
            if predicted == target:
                correct += 1
    return (correct / total if total else 0.0), total


def _tail_symbols(order: np.ndarray, symbols: np.ndarray, size: int, reverse: bool = False) -> set[str]:
    chosen = order[:size] if reverse else order[-size:]
    return {str(symbols[index]) for index in chosen}


def evaluate_group(
    group: pd.DataFrame,
    score_column: str = "score",
    top_size: int = 2,
    bottom_size: int = 2,
) -> dict[str, float]:
    # Stable sorting below now has an explicit, reproducible tie-break independent of input row order.
    group = group.sort_values("symbol", kind="stable")
    scores = group[score_column].to_numpy(float)
    returns = group["forward_return"].to_numpy(float)
    symbols = group["symbol"].to_numpy(str)
    relevance = tail_relevance(returns, top_size=top_size, bottom_size=bottom_size)
    maximum_relevance = relevance.max()
    long_relevant = relevance == maximum_relevance
    long_ndcg = ndcg(scores, relevance, top_size)
    long_map = average_precision_at_k(scores, long_relevant, top_size)
    long_mrr = reciprocal_rank_at_k(scores, long_relevant, top_size)
    long_err = expected_reciprocal_rank_at_k(scores, relevance, top_size)
    if bottom_size:
        short_relevance = maximum_relevance - relevance
        short_relevant = relevance == relevance.min()
        short_ndcg = ndcg(-scores, short_relevance, bottom_size)
        short_map = average_precision_at_k(-scores, short_relevant, bottom_size)
        short_mrr = reciprocal_rank_at_k(-scores, short_relevant, bottom_size)
        short_err = expected_reciprocal_rank_at_k(-scores, short_relevance, bottom_size)
    else:
        short_ndcg = short_map = short_mrr = short_err = 0.0
    pair_acc, pair_den = pairwise_accuracy(scores, returns)
    spearman = spearman_correlation(scores, returns)
    kendall = kendall_tau_b(scores, returns)

    predicted_order = np.argsort(scores, kind="stable")
    actual_order = np.argsort(returns, kind="stable")
    pred_top = _tail_symbols(predicted_order, symbols, top_size, reverse=False)
    act_top = _tail_symbols(actual_order, symbols, top_size, reverse=False)
    pred_bottom = _tail_symbols(predicted_order, symbols, bottom_size, reverse=True) if bottom_size else set()
    act_bottom = _tail_symbols(actual_order, symbols, bottom_size, reverse=True) if bottom_size else set()
    top_overlap = len(pred_top & act_top)
    bottom_overlap = len(pred_bottom & act_bottom)
    top_return = float(group.loc[group["symbol"].isin(pred_top), "forward_return"].mean())
    bottom_return = float(group.loc[group["symbol"].isin(pred_bottom), "forward_return"].mean()) if bottom_size else 0.0
    oracle_top_return = float(group.loc[group["symbol"].isin(act_top), "forward_return"].mean())
    oracle_bottom_return = float(group.loc[group["symbol"].isin(act_bottom), "forward_return"].mean()) if bottom_size else 0.0
    score_unique = np.unique(scores).size

    return {
        f"long_ndcg@{top_size}": long_ndcg,
        f"short_ndcg@{bottom_size}": short_ndcg,
        f"macro_tail_ndcg@{top_size}_{bottom_size}": 0.5 * (long_ndcg + short_ndcg) if bottom_size else long_ndcg,
        f"long_map@{top_size}": long_map,
        f"short_map@{bottom_size}": short_map,
        f"macro_tail_map@{top_size}_{bottom_size}": 0.5 * (long_map + short_map) if bottom_size else long_map,
        f"long_mrr@{top_size}": long_mrr,
        f"short_mrr@{bottom_size}": short_mrr,
        f"macro_tail_mrr@{top_size}_{bottom_size}": 0.5 * (long_mrr + short_mrr) if bottom_size else long_mrr,
        f"long_err@{top_size}": long_err,
        f"short_err@{bottom_size}": short_err,
        f"macro_tail_err@{top_size}_{bottom_size}": 0.5 * (long_err + short_err) if bottom_size else long_err,
        "spearman": spearman,
        "kendall_tau_b": kendall,
        "pairwise_accuracy": pair_acc,
        "pairwise_denominator": float(pair_den),
        "top_precision": top_overlap / top_size,
        "top_recall": top_overlap / top_size,
        "top_overlap": float(top_overlap),
        "bottom_precision": bottom_overlap / bottom_size if bottom_size else 0.0,
        "bottom_recall": bottom_overlap / bottom_size if bottom_size else 0.0,
        "bottom_overlap": float(bottom_overlap),
        "exact_top_hit": float(pred_top == act_top),
        "exact_bottom_hit": float(pred_bottom == act_bottom),
        "both_tails_exact": float((pred_top == act_top) and (not bottom_size or pred_bottom == act_bottom)),
        "top_return": top_return,
        "short_bottom_return": -bottom_return,
        "long_short_spread": top_return - bottom_return,
        "expected_return": top_return - bottom_return if bottom_size else top_return,
        "top_oracle_regret": oracle_top_return - top_return,
        "bottom_oracle_regret": (-oracle_bottom_return) - (-bottom_return),
        "score_dispersion": float(np.std(scores)),
        "score_unique_count": float(score_unique),
        "score_tie_rate": float(1.0 - (score_unique / len(scores))),
    }


def evaluate_rankings(
    frame: pd.DataFrame,
    score_column: str = "score",
    group_column: str = "signal_datetime",
    top_size: int = 2,
    bottom_size: int = 2,
) -> dict[str, object]:
    per_group: list[dict[str, float | str]] = []
    for date, group in frame.groupby(group_column, sort=True):
        if len(group) <= top_size + bottom_size:
            continue
        row = evaluate_group(group, score_column=score_column, top_size=top_size, bottom_size=bottom_size)
        row[group_column] = str(date)
        per_group.append(row)
    metric_frame = pd.DataFrame(per_group)
    summary = {
        key: float(metric_frame[key].mean())
        for key in metric_frame.columns
        if key != group_column
    }
    return {"summary": summary, "per_group": per_group}


def score_controls(frame: pd.DataFrame, seed: int = 7) -> dict[str, pd.Series]:
    rng = np.random.default_rng(seed)
    oracle = frame["forward_return"].copy()
    reversed_oracle = -frame["forward_return"]
    random = pd.Series(rng.normal(size=len(frame)), index=frame.index, name="random")
    constant = pd.Series(0.0, index=frame.index, name="constant")
    controls = {
        "oracle": oracle,
        "reversed_oracle": reversed_oracle,
        "seeded_random": random,
        "constant": constant,
    }
    for offset in range(1, 32):
        controls[f"seeded_random_{offset:02d}"] = pd.Series(
            np.random.default_rng(seed + offset).normal(size=len(frame)), index=frame.index
        )
    return controls
