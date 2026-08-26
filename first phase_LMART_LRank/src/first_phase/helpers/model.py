from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .metrics import evaluate_rankings

try:  # optional at import time, mandatory for neural training time
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None


def require_neural_dependencies() -> None:
    if torch is None or nn is None:
        raise RuntimeError("LambdaRank neural training requires torch to be installed")


def build_activation(name: str, negative_slope: float):
    require_neural_dependencies()
    key = str(name).strip().lower().replace("_", "")
    factories = {
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "prelu": nn.PReLU,
        "elu": nn.ELU,
        "sigmoid": nn.Sigmoid,
        "leakyrelu": lambda: nn.LeakyReLU(negative_slope),
    }
    try:
        return factories[key]()
    except KeyError as error:
        raise ValueError(f"Unsupported activation: {name}") from error


def build_normalization(name: str, dim: int):
    require_neural_dependencies()
    key = str(name).strip().lower().replace("_", "")
    factories = {
        "none": nn.Identity,
        "layer": lambda: nn.LayerNorm(dim),
        "layernorm": lambda: nn.LayerNorm(dim),
        "batch": lambda: nn.BatchNorm1d(dim),
        "batchnorm": lambda: nn.BatchNorm1d(dim),
        "rms": lambda: nn.RMSNorm(dim),
        "rmsnorm": lambda: nn.RMSNorm(dim),
        "unit": lambda: UnitNorm(),
        "unitnorm": lambda: UnitNorm(),
    }
    try:
        return factories[key]()
    except KeyError as error:
        raise ValueError(f"Unsupported normalization: {name}") from error


class UnitNorm(nn.Module if nn is not None else object):
    """Normalize each hidden vector to unit L2 length."""

    def forward(self, values):
        return F.normalize(values, p=2.0, dim=-1)


def resolve_device(requested: str):
    """Select an available accelerator without changing numerical precision."""
    require_neural_dependencies()
    requested = str(requested).strip().lower()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class RankResult:
    model: Any
    scaler: StandardScaler
    feature_names: tuple[str, ...]
    history: list[dict[str, float]]
    diagnostics: dict[str, Any]
    model_family: str
    device: str = "cpu"

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        values = self.scaler.transform(frame[list(self.feature_names)].to_numpy(np.float32))
        if self.model_family == "ridge":
            scores = self.model.predict(values)
        else:
            require_neural_dependencies()
            self.model.eval()
            with torch.inference_mode():
                inputs = torch.from_numpy(np.asarray(values, dtype=np.float32)).to(self.device)
                scores = self.model(inputs).cpu().numpy()
        return pd.Series(scores, index=frame.index, name="score")


class IdentityScaler:
    """Scikit-learn-like adapter for already standardized cross-sections."""

    def fit(self, values):
        self.n_features_in_ = int(np.asarray(values).shape[1])
        return self

    def transform(self, values):
        return np.asarray(values, dtype=np.float32)


class TorchRanker(nn.Module if nn is not None else object):
    def __init__(self, input_dim: int, config) -> None:
        require_neural_dependencies()
        super().__init__()
        if not 1 <= config.hidden_layers <= 4:
            raise ValueError("hidden_layers must be between 1 and 4")
        layers = []
        previous = input_dim
        for index in range(config.hidden_layers):
            width = max(4, config.hidden_dim // (2**index))
            layers.extend([
                nn.Linear(previous, width),
                build_normalization(config.normalization, width),
                build_activation(config.activation, config.leaky_relu_negative_slope),
            ])
            if index + 1 < config.hidden_layers and config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _one_sided_loss(
    scores,
    returns,
    group_ids,
    cutoff: int,
    relevance_top: int,
    relevance_bottom: int,
    sigma: float,
    *,
    elite_quantile: float | None = None,
    elite_gain: float = 15.0,
):
    losses = []
    active_pairs = 0
    total_weight = scores.new_tensor(0.0)
    group_sizes = torch.unique_consecutive(group_ids, return_counts=True)[1].tolist()
    grouped_scores = scores.split(group_sizes)
    grouped_returns = returns.split(group_sizes)
    for group_scores, group_returns in zip(grouped_scores, grouped_returns):
        n = int(group_scores.numel())
        if n < 2:
            continue
        order = torch.argsort(group_returns)
        relevance = torch.ones(n, device=scores.device, dtype=scores.dtype)
        top = min(relevance_top, max(1, n - 1))
        bottom = min(relevance_bottom, max(1, n - top))
        relevance[order[:bottom]], relevance[order[-top:]] = 0.0, 2.0
        gains = torch.exp2(relevance) - 1.0
        if elite_quantile is not None:
            percentiles = torch.empty_like(group_returns)
            percentiles[order] = (
                torch.arange(1, n + 1, device=scores.device, dtype=scores.dtype) / n
            )
            strength = ((percentiles - elite_quantile) / (1.0 - elite_quantile)).clamp(0.0, 1.0)
            strength = strength * (group_returns > 0.0)
            gains = gains + strength * (elite_gain - gains).clamp_min(0.0)
        positions = torch.arange(n, device=scores.device, dtype=scores.dtype)
        discounts = 1.0 / torch.log2(positions + 2.0)
        idcg = torch.sum(torch.sort(gains, descending=True).values[: min(cutoff, n)] * discounts[: min(cutoff, n)])
        predicted_order = torch.argsort(group_scores.detach(), descending=True)
        predicted_position = torch.empty(n, device=scores.device, dtype=torch.long)
        predicted_position[predicted_order] = torch.arange(n, device=scores.device)
        pi, pj = predicted_position[:, None], predicted_position[None, :]
        di = torch.where(pi < cutoff, discounts[pi], torch.zeros_like(discounts[pi]))
        dj = torch.where(pj < cutoff, discounts[pj], torch.zeros_like(discounts[pj]))
        weights = torch.abs((gains[:, None] - gains[None, :]) * (di - dj)) / idcg.clamp_min(1e-12)
        pair_mask = (group_returns[:, None] > group_returns[None, :]) & (weights > 0)
        if pair_mask.any():
            pair_loss = F.softplus(-sigma * (group_scores[:, None] - group_scores[None, :]))
            w = weights[pair_mask].detach()
            losses.append(torch.sum(pair_loss[pair_mask] * w))
            total_weight = total_weight + w.sum()
            active_pairs += int(pair_mask.sum().item())
    if not losses:
        return scores.sum() * 0.0, 0, 0.0
    return torch.stack(losses).sum() / total_weight.clamp_min(1e-12), active_pairs, float(total_weight.detach())


def _two_tail_loss(
    scores,
    returns,
    group_ids,
    top_size: int,
    bottom_size: int,
    sigma: float,
    *,
    elite_quantile: float | None = None,
    elite_gain: float = 15.0,
):
    long_loss, long_pairs, long_weight = _one_sided_loss(
        scores,
        returns,
        group_ids,
        top_size,
        top_size,
        bottom_size,
        sigma,
        elite_quantile=elite_quantile,
        elite_gain=elite_gain,
    )
    short_loss, short_pairs, short_weight = _one_sided_loss(
        -scores,
        -returns,
        group_ids,
        bottom_size,
        bottom_size,
        top_size,
        sigma,
        elite_quantile=elite_quantile,
        elite_gain=elite_gain,
    )
    return 0.5 * (long_loss + short_loss), long_pairs + short_pairs, long_weight + short_weight


def _tensor_data(frame: pd.DataFrame, feature_names: list[str], scaler, device):
    require_neural_dependencies()
    values = np.asarray(
        scaler.transform(frame[feature_names].to_numpy(np.float32)), dtype=np.float32
    )
    codes = pd.factorize(frame["signal_datetime"], sort=True)[0]
    if len(codes) > 1 and np.any(codes[1:] < codes[:-1]):
        raise ValueError("Ranking rows must be contiguous and sorted by signal_datetime")
    return (
        torch.from_numpy(values).to(device),
        torch.from_numpy(frame["forward_return"].to_numpy(np.float32)).to(device),
        torch.from_numpy(codes).to(device),
    )


def build_optimizers(model, config) -> list:
    """Build one optimizer, or Muon plus AdamW for unsupported parameters."""
    key = str(config.optimizer).strip().lower()
    common = {"lr": config.learning_rate, "weight_decay": config.weight_decay}
    factories = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "nadam": torch.optim.NAdam,
        "radam": torch.optim.RAdam,
    }
    if key in factories:
        return [factories[key](model.parameters(), **common)]
    if key == "sgd":
        return [torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True, **common)]
    if key != "muon":
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

    muon = [parameter for parameter in model.parameters() if parameter.ndim == 2 and min(parameter.shape) > 1]
    muon_ids = {id(parameter) for parameter in muon}
    other = [parameter for parameter in model.parameters() if id(parameter) not in muon_ids]
    if not muon:
        raise ValueError("Muon requires at least one two-dimensional hidden-layer parameter")
    return [
        torch.optim.Muon(muon, adjust_lr_fn="match_rms_adamw", **common),
        torch.optim.AdamW(other, **common),
    ]


def fit_ridge_baseline(train: pd.DataFrame, validation: pd.DataFrame, feature_names: list[str], config) -> RankResult:
    scaler = (
        IdentityScaler().fit(train[feature_names].to_numpy(np.float32))
        if config.preprocessing == "cross_sectional_zscore"
        else StandardScaler().fit(train[feature_names].to_numpy(np.float32))
    )
    x_train = scaler.transform(train[feature_names].to_numpy(np.float32))
    y_train = _cross_sectional_standardize_target(train)
    x_validation = scaler.transform(validation[feature_names].to_numpy(np.float32))
    y_validation = _cross_sectional_standardize_target(validation)
    model = Ridge(alpha=config.ridge_alpha, fit_intercept=config.ridge_fit_intercept)
    model.fit(x_train, y_train)
    validation_mse = float(np.mean((model.predict(x_validation) - y_validation) ** 2))
    return RankResult(
        model=model,
        scaler=scaler,
        feature_names=tuple(feature_names),
        history=[{"epoch": 1.0, "validation_mse": validation_mse}],
        diagnostics={"implementation_status": "trained_ridge_baseline", "validation_mse": validation_mse},
        model_family="ridge",
    )


def _cross_sectional_standardize_target(frame: pd.DataFrame) -> np.ndarray:
    grouped = frame.groupby("signal_datetime", sort=False)["forward_return"]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    values = ((frame["forward_return"] - mean) / std).fillna(0.0)
    return values.to_numpy(np.float32)


def fit_final_ranker(
    train_validation: pd.DataFrame,
    feature_names: list[str],
    config,
    *,
    top_size: int = 2,
    bottom_size: int = 2,
) -> RankResult:
    timestamps = sorted(train_validation["signal_datetime"].unique())
    split_at = max(1, int(len(timestamps) * (1.0 - config.final_refit_validation_fraction)))
    inner_train_dates = set(timestamps[:split_at])
    inner_valid_dates = set(timestamps[split_at:])
    inner_train = train_validation.loc[train_validation["signal_datetime"].isin(inner_train_dates)].copy()
    inner_valid = train_validation.loc[train_validation["signal_datetime"].isin(inner_valid_dates)].copy()
    if inner_valid.empty:
        inner_valid = inner_train.tail(max(1, len(inner_train) // 5)).copy()
        inner_train = inner_train.iloc[: max(1, len(inner_train) - len(inner_valid))].copy()
    return fit_ranker(
        inner_train,
        inner_valid,
        feature_names,
        config,
        top_size=top_size,
        bottom_size=bottom_size,
    )


def fit_neural_ranker(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_names: list[str],
    config,
    top_size: int = 2,
    bottom_size: int = 2,
) -> RankResult:
    require_neural_dependencies()
    torch.set_num_threads(config.torch_num_threads)
    try:
        torch.set_num_interop_threads(config.torch_num_threads)
    except RuntimeError:
        # Inter-op threads can only be set before parallel work starts.
        pass
    device = resolve_device(config.device)
    scaler = (
        IdentityScaler().fit(train[feature_names].to_numpy(np.float32))
        if config.preprocessing == "cross_sectional_zscore"
        else StandardScaler().fit(train[feature_names].to_numpy(np.float32))
    )
    train_x, train_y, train_groups = _tensor_data(train, feature_names, scaler, device)
    validation_x, validation_y, validation_groups = _tensor_data(
        validation, feature_names, scaler, device
    )
    torch.manual_seed(config.seed)
    if device.type == "cpu":
        torch.use_deterministic_algorithms(True)
    model = TorchRanker(len(feature_names), config).to(device)
    optimizers = build_optimizers(model, config)
    percentile = train.groupby("signal_datetime", sort=False)["forward_return"].rank(pct=True)
    elite_train_count = int(
        ((train["forward_return"] > 0.0) & (percentile > config.elite_positive_quantile)).sum()
    )
    history: list[dict[str, float]] = []
    best_state, best_validation, best_validation_loss, stale_epochs = (
        None,
        float("-inf"),
        float("inf"),
        0,
    )
    best_epoch = None
    last_pairs, last_weight, last_gradient_norm = 0, 0.0, 0.0
    loss_fn = _two_tail_loss if bottom_size else _one_sided_loss
    loss_args = (
        (top_size, bottom_size, config.sigma)
        if bottom_size
        else (top_size, top_size, 0, config.sigma)
    )
    for epoch in range(1, config.epochs + 1):
        model.train()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss, last_pairs, last_weight = loss_fn(
            model(train_x),
            train_y,
            train_groups,
            *loss_args,
            elite_quantile=config.elite_positive_quantile,
            elite_gain=config.elite_label_gain,
        )
        loss.backward()
        last_gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.max_gradient_norm if config.max_gradient_norm > 0.0 else float("inf"),
            ).detach()
        )
        for optimizer in optimizers:
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_scores = model(validation_x)
            validation_loss, _, _ = loss_fn(
                validation_scores,
                validation_y,
                validation_groups,
                *loss_args,
                elite_quantile=config.elite_positive_quantile,
                elite_gain=config.elite_label_gain,
            )
        validation_value = float(validation_loss.cpu())
        validation_scored = validation[["signal_datetime", "symbol", "forward_return"]].copy()
        validation_scored["score"] = validation_scores.cpu().numpy()
        validation_metrics = evaluate_rankings(
            validation_scored,
            top_size=top_size,
            bottom_size=bottom_size,
        )["summary"]
        metric_suffix = f"{top_size}_{bottom_size}"
        validation_ndcg = float(validation_metrics[f"macro_tail_ndcg@{metric_suffix}"])
        history.append(
            {
                "epoch": float(epoch),
                "train_lambdarank_loss": float(loss.detach().cpu()),
                "validation_lambdarank_loss": validation_value,
                "validation_macro_tail_ndcg_at_2": validation_ndcg,
                "validation_macro_tail_map_at_2": float(validation_metrics[f"macro_tail_map@{metric_suffix}"]),
                "validation_macro_tail_mrr_at_2": float(validation_metrics[f"macro_tail_mrr@{metric_suffix}"]),
                "validation_macro_tail_err_at_2": float(validation_metrics[f"macro_tail_err@{metric_suffix}"]),
                "validation_pairwise_accuracy": float(validation_metrics["pairwise_accuracy"]),
                "gradient_norm": last_gradient_norm,
            }
        )
        if epoch >= config.checkpoint_start_epoch:
            if validation_ndcg > best_validation + 1e-8:
                best_validation = validation_ndcg
                best_validation_loss = validation_value
                best_state = {
                    name: value.detach().clone() for name, value in model.state_dict().items()
                }
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
        if epoch >= config.min_epochs and stale_epochs >= config.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return RankResult(
        model=model,
        scaler=scaler,
        feature_names=tuple(feature_names),
        history=history,
        diagnostics={
            "implementation_status": "trained_two_tail_lambdarank" if bottom_size else "trained_long_only_lambdarank",
            "device": str(device),
            "pytorch_version": str(torch.__version__),
            "best_validation_macro_tail_ndcg_at_2": best_validation,
            "best_validation_loss": best_validation_loss,
            "selected_checkpoint_epoch": best_epoch,
            "early_stopping_metric": "validation macro-tail NDCG@2 (higher is better)",
            "checkpoint_start_epoch": config.checkpoint_start_epoch,
            "minimum_epochs": config.min_epochs,
            "patience": config.patience,
            "active_pair_count": last_pairs,
            "active_pair_weight": last_weight,
            "max_gradient_norm": config.max_gradient_norm,
            "last_unclipped_gradient_norm": last_gradient_norm,
            "elite_positive_quantile": config.elite_positive_quantile,
            "elite_weighting_policy": "within_date_percentile_linear_ramp",
            "elite_label_gain": config.elite_label_gain,
            "elite_train_count": elite_train_count,
        },
        model_family="neural",
        device=str(device),
    )


def fit_ranker(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_names: list[str],
    config,
    top_size: int = 2,
    bottom_size: int = 2,
) -> RankResult:
    if config.model_family == "ridge":
        return fit_ridge_baseline(train, validation, feature_names, config)
    if config.model_family != "neural":
        raise ValueError(f"Unsupported model_family: {config.model_family}")
    return fit_neural_ranker(
        train,
        validation,
        feature_names,
        config,
        top_size=top_size,
        bottom_size=bottom_size,
    )
