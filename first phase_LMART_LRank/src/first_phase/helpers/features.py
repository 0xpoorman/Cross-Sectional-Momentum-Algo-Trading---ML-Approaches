from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from .data import load_market_frame
from .splits import chronological_split

FEATURES = [
    "ret_1",
    "ret_5",
    "ret_20",
    "volatility_20",
    "range_1",
    "atr_14",
    "close_location",
    "volume_z_20",
    "volume_change_1",
    "vwap_dev_20",
    "poh_baz_vol_normalized_macd",
]


def _baz_response(signal: pd.Series) -> pd.Series:
    """Bounded response used by Baz et al. and reproduced by Poh et al."""
    return signal * np.exp(-(signal**2) / 4.0) / 0.89


def poh_baz_vol_normalized_macd(price: pd.Series) -> pd.Series:
    """Published Poh/Baz composite using 8/24, 16/48 and 32/96 horizons."""
    price_volatility = price.rolling(63, min_periods=63).std(ddof=0).replace(0.0, np.nan)
    responses = []
    for fast, slow in ((8, 24), (16, 48), (32, 96)):
        fast_ema = price.ewm(alpha=1.0 / fast, adjust=False, min_periods=fast).mean()
        slow_ema = price.ewm(alpha=1.0 / slow, adjust=False, min_periods=slow).mean()
        volatility_scaled_macd = (fast_ema - slow_ema) / price_volatility
        long_run_scale = (
            volatility_scaled_macd.rolling(252, min_periods=252)
            .std(ddof=0)
            .replace(0.0, np.nan)
        )
        responses.append(_baz_response(volatility_scaled_macd / long_run_scale))
    return pd.concat(responses, axis=1).sum(axis=1, min_count=len(responses))


def load_frame(path):
    frame, _ = load_market_frame(path)
    return frame.set_index(["timestamp", "symbol"]).sort_index()


def select_ranked_universe(indexed: pd.DataFrame) -> pd.DataFrame:
    """Use every symbol in the ranked dataset except the SPY benchmark."""
    symbols = indexed.index.get_level_values("symbol")
    return indexed.loc[symbols != "SPY"].copy()


def select_benchmark(indexed: pd.DataFrame) -> pd.DataFrame:
    return indexed.loc[indexed.index.get_level_values("symbol") == "SPY"].copy()


def engineer_features(indexed: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = indexed.reset_index().sort_values(["symbol", "timestamp"]).copy()
    grouped = df.groupby("symbol", sort=False, group_keys=False)
    previous_close = grouped["close"].shift(1)

    df["ret_1"] = grouped["close"].pct_change(1)
    df["ret_5"] = grouped["close"].pct_change(5)
    df["ret_20"] = grouped["close"].pct_change(20)
    df["volatility_20"] = df["ret_1"].groupby(df["symbol"]).transform(lambda values: values.rolling(20, min_periods=20).std())
    df["range_1"] = (df["high"] - df["low"]) / df["close"]
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = true_range.groupby(df["symbol"]).transform(lambda values: values.rolling(14, min_periods=14).mean()) / df["close"]
    intraday_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["close_location"] = (df["close"] - df["low"]) / intraday_range
    volume_mean = grouped["volume"].transform(lambda values: values.rolling(20, min_periods=20).mean())
    volume_std = grouped["volume"].transform(lambda values: values.rolling(20, min_periods=20).std())
    df["volume_z_20"] = (df["volume"] - volume_mean) / volume_std.replace(0, np.nan)
    df["volume_change_1"] = grouped["volume"].pct_change()
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    rolling_pv = (typical_price * df["volume"]).groupby(df["symbol"]).transform(lambda values: values.rolling(20, min_periods=20).sum())
    rolling_volume = grouped["volume"].transform(lambda values: values.rolling(20, min_periods=20).sum())
    df["vwap_dev_20"] = df["close"] / (rolling_pv / rolling_volume.replace(0, np.nan)) - 1.0
    df["poh_baz_vol_normalized_macd"] = grouped["close"].transform(
        poh_baz_vol_normalized_macd
    )

    df["signal_datetime"] = df["timestamp"]
    df["entry_datetime"] = grouped["timestamp"].shift(-1)
    df["exit_datetime"] = grouped["timestamp"].shift(-(horizon + 1))
    df["entry_price"] = grouped["open"].shift(-1)
    df["exit_price"] = grouped["open"].shift(-(horizon + 1))
    df["forward_return"] = df["exit_price"] / df["entry_price"] - 1.0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=FEATURES + ["signal_datetime", "entry_datetime", "exit_datetime", "entry_price", "exit_price", "forward_return"], inplace=True)
    return df.sort_values(["timestamp", "symbol"]).set_index(["timestamp", "symbol"])


def filter_complete_dates(df: pd.DataFrame, min_symbols: int) -> pd.DataFrame:
    counts = df.groupby(level="timestamp").size()
    usable = counts[counts >= min_symbols].index
    return df.loc[df.index.get_level_values("timestamp").isin(usable)].copy()
