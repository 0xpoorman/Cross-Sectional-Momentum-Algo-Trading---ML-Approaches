from __future__ import annotations

import pandas as pd


def chronological_split(
    frame: pd.DataFrame,
    horizon: int,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.Index(sorted(frame["signal_datetime"].dropna().unique()))
    train_end = int(len(dates) * train_fraction)
    validation_end = int(len(dates) * (train_fraction + validation_fraction))
    purge = max(1, horizon + 1)
    if train_end <= purge or validation_end - train_end <= purge:
        raise ValueError("Not enough timestamps for the requested split with purge.")
    train_dates = dates[: train_end - purge]
    validation_dates = dates[train_end : validation_end - purge]
    test_dates = dates[validation_end:]
    train = frame.loc[frame["signal_datetime"].isin(train_dates)].copy()
    validation = frame.loc[frame["signal_datetime"].isin(validation_dates)].copy()
    test = frame.loc[frame["signal_datetime"].isin(test_dates)].copy()
    _assert_purged(train, validation)
    _assert_purged(validation, test)
    return train, validation, test


def _assert_purged(left: pd.DataFrame, right: pd.DataFrame) -> None:
    if left.empty or right.empty:
        return
    left_max = left["exit_datetime"].dropna().max()
    right_min = right["signal_datetime"].dropna().min()
    if left_max >= right_min:
        raise AssertionError(
            f"Label leakage across split boundary: left exits at {left_max}, right begins at {right_min}"
        )
