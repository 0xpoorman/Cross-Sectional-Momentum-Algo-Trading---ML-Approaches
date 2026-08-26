"""Preprocessing helpers frozen for the Phase 1 reproduction.

Phase 1 standardizes each feature inside each signal-date cross-section. This
keeps the model from learning absolute price or volume scale across dates while
preserving the relative ordering information used by the ranker.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_zscore(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Z-score each feature independently within every decision timestamp.

    The operation is intentionally cross-sectional rather than global: for a
    given date, each asset is compared only with the other assets available on
    that date. A zero standard deviation is replaced with a neutral zero value,
    and any remaining non-finite value fails closed because it would invalidate
    the ranking experiment.
    """
    result = frame.copy()
    grouped = result.groupby("signal_datetime", sort=False)[feature_names]
    means = grouped.transform("mean")
    standard_deviations = grouped.transform(lambda column: column.std(ddof=0))
    result[feature_names] = (
        (result[feature_names] - means)
        / standard_deviations.replace(0.0, np.nan)
    ).fillna(0.0)
    if not np.isfinite(result[feature_names].to_numpy()).all():
        raise ValueError("Phase 1 feature matrix contains non-finite values")
    return result
