from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recreate_first_trials import TrialContract, fixed_horizon_backtest
from first_phase.helpers.model import _two_tail_loss
import torch


class FixedHorizonBacktestTests(unittest.TestCase):
    def test_uses_non_overlapping_three_bar_cohorts(self):
        rows = []
        for day in range(6):
            for symbol, value in (("A", .03), ("B", .02), ("C", -.01), ("D", -.02), ("E", 0.0)):
                rows.append({
                    "signal_datetime": pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(day)),
                    "entry_datetime": pd.Timestamp("2024-01-02") + pd.Timedelta(days=int(day)),
                    "exit_datetime": pd.Timestamp("2024-01-05") + pd.Timedelta(days=int(day)),
                    "symbol": symbol, "forward_return": value,
                })
        frame = pd.DataFrame(rows)
        scores = np.tile([5, 4, 3, 2, 1], 6)
        result, metrics = fixed_horizon_backtest(
            frame, scores, TrialContract(dataset="synthetic", min_symbols=5)
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(metrics["annualization_periods"], 84.0)
        self.assertTrue((result["net_return"] > 0).all())

    def test_phase_one_lambdarank_pairs_are_active_without_elite_weighting(self):
        """The historical path must not collapse to a zero-loss scorer."""
        scores = torch.arange(11, dtype=torch.float32)
        returns = torch.arange(11, dtype=torch.float32)
        groups = torch.zeros(11, dtype=torch.long)
        loss, pair_count, weight = _two_tail_loss(
            scores, returns, groups, 2, 2, 1.0, elite_quantile=None, elite_gain=0.0
        )
        self.assertGreater(float(loss), 0.0)
        self.assertGreater(pair_count, 0)
        self.assertGreater(weight, 0.0)


if __name__ == "__main__":
    unittest.main()
