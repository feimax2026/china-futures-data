import unittest

import numpy as np
import pandas as pd

from src.basis_history import basis_statistics
from src.model_selection import neutral_threshold, select_model
from src.product_config import SENTINEL_PRODUCTS


class BasisModelTests(unittest.TestCase):
    def rows(self):
        return [{"date": d.date().isoformat(), "contract": "CU2612", "basis": i,
                 "basis_rate_pct": float(i)} for i, d in enumerate(pd.bdate_range("2026-01-01", periods=81))]

    def test_real_prior_standard_deviation_and_percentile(self):
        rows = self.rows()
        result = basis_statistics(rows, rows[-1]["date"], "CU2612")
        self.assertEqual(result["samples"], 80)
        self.assertAlmostEqual(result["basis_z"], (80 - np.mean(range(80))) / np.std(range(80), ddof=1))
        self.assertEqual(result["basis_percentile"], 100)

    def test_stale_mismatched_short_and_constant_samples_not_faked(self):
        rows = self.rows()
        self.assertEqual(basis_statistics(rows[:20], rows[19]["date"])["status"], "insufficient_history")
        self.assertEqual(basis_statistics(rows, "2027-01-01")["status"], "stale")
        self.assertEqual(basis_statistics(rows, rows[-1]["date"], "CU2701")["status"], "contract_mismatch")
        for row in rows: row["basis_rate_pct"] = 1
        result = basis_statistics(rows, rows[-1]["date"])
        self.assertIsNone(result["basis_z"])
        self.assertEqual(result["basis_percentile"], 50)

    def test_horizon_and_volatility_scale_neutral_zone(self):
        self.assertGreater(float(neutral_threshold(2, 10)), float(neutral_threshold(1, 5)))
        self.assertEqual(float(neutral_threshold(0, 5)), 0.05)
        self.assertEqual(set(SENTINEL_PRODUCTS), {"JM", "I", "SM", "SF", "AU", "AG", "CU"})

    def test_selection_refits_only_provided_training_data(self):
        dates = pd.bdate_range("2020-01-01", periods=510)
        train = pd.DataFrame({"date": dates, "label_end_date": dates + pd.Timedelta(days=10),
                              "x": np.sin(np.arange(510)), "y": np.cos(np.arange(510))})
        model, metadata = select_model(train, ["x"], "y", "SF")
        self.assertEqual(metadata["selection"], "past_126_rows_purged_mse")
        self.assertIn(metadata["max_depth"], (2, 3))
        self.assertTrue(np.isfinite(model.predict(train[["x"]])).all())
