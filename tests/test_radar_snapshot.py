import unittest

import numpy as np
import pandas as pd

from src.radar_snapshot import contract_symbol, indicators, select_contract


class RadarSnapshotTests(unittest.TestCase):
    def bars(self):
        dates = pd.bdate_range("2026-01-05", periods=100).strftime("%Y-%m-%d").tolist()
        close = np.arange(100, 200, dtype=float)
        frame = pd.DataFrame({"date": dates, "open": close, "high": close + 2,
                              "low": close - 2, "close": close, "volume": 100., "hold": 1000.})
        return frame, dates

    def test_real_symbols_and_czce_decade(self):
        self.assertEqual(contract_symbol("SM701", "SM", "2026-09-30"), "SM2701")
        self.assertEqual(contract_symbol("SF001", "SF", "2029-12-10"), "SF3001")
        for value in ("JM0", "JM2613", "JM2601"):
            with self.assertRaises(ValueError):
                contract_symbol(value, "JM", "2026-09-30")

    def test_selection_filters_continuous_and_wrong_date(self):
        quotes = pd.DataFrame({"symbol": ["JM0", "JM2701", "JM2611"],
                               "tradedate": ["2026-09-30"] * 3, "position": [9999, 1000, 200],
                               "volume": [100, 100, 100], "close": [100, 100, 100]})
        self.assertEqual(select_contract(quotes, "JM", "2026-09-30")[0], "JM2701")
        with self.assertRaises(ValueError):
            select_contract(quotes, "JM", "2026-10-08")

    def test_prior_windows_exclude_current_bar(self):
        frame, dates = self.bars()
        frame.loc[99, "volume"] = 300
        result = indicators(frame, dates[-1], dates)
        self.assertEqual(result["volume_ratio_prior20"], 3)
        self.assertEqual(result["prior20_high"], 200)
        self.assertEqual(result["previous_high"], 200)
        self.assertEqual(result["bandwidth_rank"]["samples"], 81)

    def test_gap_and_bad_recent_ohlc_fail_closed(self):
        frame, dates = self.bars()
        with self.assertRaises(ValueError):
            indicators(frame.drop(90), dates[-1], dates)
        frame.loc[95, "low"] = 999
        with self.assertRaises(ValueError):
            indicators(frame, dates[-1], dates)


if __name__ == "__main__":
    unittest.main()
