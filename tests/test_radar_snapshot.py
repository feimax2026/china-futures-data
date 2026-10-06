import unittest

import numpy as np
import pandas as pd

from src.radar_snapshot import contract_symbol, discover_contracts, indicators, select_contract


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

    def test_discovery_filters_continuous_and_future_date(self):
        quotes = pd.DataFrame({"symbol": ["JM0", "JM2701", "JM2611"],
                               "tradedate": ["2026-09-30"] * 3, "position": [9999, 1000, 200],
                               "volume": [100, 100, 100], "close": [100, 100, 100]})
        self.assertEqual(discover_contracts(quotes, "JM", "2026-09-30", "2026-10-08"), ["JM2611", "JM2701"])
        with self.assertRaises(ValueError):
            discover_contracts(quotes, "JM", "2026-10-08", "2026-10-09")

    def test_night_quotes_discover_but_do_not_rank_completed_day(self):
        quotes = pd.DataFrame({"symbol": ["JM2701", "JM2611"], "tradedate": ["2026-10-09"] * 2,
                               "position": [2000, 4000]})
        symbols = discover_contracts(quotes, "JM", "2026-10-08", "2026-10-09")
        histories = {s: pd.DataFrame({"date": ["2026-10-08", "2026-10-09"],
            "hold": [3000 if s == "JM2701" else 1000, 2000 if s == "JM2701" else 4000],
            "close": [100, 110], "volume": [100, 100]}) for s in symbols}
        self.assertEqual(select_contract(histories, "2026-10-08"), "JM2701")

    def test_stale_selection_history_fails_closed(self):
        history = pd.DataFrame({"date": ["2026-10-07"], "hold": [2000], "close": [100], "volume": [100]})
        with self.assertRaises(ValueError):
            select_contract({"JM2701": history}, "2026-10-08")

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
