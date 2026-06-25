from __future__ import annotations

import math
import unittest

import pandas as pd

from src.build_jm_weighted_index import build_index


class BuildJmWeightedIndexTests(unittest.TestCase):
    def test_build_index_uses_previous_open_interest_weighted_returns(self) -> None:
        frame = pd.DataFrame(
            [
                self.contract_row("2024-01-02", "JM2401", 100.0, volume=10, hold=3),
                self.contract_row("2024-01-02", "JM2405", 200.0, volume=10, hold=1),
                self.contract_row("2024-01-03", "JM2401", 110.0, volume=10, hold=3),
                self.contract_row("2024-01-03", "JM2405", 210.0, volume=10, hold=1),
            ]
        )

        index = build_index(frame)
        second = index.iloc[1]
        expected_return = ((3 * math.log(110.0 / 100.0)) + (1 * math.log(210.0 / 200.0))) / 4 * 100
        expected_close = index.iloc[0]["close"] * math.exp(expected_return / 100)

        self.assertIn("oi_weighted_return_pct", index.columns)
        self.assertAlmostEqual(second["oi_weighted_return_pct"], expected_return)
        self.assertAlmostEqual(second["close"], expected_close)
        self.assertIn("volume_weighted_close", index.columns)

    @staticmethod
    def contract_row(date: str, contract: str, close: float, volume: int, hold: int) -> dict[str, object]:
        return {
            "date": pd.Timestamp(date),
            "contract": contract,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "settle": close,
            "volume": volume,
            "hold": hold,
        }


if __name__ == "__main__":
    unittest.main()
