from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.export_latest_signals import build_payload, run_models
from src.product_config import get_product


class ExportLatestSignalsTests(unittest.TestCase):
    def test_build_payload_includes_forecasts_and_matching_quality(self) -> None:
        results = {
            "JM": {
                5: self.result(5, 1.25, [1.0, -1.0], [2.0, -3.0]),
                10: self.result(10, -0.75, [-1.0, -2.0], [-2.0, 1.0]),
            }
        }

        payload = build_payload(results, generated_at="2026-06-25T03:10:00+09:00")
        product = payload["products"]["JM"]
        five_day = product["forecasts"]["5"]
        ten_day = product["forecasts"]["10"]

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(product["source_date"], "2026-06-24")
        self.assertEqual(product["name_zh"], "焦煤")
        self.assertEqual(five_day["direction"], "bullish")
        self.assertAlmostEqual(five_day["predicted_return_pct"], 1.25)
        self.assertEqual(five_day["quality"]["samples"], 1)
        self.assertAlmostEqual(five_day["quality"]["direction_accuracy"], 1.0)
        self.assertEqual(ten_day["direction"], "bearish")
        self.assertEqual(ten_day["quality"]["samples"], 2)
        self.assertAlmostEqual(ten_day["quality"]["direction_accuracy"], 0.5)

    @patch("src.export_latest_signals.run_dataset")
    @patch("src.export_latest_signals.product_configs")
    @patch("src.export_latest_signals.ensure_dirs")
    def test_run_models_creates_output_directories_on_clean_runner(
        self,
        ensure_dirs,
        product_configs,
        run_dataset,
    ) -> None:
        product_configs.return_value = [SimpleNamespace()]
        run_dataset.return_value = SimpleNamespace()

        with patch("src.export_latest_signals.PRODUCTS", {"JM": get_product("JM")}):
            run_models()

        ensure_dirs.assert_called_once_with()

    @staticmethod
    def result(
        horizon: int,
        latest_prediction: float,
        historical_predictions: list[float],
        realized_returns: list[float],
    ) -> SimpleNamespace:
        pred_col = f"pred_future_{horizon}d_return_pct"
        target_col = f"future_{horizon}d_return_pct"
        signal = 1 if latest_prediction > 0 else -1
        return SimpleNamespace(
            config=SimpleNamespace(label=f"{get_product('JM').code}0 主力连续"),
            latest_signal=pd.DataFrame(
                [
                    {
                        "date": pd.Timestamp("2026-06-24"),
                        "close": 1245.5,
                        pred_col: latest_prediction,
                        "signal": signal,
                    }
                ]
            ),
            predictions=pd.DataFrame(
                {
                    pred_col: historical_predictions,
                    target_col: realized_returns,
                }
            ),
        )


if __name__ == "__main__":
    unittest.main()
