from __future__ import annotations

import unittest

import pandas as pd

from src.train_jm_xgboost_compare import (
    DatasetConfig,
    add_features,
    artifact_prefix,
    evaluate_forecasts,
    normalize_market_frame,
    prediction_column,
    purged_train_end,
)


class TrainJmXgboostCompareTests(unittest.TestCase):
    def test_normalize_market_frame_maps_hold_to_open_interest(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2024-01-02"],
                "open": [1800.0],
                "high": [1820.0],
                "low": [1780.0],
                "close": [1810.0],
                "volume": [10000],
                "hold": [9000],
                "settle": [1808.0],
            }
        )

        normalized = normalize_market_frame(frame, "jm_main")

        self.assertIn("open_interest", normalized.columns)
        self.assertEqual(float(normalized.loc[0, "open_interest"]), 9000.0)
        self.assertEqual(normalized.loc[0, "dataset"], "jm_main")

    def test_add_features_keeps_weighted_roll_features_when_present(self) -> None:
        rows = []
        for i in range(90):
            rows.append(
                {
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                    "open": 100 + i,
                    "high": 102 + i,
                    "low": 99 + i,
                    "close": 101 + i,
                    "volume": 1000 + i * 10,
                    "open_interest": 2000 + i * 5,
                    "settle": 100.5 + i,
                    "dominant_volume_share": 0.55,
                    "dominant_oi_share": 0.60,
                    "volume_roll_pressure": 0.7,
                    "oi_roll_pressure": 0.5,
                    "dominant_contract_changed": 1 if i == 45 else 0,
                    "dominant_volume_contract_changed": 1 if i == 46 else 0,
                    "active_contracts": 5,
                    "oi_weighted_close": 100.8 + i,
                }
            )
        config = DatasetConfig(
            name="weighted",
            label="Weighted",
            path="unused.csv",
            output_prefix="jm_weighted",
        )

        _, model_data, feature_cols = add_features(pd.DataFrame(rows), config)

        self.assertIn("dominant_volume_share", feature_cols)
        self.assertIn("oi_volume_weighted_close_gap_pct", feature_cols)
        self.assertGreater(len(model_data), 0)

    def test_add_features_builds_configurable_ten_day_target(self) -> None:
        rows = []
        for i in range(90):
            rows.append(
                {
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                    "open": 100 + i,
                    "high": 102 + i,
                    "low": 99 + i,
                    "close": 101 + i,
                    "volume": 1000 + i * 10,
                    "open_interest": 2000 + i * 5,
                    "settle": 100.5 + i,
                }
            )
        config = DatasetConfig(
            name="main",
            label="Main",
            path="unused.csv",
            output_prefix="jm_main",
        )

        feature_data, model_data, _ = add_features(pd.DataFrame(rows), config, horizon=10)

        self.assertIn("future_10d_return_pct", feature_data.columns)
        self.assertIn("future_10d_up", feature_data.columns)
        self.assertNotIn("future_5d_return_pct", feature_data.columns)
        self.assertGreater(len(model_data), 0)

    def test_horizon_names_are_dynamic_without_breaking_five_day_artifacts(self) -> None:
        config = DatasetConfig(
            name="main",
            label="Main",
            path="unused.csv",
            output_prefix="jm_main",
        )

        self.assertEqual(prediction_column(5), "pred_future_5d_return_pct")
        self.assertEqual(prediction_column(10), "pred_future_10d_return_pct")
        self.assertEqual(artifact_prefix(config, 5), "jm_main")
        self.assertEqual(artifact_prefix(config, 10), "jm_main_10d")

    def test_purged_train_end_leaves_horizon_rows_before_test(self) -> None:
        self.assertEqual(purged_train_end(test_start=500, horizon=10), 490)

    def test_evaluate_forecasts_separates_bullish_and_bearish_outcomes(self) -> None:
        predictions = pd.DataFrame(
            {
                "pred_future_5d_return_pct": [1.0, 2.0, -1.0, -2.0],
                "future_5d_return_pct": [2.0, -1.0, -3.0, 1.0],
            }
        )

        quality = evaluate_forecasts(predictions, horizon=5)
        bullish = quality.loc[quality["forecast"] == "bullish"].iloc[0]
        bearish = quality.loc[quality["forecast"] == "bearish"].iloc[0]

        self.assertEqual(int(bullish["samples"]), 2)
        self.assertAlmostEqual(float(bullish["avg_realized_return_pct"]), 0.5)
        self.assertAlmostEqual(float(bullish["direction_accuracy"]), 0.5)
        self.assertAlmostEqual(float(bearish["avg_realized_return_pct"]), -1.0)
        self.assertAlmostEqual(float(bearish["direction_accuracy"]), 0.5)


if __name__ == "__main__":
    unittest.main()
