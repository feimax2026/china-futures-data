from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.contract_chain import normalize_chain
from src.data_quality import DataQualityError, validate_daily, validate_signal_payload
from src.import_power_data import asof_power, validate_power
from src.option_data import attach_underlyings, exercise_style, normalize_options, parse_contract, option_summary
from src.product_config import LEGACY_PRODUCTS, PRODUCTS
from src.research_models import week_anchor, frozen_model, frozen_predict
from src.research_storage import ResearchStore
from src.train_xgboost_compare import DatasetConfig, add_features, purged_training_rows
from src.feed_access import bounded_ak_frame
from src.research_pipeline import run


class ResearchPlatformTests(unittest.TestCase):
    def test_registry_and_legacy_coverage(self):
        self.assertEqual(len(PRODUCTS), 10)
        self.assertEqual(PRODUCTS["JM"].exchange, "DCE")
        self.assertEqual(PRODUCTS["SM"].exchange, "CZCE")
        self.assertEqual(set(LEGACY_PRODUCTS), {"JM", "I", "SM", "CU"})

    def test_source_process_is_bounded(self):
        import subprocess
        with patch("src.feed_access.subprocess.run", side_effect=subprocess.TimeoutExpired("feed", 120)):
            with self.assertRaisesRegex(RuntimeError, "exceeded 120"):
                bounded_ak_frame("main", "AU0")

    def daily(self):
        return pd.DataFrame({"date": ["2026-09-30"], "open": [100], "high": [102], "low": [99],
                             "close": [101], "settle": [100], "volume": [10], "hold": [20]})

    def test_rejects_stale_and_infinite_main(self):
        with self.assertRaises(DataQualityError):
            validate_daily(self.daily(), "2026-10-09")
        frame = self.daily()
        frame["close"] = np.inf
        with self.assertRaises(DataQualityError):
            validate_daily(frame)

    def test_signal_freshness_enforced(self):
        payload = {"expected_source_date": "2026-09-30", "products": {"JM": {"source_date": "2026-09-29", "forecasts": {}}}}
        with self.assertRaises(DataQualityError):
            validate_signal_payload(payload, {"JM"})

    def test_missing_historical_settle_keeps_original_calendar_without_filling(self):
        history = pd.concat([self.daily().assign(date="2026-09-29", settle=0), self.daily()], ignore_index=True)
        normalized = validate_daily(history, "2026-09-30", allow_missing_historical_settle=True)
        self.assertEqual(len(normalized), 2)
        self.assertTrue(pd.isna(normalized["settle"].iloc[0]))
        self.assertTrue(normalized["settle_missing"].iloc[0])
        with self.assertRaises(DataQualityError):
            validate_daily(self.daily().assign(settle=0), allow_missing_historical_settle=True)

    def test_option_symbols_and_styles(self):
        self.assertEqual(parse_contract("jm2701-P-1200"), ("JM", "JM2701", "P", 1200))
        self.assertEqual(parse_contract("au2612C900"), ("AU", "AU2612", "C", 900))
        self.assertEqual(exercise_style("AU", "AU2210"), "european")
        self.assertEqual(exercise_style("AU", "AU2212"), "american")
        with self.assertRaises(DataQualityError):
            parse_contract("JM0P1000")

    def option_frame(self):
        return normalize_options(pd.DataFrame({"contract": ["JM2701-C-1200", "JM2701-P-1200"],
            "close": [10, 20], "settle": [10, 20], "volume": [0, 10], "open_interest": [4, 8]}), "JM", "2026-09-30")

    def test_no_continuous_underlying_or_fabricated_expiry(self):
        options = self.option_frame()
        matched = attach_underlyings(options, pd.DataFrame({"symbol": ["JM0"], "settle": [1200]}))
        self.assertFalse(matched["underlying_verified"].any())
        self.assertTrue(matched["expiry_date"].isna().all())
        self.assertEqual(matched["price_usable"].tolist(), [False, True])
        self.assertIsNone(option_summary(matched)["put_call_volume_ratio"])

    def test_exact_underlying_mapping(self):
        matched = attach_underlyings(self.option_frame(), pd.DataFrame({"symbol": ["JM2701"], "settle": [1250]}))
        self.assertTrue(matched["underlying_verified"].all())
        self.assertAlmostEqual(matched["moneyness"].iloc[0], 1200/1250)

    def test_purge_uses_original_label_end_even_when_feature_rows_missing(self):
        dates = pd.bdate_range("2024-01-01", periods=100)
        frame = pd.DataFrame({"date": dates, "open": np.arange(100)+100, "high": np.arange(100)+102,
            "low": np.arange(100)+99, "close": np.arange(100)+101, "settle": np.arange(100)+100,
            "hold": np.arange(100)+2000, "volume": np.arange(100)+1000})
        config = DatasetConfig("test", "test", "unused", "test")
        _, model, _ = add_features(frame, config, 5)
        model = model.iloc[::3]
        cutoff = dates[90]
        train = purged_training_rows(model, cutoff)
        self.assertTrue((train["label_end_date"] < cutoff).all())
        self.assertEqual(frame.loc[model.index[0]+5, "date"], model["label_end_date"].iloc[0])

    def test_holiday_short_week_anchor(self):
        dates = pd.Series(pd.to_datetime(["2026-09-28", "2026-09-29", "2026-09-30", "2026-10-09"]))
        self.assertEqual(week_anchor(dates, pd.Timestamp("2026-10-09")), pd.Timestamp("2026-10-09"))

    def test_frozen_models_roundtrip_without_pickle(self):
        frame = pd.DataFrame({"return_5d_pct": np.linspace(-2, 2, 30), "other": np.arange(30),
                              "target": np.linspace(-1, 1, 30), "label_end_date": pd.bdate_range("2026-01-01", periods=30)})
        model = frozen_model(frame, ["return_5d_pct", "other"], "target")
        import json
        restored = json.loads(json.dumps(model))
        self.assertEqual(frozen_predict(model, frame.iloc[[-1]]), frozen_predict(restored, frame.iloc[[-1]]))
        self.assertTrue(np.isfinite(frozen_predict(restored, frame.iloc[[-1]])["xgboost"]))

    def test_immutable_forecast_and_parquet_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ResearchStore(Path(tmp), bucket="")
            payload = {"generated_at": "2026-10-06T00:00:00Z", "report_date": "2026-10-06", "value": 1}
            path = store.archive_forecasts(payload)
            self.assertTrue(Path(path).exists())
            self.assertEqual(path, store.archive_forecasts(payload))
            with self.assertRaises(ValueError):
                store.archive_forecasts({**payload, "value": 2})
            meta = store.snapshot(self.daily(), "main", "JM", "fixture")
            self.assertEqual(meta["rows"], 1)
            self.assertEqual(len(meta["sha256"]), 64)
            self.assertEqual(len(pd.read_parquet(meta["data_path"])), 1)
            with self.assertRaises(ValueError):
                store.publish_json(payload, "../escape.json")

    def test_gcs_immutable_upload_uses_generation_precondition(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "x.json"
            file.write_text("{}")
            store = ResearchStore(Path(tmp), bucket="test")
            with patch.object(ResearchStore, "bucket", new_callable=unittest.mock.PropertyMock) as bucket:
                store.upload(file, "raw/x.json")
                bucket.return_value.blob.return_value.upload_from_filename.assert_called_once_with(str(file), if_generation_match=0)

    def test_power_availability_not_delivery_date_controls_asof(self):
        frame = pd.DataFrame({"region": ["Yunnan"], "metric": ["industrial_tariff"], "delivery_date": ["2026-09-01"],
             "published_at": ["2026-09-02T09:00:00+08:00"], "available_at": ["2026-09-02T10:00:00+08:00"],
             "value": [0.4], "unit": ["CNY/kWh"], "source_url": ["https://example.com/source"]})
        normalized = validate_power(frame)
        self.assertTrue(asof_power(normalized, "2026-09-02T09:30:00+08:00").empty)
        self.assertEqual(len(asof_power(normalized, "2026-09-02T10:00:00+08:00")), 1)

    def test_partial_failure_is_archived_without_faking_success(self):
        from types import SimpleNamespace
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ResearchStore(root / "lake", bucket="")
            csv = root / "au.csv"
            self.daily().to_csv(csv, index=False)
            legacy = {"generated_at": "2026-10-05T00:00:00Z", "report_date": "2026-10-05", "products": {}}
            (root / "latest_signals.json").write_text(json.dumps(legacy))
            args = SimpleNamespace(products=["JM", "AU"], options=["JM"], trade_date=None, chains_only=False, evaluate=False)
            def download(code, *_):
                if code == "JM":
                    raise DataQualityError("stale")
                return csv
            def chain(exchange, *_):
                if exchange == "DCE":
                    raise RuntimeError("exchange timeout")
                return pd.DataFrame({"product": ["AU"], "symbol": ["AU2612"], "settle": [900]})
            from datetime import datetime, timezone
            class PublicationClock:
                calls = 0
                @classmethod
                def now(cls, _):
                    cls.calls += 1
                    return datetime(2026, 10, 6, 0 if cls.calls == 1 else 1, tzinfo=timezone.utc)
            with patch("src.research_pipeline.ResearchStore", return_value=store), patch("src.research_pipeline.PROJECT_ROOT", root), \
                 patch("src.research_pipeline.datetime", PublicationClock), \
                 patch("src.research_pipeline.current_trading_day_metadata", return_value={"report_date": "2026-10-06", "expected_source_date": "2026-09-30"}), \
                 patch("src.research_pipeline.download_product", side_effect=download), patch("src.research_pipeline.fetch_chain", side_effect=chain), \
                 patch("src.research_pipeline.forecast_product", return_value={"source_date": "2026-09-30"}), \
                 patch("src.research_pipeline.fetch_options", side_effect=RuntimeError("JM unavailable")):
                report = run(args)
            self.assertEqual(report["collection_status"], "partial")
            self.assertTrue(report["generated_at"].startswith("2026-10-06T01:"))
            self.assertEqual(report["products"]["JM"]["status"], "failed")
            self.assertEqual(report["products"]["AU"]["status"], "ok")
            self.assertEqual(report["options"]["JM"]["status"], "failed")
            self.assertTrue((root / "lake/state/research/latest.json").exists())
            self.assertEqual(json.loads((root / "latest_signals.json").read_text()), legacy)
            self.assertEqual(json.loads(Path(report["legacy_archive"]["path"]).read_text()), legacy)


if __name__ == "__main__":
    unittest.main()
