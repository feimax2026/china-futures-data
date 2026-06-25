from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.product_config import PROJECT_ROOT, PRODUCTS, get_product
from src.train_xgboost_compare import (
    DatasetResult,
    ensure_dirs,
    evaluate_forecasts,
    prediction_column,
    product_configs,
    run_dataset,
)


OUTPUT_PATH = PROJECT_ROOT / "latest_signals.json"
HORIZONS = (5, 10)
TOKYO = ZoneInfo("Asia/Tokyo")


def build_payload(
    results: dict[str, dict[int, DatasetResult]],
    generated_at: str | None = None,
    calendar_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    products: dict[str, Any] = {}

    for product_code, horizon_results in results.items():
        product = get_product(product_code)
        forecasts: dict[str, Any] = {}
        source_dates: set[str] = set()
        latest_close: float | None = None

        for horizon, result in sorted(horizon_results.items()):
            latest = result.latest_signal.iloc[-1]
            pred_col = prediction_column(horizon)
            direction = "bullish" if int(latest["signal"]) > 0 else "bearish"
            quality = evaluate_forecasts(result.predictions, horizon)
            matching = quality.loc[quality["forecast"] == direction]
            quality_row = matching.iloc[0] if not matching.empty else None
            source_date = pd_timestamp_iso(latest["date"])
            source_dates.add(source_date)
            latest_close = float(latest["close"])

            forecasts[str(horizon)] = {
                "predicted_return_pct": float(latest[pred_col]),
                "signal": int(latest["signal"]),
                "direction": direction,
                "quality": (
                    {
                        "samples": int(quality_row["samples"]),
                        "avg_realized_return_pct": float(
                            quality_row["avg_realized_return_pct"]
                        ),
                        "direction_accuracy": float(
                            quality_row["direction_accuracy"]
                        ),
                    }
                    if quality_row is not None
                    else None
                ),
            }

        if len(source_dates) != 1:
            raise ValueError(
                f"{product_code} horizons have inconsistent source dates: {sorted(source_dates)}"
            )

        products[product_code] = {
            "name_zh": product.name_zh,
            "source_date": source_dates.pop(),
            "close": latest_close,
            "forecasts": forecasts,
        }

    metadata = calendar_metadata or current_trading_day_metadata()
    return {
        "schema_version": 1,
        "generated_at": generated_at or datetime.now(TOKYO).isoformat(timespec="seconds"),
        **metadata,
        "products": products,
    }


def pd_timestamp_iso(value: Any) -> str:
    return value.date().isoformat() if hasattr(value, "date") else str(value)


def trading_day_metadata(report_date: date, trade_dates: list[date]) -> dict[str, Any]:
    normalized = sorted(set(trade_dates))
    previous = [trade_date for trade_date in normalized if trade_date < report_date]
    if not previous:
        raise ValueError(f"No Chinese trading day before {report_date.isoformat()}")
    return {
        "report_date": report_date.isoformat(),
        "is_china_trading_day": report_date in normalized,
        "expected_source_date": max(previous).isoformat(),
    }


def current_trading_day_metadata() -> dict[str, Any]:
    report_date = datetime.now(TOKYO).date()
    calendar = ak.tool_trade_date_hist_sina()
    trade_dates = [pd.Timestamp(value).date() for value in calendar["trade_date"]]
    return trading_day_metadata(report_date, trade_dates)


def run_models() -> dict[str, dict[int, DatasetResult]]:
    ensure_dirs()
    results: dict[str, dict[int, DatasetResult]] = {}
    for product_code in PRODUCTS:
        main_config = product_configs(product_code)[0]
        results[product_code] = {
            horizon: run_dataset(main_config, horizon) for horizon in HORIZONS
        }
    return results


def main() -> None:
    payload = build_payload(run_models())
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"latest signals -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
