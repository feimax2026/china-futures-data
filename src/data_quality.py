"""Fail closed on stale or malformed prices before publishing predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd


class DataQualityError(ValueError):
    pass


def validate_daily(frame: pd.DataFrame, expected_date: str | None = None, *, allow_missing_historical_settle: bool = False,
                   allow_historical_anomalies: bool = False) -> pd.DataFrame:
    required = {"date", "open", "high", "low", "close", "settle", "volume", "hold"}
    if frame is None or frame.empty or not required.issubset(frame.columns):
        raise DataQualityError("Empty daily data or missing required fields")
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise").dt.normalize()
    if out["date"].isna().any() or out["date"].duplicated().any():
        raise DataQualityError("Missing or duplicate trading dates")
    for col in required - {"date"}:
        out[col] = pd.to_numeric(out[col], errors="raise")
    out = out.sort_values("date").reset_index(drop=True)
    bad_range = (out["high"] < out[["low", "close", "open"]].max(axis=1)) | (out["low"] > out[["close", "open"]].min(axis=1))
    range_reject = bad_range & out["volume"].gt(0)
    price_columns = ["open", "high", "low", "close"]
    if allow_historical_anomalies:
        invalid = range_reject | ~np.isfinite(out[price_columns].to_numpy(dtype=float)).all(axis=1) | out["close"].le(0)
        if invalid.iloc[-1]:
            raise DataQualityError("Latest OHLC prices are invalid")
        out["price_invalid"] = invalid
        # Keep dates and mark unusable prices: do not shorten prediction horizons.
        out.loc[invalid, price_columns] = np.nan
    numeric_columns = list(required - {"date"})
    if allow_missing_historical_settle:
        # Sina has historical zero settlements. Mark missing, NEVER fill from close.
        out["settle_missing"] = out["settle"].isna() | out["settle"].eq(0)
        out.loc[out["settle_missing"], "settle"] = np.nan
        numeric_columns.remove("settle")
    check_rows = ~out["price_invalid"] if allow_historical_anomalies else pd.Series(True, index=out.index)
    if not np.isfinite(out.loc[check_rows, numeric_columns].to_numpy(dtype=float)).all():
        raise DataQualityError("Daily data contains missing numeric values")
    if (out[["close", "settle"]] <= 0).any().any():
        raise DataQualityError("Non-positive close or settlement")
    if np.isinf(out["settle"].to_numpy(dtype=float)).any():
        raise DataQualityError("Infinite historical settlement")
    if not np.isfinite(out[["volume", "hold"]].to_numpy(dtype=float)).all():
        raise DataQualityError("Invalid volume or open interest")
    if (out[["volume", "hold"]] < 0).any().any():
        raise DataQualityError("Negative volume or open interest")
    if range_reject.any() and not allow_historical_anomalies:
        raise DataQualityError("OHLC prices violate range bounds")
    out = out.sort_values("date").reset_index(drop=True)
    if not np.isfinite(out["settle"].iloc[-1]) or out["settle"].iloc[-1] <= 0:
        raise DataQualityError("Latest settlement is unavailable")
    if expected_date and out["date"].iloc[-1].date().isoformat() != expected_date:
        raise DataQualityError(f"Expected {expected_date}, received {out['date'].iloc[-1].date()}")
    return out


def validate_signal_payload(payload: dict, products: set[str]) -> None:
    if set(payload["products"]) != products:
        raise DataQualityError("Signal product coverage is incomplete")
    expected = payload["expected_source_date"]
    for code, product in payload["products"].items():
        if product["source_date"] != expected:
            raise DataQualityError(f"{code} signal is stale: {product['source_date']} != {expected}")
        if set(product["forecasts"]) != {"5", "10"}:
            raise DataQualityError(f"{code} missing forecast horizons")
        for forecast in product["forecasts"].values():
            value = float(forecast["predicted_return_pct"])
            if not -float("inf") < value < float("inf"):
                raise DataQualityError(f"{code} has a non-finite prediction")
