"""Observed real-contract indicators for the change-only seven-product radar.

No orders and no executable P&L from stitched continuous prices. Futures quotes
are selected by open interest on the same completed trading day. All levels are
from that real contract, with strict recent calendar/OHLC validation.
"""
from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.export_latest_signals import trading_day_metadata
from src.feed_access import bounded_ak_frame
from src.product_config import PRODUCTS
from src.research_storage import ResearchStore

UNIVERSE = ("AU", "AG", "CU", "JM", "I", "SM", "SF")
QUOTE_NAMES = {"AU": "黄金", "AG": "白银", "CU": "沪铜", "JM": "焦煤",
               "I": "铁矿石", "SM": "锰硅", "SF": "硅铁"}


def contract_symbol(value: str, code: str, source_date: str) -> str:
    """Expand CZCE YMM only within the next five years, never invent a listing."""
    symbol = str(value).upper().strip()
    match = re.fullmatch(re.escape(code) + r"(\d{3,4})", symbol)
    if not match:
        raise ValueError("Not an actual listed contract")
    digits = match[1]
    year = pd.Timestamp(source_date).year
    if len(digits) == 3:
        matches = [y for y in range(year, year + 6) if y % 10 == int(digits[0])]
        if len(matches) != 1:
            raise ValueError("Ambiguous CZCE year")
        digits = f"{matches[0] % 100:02d}{digits[1:]}"
    maturity = pd.Timestamp(f"{2000 + int(digits[:2])}-{digits[2:]}-01")
    if maturity < pd.Timestamp(source_date).replace(day=1) or maturity.year > year + 5:
        raise ValueError("Expired or implausible maturity")
    return code + digits


def select_contract(quotes: pd.DataFrame, code: str, source_date: str) -> tuple[str, float]:
    required = {"symbol", "tradedate", "position", "volume", "close"}
    if not required.issubset(quotes.columns):
        raise ValueError("Missing quote fields")
    actual = quotes.loc[quotes.symbol.astype(str).str.upper().str.fullmatch(code + r"\d{3,4}")].copy()
    actual = actual.loc[actual.tradedate.astype(str).eq(source_date)]
    for col in ("position", "volume", "close"):
        actual[col] = pd.to_numeric(actual[col], errors="raise")
    actual = actual.loc[(actual.position > 0) & (actual.volume > 0) & (actual.close > 0)]
    if actual.empty or actual.symbol.duplicated().any():
        raise ValueError("No unique, liquid, same-day actual contracts")
    actual = actual.sort_values(["position", "symbol"], ascending=[False, True])
    return contract_symbol(actual.iloc[0].symbol, code, source_date), float(actual.iloc[0]["close"])


def indicators(frame: pd.DataFrame, source_date: str, trade_dates: list[str]) -> dict:
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    frame = frame.loc[frame.date <= source_date].sort_values("date").reset_index(drop=True)
    if frame.empty or frame.date.duplicated().any() or frame.date.iloc[-1] != source_date:
        raise ValueError("Missing latest date or duplicate contract bars")
    cols = ["open", "high", "low", "close", "volume", "hold"]
    for col in cols:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    valid = np.isfinite(frame[cols].to_numpy()).all(axis=1)
    valid &= (frame[["open", "high", "low", "close"]] > 0).all(axis=1).to_numpy()
    valid &= (frame[["volume", "hold"]] >= 0).all(axis=1).to_numpy()
    valid &= (frame.high >= frame[["open", "low", "close"]].max(axis=1)).to_numpy()
    valid &= (frame.low <= frame[["open", "close"]].min(axis=1)).to_numpy()
    # Keep only the contiguous valid suffix. Never silently drop bad bars.
    bad = np.flatnonzero(~valid)
    if len(bad):
        frame = frame.iloc[bad[-1] + 1:].reset_index(drop=True)
    recent_calendar = sorted(d for d in set(trade_dates) if d <= source_date)[-65:]
    if len(frame) < 65 or frame.date.tail(65).tolist() != recent_calendar:
        raise ValueError("Insufficient or gapped recent 65-trading-day contract history")
    close = frame.close
    tr = pd.concat([frame.high - frame.low, (frame.high - close.shift()).abs(),
                    (frame.low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.iloc[:14].mean()
    for value in tr.iloc[14:]:
        atr = (13 * atr + value) / 14
    base = frame.volume.iloc[-21:-1].mean()
    if not np.isfinite(atr) or atr <= 0 or base <= 0 or frame.hold.iloc[-2] <= 0:
        raise ValueError("Unusable ATR, volume baseline or prior open interest")
    ma20, ma60 = close.rolling(20).mean(), close.rolling(60).mean()
    rv = np.log(close / close.shift()).rolling(20).std() * 100
    band = close.rolling(20).std(ddof=0) * 4 / ma20 * 100
    def rank(series):
        window = series.dropna().tail(250)
        return {"percentile": float(window.lt(series.iloc[-1]).mean() * 100), "samples": len(window)}
    return {"history_rows": len(frame), "close": float(close.iloc[-1]),
            "daily_change_pct": float((close.iloc[-1] / close.iloc[-2] - 1) * 100),
            "ma20": float(ma20.iloc[-1]), "ma60": float(ma60.iloc[-1]),
            "ma20_slope_5d_pct": float((ma20.iloc[-1] / ma20.iloc[-6] - 1) * 100),
            "atr14": float(atr), "volume_ratio_prior20": float(frame.volume.iloc[-1] / base),
            "oi_change_pct": float((frame.hold.iloc[-1] / frame.hold.iloc[-2] - 1) * 100),
            "prior5_high": float(frame.high.iloc[-6:-1].max()),
            "prior5_low": float(frame.low.iloc[-6:-1].min()),
            "prior20_high": float(frame.high.iloc[-21:-1].max()),
            "prior20_low": float(frame.low.iloc[-21:-1].min()),
            "prior60_high": float(frame.high.iloc[-61:-1].max()),
            "prior60_low": float(frame.low.iloc[-61:-1].min()),
            "previous_high": float(frame.high.iloc[-2]), "previous_low": float(frame.low.iloc[-2]),
            "rv20_daily_pct": float(rv.iloc[-1]), "rv20_rank": rank(rv), "bandwidth_rank": rank(band),
            "last_bars": frame.tail(12)[["date", *cols]].to_dict(orient="records")}


def collect(store: ResearchStore, *, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    calendar = bounded_ak_frame("calendar", "China")
    dates = sorted(set(pd.to_datetime(calendar.trade_date).dt.strftime("%Y-%m-%d")))
    day = now.astimezone(ZoneInfo("Asia/Tokyo")).date()
    # A calendar with no future coverage must not classify a new year as closed.
    if not dates or dates[-1] < day.isoformat():
        raise ValueError("Trading calendar no longer covers report day")
    metadata = trading_day_metadata(day, [pd.Timestamp(d).date() for d in dates])
    source_date = metadata["expected_source_date"]
    payload = {"schema_version": 1, "kind": "seven_product_radar", "generated_at": now.isoformat(),
               **metadata, "calendar": {"source": "Sina China trading dates", "coverage_end": dates[-1],
                "trading_dates": dates, "observed_at": now.isoformat()}, "products": {}}
    models = store.read_json("state/research/latest.json") or {}
    def scan(code):
        try:
            quotes = bounded_ak_frame("quotes", QUOTE_NAMES[code])
            store.snapshot(quotes, "radar_contract_quotes", code, "sina:actual_contracts_by_oi")
            symbol, quote_close = select_contract(quotes, code, source_date)
            frame = bounded_ak_frame("main", symbol)
            stats = indicators(frame, source_date, dates)
            if not np.isclose(stats["close"], quote_close, rtol=1e-8, atol=1e-6):
                raise ValueError("Quote/history closing-price mismatch")
            evidence = store.snapshot(frame, "radar_contract_history", code, f"sina:{symbol}")
            model = models.get("products", {}).get(code, {})
            quant = None
            if model.get("status") == "ok" and model.get("source_date") == source_date:
                quant = {"forecasts": model["forecasts"], "price_series": model["price_series"],
                         "issued_at": models["generated_at"], "status": "research_only_not_win_rate"}
            return code, {"status": "ok", "contract": symbol, "source_date": source_date,
                          "name_zh": PRODUCTS[code].name_zh, "selection": "Sina same-day maximum open interest",
                          "evidence": evidence, "metrics": stats, "quant": quant}
        except Exception as error:
            return code, {"status": "failed", "error": str(error)}
    with ThreadPoolExecutor(max_workers=3) as pool:
        payload["products"] = dict(pool.map(scan, UNIVERSE))
    payload["collection_status"] = "complete" if all(p["status"] == "ok" for p in payload["products"].values()) else "partial"
    store.archive_forecasts(payload, "radar")
    store.publish_json(payload, "state/radar/latest.json")
    return payload


if __name__ == "__main__":
    result = collect(ResearchStore())
    print(json.dumps({"report_date": result["report_date"], "collection_status": result["collection_status"],
                      "products": {k: {"status": p["status"], "contract": p.get("contract"),
                                        "error": p.get("error")} for k, p in result["products"].items()}}, ensure_ascii=False))
    if result["collection_status"] != "complete":
        raise SystemExit(1)
