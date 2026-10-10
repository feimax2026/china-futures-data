"""Observed, unit-normalized main-contract basis with actual historical ranks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.feed_access import bounded_ak_frame
from src.product_config import SENTINEL_PRODUCTS

WINDOW = 180
MIN_SAMPLES = 60


def basis_statistics(rows: list[dict], source_date: str, contract: str | None = None) -> dict:
    missing = {"status": "unavailable", "basis_z": None, "basis_percentile": None, "samples": 0}
    if not rows:
        return missing
    frame = pd.DataFrame(rows).drop_duplicates("date", keep="last").sort_values("date")
    frame = frame.loc[frame.date <= source_date]
    if frame.empty:
        return missing
    current = frame.iloc[-1]
    if current.date != source_date:
        return {**missing, "status": "stale", "source_date": current.date}
    current_contract = str(current.contract).upper()
    if len(current_contract.rstrip("0123456789")) + 3 == len(current_contract):
        digits = current_contract[-3:]
        years = [year for year in range(pd.Timestamp(source_date).year, pd.Timestamp(source_date).year + 6) if year % 10 == int(digits[0])]
        if len(years) == 1:
            current_contract = current_contract[:-3] + f"{years[0] % 100:02d}" + digits[1:]
    if contract and current_contract != contract.upper():
        return {**missing, "status": "contract_mismatch", "source_date": source_date}
    prior = pd.to_numeric(frame.loc[frame.date < source_date, "basis_rate_pct"], errors="coerce")
    prior = prior.replace([np.inf, -np.inf], np.nan).dropna().tail(WINDOW)
    value = float(current.basis_rate_pct)
    result = {**missing, "source_date": source_date, "contract": current_contract,
              "basis": float(current.basis), "basis_rate_pct": value, "samples": len(prior),
              "window": WINDOW, "definition": "(spot-main_settlement)/spot; prior observations excluding current",
              "rollover": len(frame) > 1 and frame.iloc[-2].contract != current.contract}
    if not np.isfinite(value) or len(prior) < MIN_SAMPLES:
        return {**result, "status": "insufficient_history"}
    std = float(prior.std(ddof=1))
    percentile = float(((prior < value).sum() + (prior == value).sum() * 0.5) / len(prior) * 100)
    return {**result, "status": "ok" if std > 0 else "zero_variance",
            "basis_z": float((value - prior.mean()) / std) if std > 0 else None,
            "basis_percentile": percentile}


def collect_basis(store, source_date: str) -> dict[str, list[dict]]:
    key = "state/basis/history-v1.json"
    cached = store.read_json(key, refresh=True) or {"rows": [], "coverage_start": None, "coverage_end": None}
    end = pd.Timestamp(source_date)
    # Incremental backfill keeps source requests bounded. Missing history stays
    # missing until 60 actual observations exist; never substitute a range proxy.
    intervals = []
    if cached["coverage_end"]:
        start = pd.Timestamp(cached["coverage_end"]) + pd.Timedelta(days=1)
    else:
        start = end - pd.Timedelta(days=29)
    if start <= end:
        intervals.append((start, end))
    earliest = pd.Timestamp(cached["coverage_start"] or start)
    if (end - earliest).days < 400 and cached["coverage_start"]:
        intervals.append((max(end - pd.Timedelta(days=400), earliest - pd.Timedelta(days=30)), earliest - pd.Timedelta(days=1)))
    rows = list(cached["rows"])
    for start, finish in intervals:
        while start <= finish:
            chunk_end = min(start + pd.Timedelta(days=6), finish)
            frame = bounded_ak_frame("basis", ",".join(SENTINEL_PRODUCTS), start.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d"))
            if frame is not None and not frame.empty:
                for row in frame.to_dict("records"):
                    spot, future = float(row["sp"]), float(row["dom_price"])
                    if row["var"] in SENTINEL_PRODUCTS and np.isfinite([spot, future]).all() and spot > 0 and future > 0:
                        rows.append({"code": row["var"], "date": pd.Timestamp(str(row["date"])).date().isoformat(),
                                     "contract": str(row["dom_symbol"]).upper(), "basis": spot - future,
                                     "basis_rate_pct": (spot - future) / spot * 100})
            cached["coverage_start"] = min(cached["coverage_start"] or start.date().isoformat(), start.date().isoformat())
            cached["coverage_end"] = max(cached["coverage_end"] or chunk_end.date().isoformat(), chunk_end.date().isoformat())
            cached["rows"] = list({(r["code"], r["date"]): r for r in rows if r["date"] >= (end - pd.Timedelta(days=500)).date().isoformat()}.values())
            store.publish_json(cached, key)
            start = chunk_end + pd.Timedelta(days=1)
    return {code: [r for r in cached["rows"] if r["code"] == code] for code in SENTINEL_PRODUCTS}
