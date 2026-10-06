"""Observed exchange-listed futures, including inactive contracts (no invented symbols)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_quality import DataQualityError
from src.option_data import exchange_json
from src.product_config import PRODUCTS
from src.research_storage import ResearchStore
from src.feed_access import bounded_ak_frame


def normalize_chain(frame: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    required = {"symbol", "date", "open", "high", "low", "close", "settle", "volume", "open_interest"}
    if frame.empty or not required.issubset(frame.columns):
        raise DataQualityError("Empty futures chain or missing required fields")
    out = frame.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
    out["product"] = out["symbol"].str.extract(r"^([A-Z]+)\d{3,4}$", expand=False)
    out = out.loc[out["product"].isin(PRODUCTS)].copy()
    out["date"] = pd.to_datetime(out["date"].astype(str)).dt.strftime("%Y-%m-%d")
    if out.empty or set(out["date"]) != {trade_date} or out["symbol"].duplicated().any():
        raise DataQualityError("Wrong futures chain date, duplicate contracts or empty coverage")
    for col in required - {"symbol", "date"}:
        out[col] = pd.to_numeric(out[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
    values = out[["settle", "volume", "open_interest"]]
    if not np.isfinite(values.to_numpy()).all() or (values < 0).any().any():
        raise DataQualityError("Invalid futures settlement or interest")
    return out[["date", "product", "symbol", "open", "high", "low", "close", "settle", "volume", "open_interest"]].reset_index(drop=True)


def fetch_chain(exchange: str, trade_date: str, store: ResearchStore) -> pd.DataFrame:
    day = trade_date.replace("-", "")
    if exchange == "DCE":
        # Avoid AKShare's positional-column mapping, which breaks on new products.
        url = "https://www.dce.com.cn/dcereport/publicweb/dailystat/dayQuotes"
        payload = {"contractId": "", "lang": "zh", "optionSeries": "", "statisticsType": 0,
                   "tradeDate": day, "tradeType": "1", "varietyId": "all"}
        result = exchange_json(url, store, "futures_DCE", trade_date, payload)
        frame = pd.DataFrame(result.get("data", []))
        frame = frame.rename(columns={"contractId": "symbol", "clearPrice": "settle", "volumn": "volume", "openInterest": "open_interest"})
        if "symbol" in frame:
            frame = frame.loc[frame["symbol"].astype(str).str.match(r"(?i)^[a-z]+\d{4}$")].copy()
        frame["date"] = trade_date
    else:
        frame = bounded_ak_frame("chain", exchange, day)
        store.snapshot(frame, "chain_observed", exchange, f"akshare:get_futures_daily:{exchange}")
    return normalize_chain(frame, trade_date)
