"""Daily exchange chains, with exact-month futures rather than continuous underlyings."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.data_quality import DataQualityError
from src.research_storage import ResearchStore


@dataclass(frozen=True)
class OptionConfig:
    code: str
    exchange: str
    multiplier: int
    tick: float
    name: str


OPTIONS = {
    "AU": OptionConfig("AU", "SHFE", 1000, 0.02, "黄金期权"),
    "AG": OptionConfig("AG", "SHFE", 15, 0.5, "白银期权"),
    "CU": OptionConfig("CU", "SHFE", 5, 2.0, "铜期权"),
    "SC": OptionConfig("SC", "INE", 1000, 0.1, "原油期权"),
    "JM": OptionConfig("JM", "DCE", 60, 0.1, "焦煤期权"),
}


def parse_contract(symbol: str) -> tuple[str, str, str, float]:
    match = re.fullmatch(r"([A-Za-z]+)(\d{4})-?([CPcp])-?(\d+(?:\.\d+)?)", symbol.strip())
    if not match:
        raise DataQualityError(f"Unrecognized option contract: {symbol}")
    product, month, side, strike = match.groups()
    if not 1 <= int(month[2:]) <= 12 or float(strike) <= 0:
        raise DataQualityError(f"Invalid month or strike: {symbol}")
    return product.upper(), product.upper() + month, side.upper(), float(strike)


def exercise_style(product: str, underlying: str) -> str:
    # AU/CU changed style by contract month, not simply observation date.
    month = underlying[-4:]
    if (product == "AU" and month < "2212") or (product == "CU" and month < "2211"):
        return "european"
    return "american"


def exchange_json(url: str, store: ResearchStore, kind: str, trade_date: str, payload=None) -> dict:
    session = requests.Session()
    retry = Retry(total=2, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET", "POST"])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    with session:
        response = session.request("POST" if payload else "GET", url, json=payload, timeout=(10, 40))
        response.raise_for_status()
        store.raw_response(response.content, url, kind, trade_date)
        result = response.json()
    if not isinstance(result, dict):
        raise DataQualityError("Unexpected exchange JSON type")
    return result


def fetch_options(code: str, trade_date: str, store: ResearchStore) -> pd.DataFrame:
    config = OPTIONS[code]
    day = trade_date.replace("-", "")
    if config.exchange == "DCE":
        url = "https://www.dce.com.cn/dcereport/publicweb/dailystat/dayQuotes"
        payload = {"contractId": "", "lang": "zh", "optionSeries": "", "statisticsType": 0,
                   "tradeDate": day, "tradeType": "2", "varietyId": code.lower()}
        result = exchange_json(url, store, "options_DCE_JM", trade_date, payload)
        frame = pd.DataFrame(result.get("data", []))
        frame = frame.rename(columns={"contractId": "contract", "clearPrice": "settle", "volumn": "volume",
                                      "openInterest": "open_interest", "delta": "exchange_delta",
                                      "impliedVolatility": "exchange_iv_pct"})
    else:
        host = "www.ine.cn" if config.exchange == "INE" else "www.shfe.com.cn"
        url = f"https://{host}/data/tradedata/option/dailydata/kx{day}.dat"
        result = exchange_json(url, store, f"options_{config.exchange}_{code}", trade_date)
        frame = pd.DataFrame(result.get("o_curinstrument", []))
        frame = frame.rename(columns={"INSTRUMENTID": "contract", "OPENPRICE": "open", "HIGHESTPRICE": "high",
                                      "LOWESTPRICE": "low", "CLOSEPRICE": "close", "SETTLEMENTPRICE": "settle",
                                      "VOLUME": "volume", "OPENINTEREST": "open_interest", "DELTA": "exchange_delta"})
    if "contract" not in frame:
        raise DataQualityError(f"{code}: empty exchange chain")
    mask = frame["contract"].astype(str).str.match(rf"(?i)^{code}\d{{4}}-?[CP]-?\d")
    return normalize_options(frame.loc[mask].copy(), code, trade_date)


def normalize_options(frame: pd.DataFrame, code: str, trade_date: str) -> pd.DataFrame:
    config = OPTIONS[code]
    required = {"contract", "close", "settle", "volume", "open_interest"}
    if frame.empty or not required.issubset(frame.columns):
        raise DataQualityError(f"{code}: missing option fields or empty data")
    out = frame.copy()
    out["contract"] = out["contract"].str.upper().str.strip()
    if out["contract"].duplicated().any():
        raise DataQualityError("Duplicate option contracts")
    parts = [parse_contract(symbol) for symbol in out["contract"]]
    if any(part[0] != code for part in parts):
        raise DataQualityError("Wrong option product returned by source")
    out["product"] = code
    out["underlying_contract"] = [part[1] for part in parts]
    out["option_type"] = [part[2] for part in parts]
    out["strike"] = [part[3] for part in parts]
    for column in ["open", "high", "low", "close", "settle", "volume", "open_interest", "exchange_delta", "exchange_iv_pct"]:
        if column not in out:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column].astype(str).str.replace(",", "", regex=False), errors="coerce")
    numeric = out[["settle", "volume", "open_interest"]]
    if not np.isfinite(numeric.to_numpy()).all() or (numeric < 0).any().any():
        raise DataQualityError("Invalid option settlement, volume or open interest")
    out["trade_date"] = trade_date
    out["exchange"] = config.exchange
    out["multiplier"] = config.multiplier
    out["tick"] = config.tick
    out["exercise_style"] = [exercise_style(code, part[1]) for part in parts]
    # Do not invent expiry dates or proprietary Greeks before verifying the calendar.
    out["expiry_date"] = None
    out["expiry_verified"] = False
    out["zero_volume"] = out["volume"] == 0
    out["price_usable"] = (out["volume"] > 0) & (out["close"] > 0) & (out["settle"] > 0)
    columns = ["trade_date", "product", "exchange", "contract", "underlying_contract", "option_type", "strike",
               "open", "high", "low", "close", "settle", "volume", "open_interest", "exchange_delta", "exchange_iv_pct",
               "multiplier", "tick", "exercise_style", "expiry_date", "expiry_verified", "zero_volume", "price_usable"]
    return out[columns].reset_index(drop=True)


def attach_underlyings(options: pd.DataFrame, futures: pd.DataFrame | None) -> pd.DataFrame:
    out = options.copy()
    if futures is None or futures.empty:
        out["underlying_settle"] = np.nan
    else:
        out["underlying_settle"] = out["underlying_contract"].map(futures.set_index("symbol")["settle"])
    out["underlying_verified"] = out["underlying_settle"].gt(0) & out["underlying_settle"].notna()
    out["moneyness"] = out["strike"] / out["underlying_settle"].where(out["underlying_verified"])
    return out


def option_summary(frame: pd.DataFrame) -> dict:
    call = frame.loc[frame["option_type"] == "C"]
    put = frame.loc[frame["option_type"] == "P"]
    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator > 0 else None
    return {"contracts": len(frame), "traded_contracts": int(frame["price_usable"].sum()),
            "underlying_matched": int(frame["underlying_verified"].sum()),
            "put_call_volume_ratio": ratio(put["volume"].sum(), call["volume"].sum()),
            "put_call_oi_ratio": ratio(put["open_interest"].sum(), call["open_interest"].sum()),
            "iv_surface_status": "blocked_pending_verified_expiry_and_quote_quality"}
