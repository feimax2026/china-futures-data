from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_quality import validate_daily
from src.product_config import PROJECT_ROOT, PRODUCTS, LEGACY_PRODUCTS, get_product
from src.research_storage import ResearchStore
from src.feed_access import bounded_ak_frame
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

def download_product(code: str, store: ResearchStore, expected_date: str | None = None) -> Path:
    product = get_product(code)
    frame = bounded_ak_frame("main", product.continuous_symbol)
    store.snapshot(frame, "main_observed", code, "akshare:futures_zh_daily_sina")
    incoming = validate_daily(frame, expected_date, allow_missing_historical_settle=True, allow_historical_anomalies=True)
    path = RAW_DATA_DIR / product.main_csv
    if not path.exists():
        store.restore_main(product)
    if path.exists():
        old = pd.read_csv(path)
        old["date"] = pd.to_datetime(old["date"])
        incoming = pd.concat([old, incoming]).drop_duplicates("date", keep="last")
        incoming = validate_daily(incoming, expected_date, allow_missing_historical_settle=True, allow_historical_anomalies=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    incoming.to_csv(path, index=False, date_format="%Y-%m-%d", encoding="utf-8-sig")
    store.save_main_state(product, path)
    print(f"saved {code}: {len(incoming)} rows, through {incoming['date'].max().date()}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", nargs="+", choices=sorted(PRODUCTS), default=list(LEGACY_PRODUCTS))
    args = parser.parse_args()
    store = ResearchStore()
    for code in args.products:
        download_product(code, store)


if __name__ == "__main__":
    main()
