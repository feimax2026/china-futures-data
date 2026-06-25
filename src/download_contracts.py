from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import akshare as ak
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.product_config import PROJECT_ROOT, get_product


def contract_dir(product_code: str) -> Path:
    return PROJECT_ROOT / "data" / "raw" / "contracts" / product_code.upper()


def manifest_path(product_code: str) -> Path:
    return PROJECT_ROOT / "data" / "raw" / "contracts" / f"{product_code.lower()}_contract_manifest.csv"


def contract_symbols(product_code: str, start_year: int, end_year: int) -> list[str]:
    code = product_code.upper()
    symbols: list[str] = []
    for year in range(start_year, end_year + 1):
        yy = year % 100
        for month in range(1, 13):
            symbols.append(f"{code}{yy:02d}{month:02d}")
    return symbols


def fetch_contract(symbol: str) -> pd.DataFrame | None:
    try:
        df = ak.futures_zh_daily_sina(symbol=symbol)
    except Exception as exc:
        print(f"skip {symbol}: {type(exc).__name__}: {exc}", flush=True)
        return None

    if df.empty:
        print(f"skip {symbol}: empty", flush=True)
        return None

    expected = ["date", "open", "high", "low", "close", "volume", "hold", "settle"]
    if list(df.columns) != expected:
        print(f"skip {symbol}: unexpected columns {df.columns.tolist()}", flush=True)
        return None

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df.insert(0, "contract", symbol)
    return df.sort_values("date").reset_index(drop=True)


def download_contracts(product_code: str, start_year: int, end_year: int, sleep_seconds: float) -> pd.DataFrame:
    product = get_product(product_code)
    output_dir = contract_dir(product.code)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for symbol in contract_symbols(product.code, start_year, end_year):
        df = fetch_contract(symbol)
        if df is None:
            time.sleep(sleep_seconds)
            continue

        output_path = output_dir / f"{symbol}.csv"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        rows.append(
            {
                "contract": symbol,
                "rows": len(df),
                "start_date": df["date"].iloc[0],
                "end_date": df["date"].iloc[-1],
                "volume_sum": int(df["volume"].sum()),
                "path": output_path.relative_to(PROJECT_ROOT).as_posix(),
            }
        )
        print(f"saved {symbol}: rows={len(df)} {df['date'].iloc[0]}->{df['date'].iloc[-1]}", flush=True)
        time.sleep(sleep_seconds)

    manifest = pd.DataFrame(rows)
    if not manifest.empty:
        manifest = manifest.sort_values(["start_date", "contract"])
    manifest_file = manifest_path(product.code)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_file, index=False, encoding="utf-8-sig")
    print(f"manifest -> {manifest_file}", flush=True)
    print(f"valid contracts -> {len(manifest)}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download individual futures contracts from AkShare/Sina.")
    parser.add_argument("--product", default="JM", help="Product code, e.g. JM or I.")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int, default=date.today().year + 2)
    parser.add_argument("--sleep", type=float, default=0.08, help="Seconds to sleep between requests.")
    args = parser.parse_args()

    product = get_product(args.product)
    start_year = args.start_year if args.start_year is not None else product.default_start_year
    download_contracts(product.code, start_year, args.end_year, args.sleep)


if __name__ == "__main__":
    main()
