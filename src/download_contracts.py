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


# China commodity exchanges list a contract roughly a year ahead of delivery.
# Requesting further out than this just draws guaranteed-empty responses (seen
# in practice as `ValueError: Length mismatch` from Sina) for every contract,
# on every product, on every run.
MAX_FORWARD_MONTHS = 15

# Once a contract's delivery month is this far in the past, trading has closed
# for good and its history file will never change again, so a cached copy can
# be trusted instead of re-downloading the full series from scratch.
SETTLEMENT_BUFFER_MONTHS = 2


def contract_symbols(product_code: str, start_year: int, end_year: int) -> list[str]:
    code = product_code.upper()
    symbols: list[str] = []
    for year in range(start_year, end_year + 1):
        yy = year % 100
        for month in range(1, 13):
            symbols.append(f"{code}{yy:02d}{month:02d}")
    return symbols


def contract_period(product_code: str, symbol: str) -> date:
    suffix = symbol[len(product_code):]
    year = 2000 + int(suffix[:2])
    month = int(suffix[2:4])
    return date(year, month, 1)


def months_between(later: date, earlier: date) -> int:
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


def within_listing_window(product_code: str, symbol: str, today: date) -> bool:
    period = contract_period(product_code, symbol)
    return months_between(period, today) <= MAX_FORWARD_MONTHS


def is_settled(product_code: str, symbol: str, today: date) -> bool:
    period = contract_period(product_code, symbol)
    return months_between(today, period) > SETTLEMENT_BUFFER_MONTHS


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


def manifest_row(symbol: str, df: pd.DataFrame, output_path: Path) -> dict[str, object]:
    return {
        "contract": symbol,
        "rows": len(df),
        "start_date": df["date"].iloc[0],
        "end_date": df["date"].iloc[-1],
        "volume_sum": int(df["volume"].sum()),
        "path": output_path.relative_to(PROJECT_ROOT).as_posix(),
    }


def download_contracts(
    product_code: str,
    start_year: int,
    end_year: int,
    sleep_seconds: float,
    force: bool = False,
    today: date | None = None,
) -> pd.DataFrame:
    product = get_product(product_code)
    output_dir = contract_dir(product.code)
    output_dir.mkdir(parents=True, exist_ok=True)
    today = today or date.today()
    rows: list[dict[str, object]] = []

    all_symbols = contract_symbols(product.code, start_year, end_year)
    symbols = [symbol for symbol in all_symbols if within_listing_window(product.code, symbol, today)]
    skipped_far_future = len(all_symbols) - len(symbols)
    if skipped_far_future:
        print(f"skipping {skipped_far_future} contracts beyond the {MAX_FORWARD_MONTHS}-month listing window", flush=True)

    for symbol in symbols:
        output_path = output_dir / f"{symbol}.csv"

        if not force and output_path.exists() and is_settled(product.code, symbol, today):
            cached = pd.read_csv(output_path)
            if not cached.empty:
                rows.append(manifest_row(symbol, cached, output_path))
                print(f"cached {symbol}: rows={len(cached)} (settled, skipped refetch)", flush=True)
                continue

        df = fetch_contract(symbol)
        if df is None:
            time.sleep(sleep_seconds)
            continue

        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        rows.append(manifest_row(symbol, df, output_path))
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download every contract even if a settled contract's file is already cached.",
    )
    args = parser.parse_args()

    product = get_product(args.product)
    start_year = args.start_year if args.start_year is not None else product.default_start_year
    download_contracts(product.code, start_year, args.end_year, args.sleep, force=args.force)


if __name__ == "__main__":
    main()
