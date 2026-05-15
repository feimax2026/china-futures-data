from __future__ import annotations

from pathlib import Path

import akshare as ak
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

FUTURES_SYMBOLS = {
    "SM0": "manganese_silicon",
    "JM0": "coking_coal",
    "I0": "iron_ore",
}


def download_symbol(symbol: str, slug: str) -> Path:
    df = ak.futures_zh_daily_sina(symbol=symbol)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("date").reset_index(drop=True)

    output_path = RAW_DATA_DIR / f"{slug}_{symbol}.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for symbol, slug in FUTURES_SYMBOLS.items():
        output_path = download_symbol(symbol=symbol, slug=slug)
        saved_paths.append(output_path)
        print(f"saved {symbol} -> {output_path}")


if __name__ == "__main__":
    main()
