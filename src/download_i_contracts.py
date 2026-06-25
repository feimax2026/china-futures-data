from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.download_contracts import download_contracts


def main() -> None:
    parser = argparse.ArgumentParser(description="Download iron ore individual futures contracts.")
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=date.today().year + 2)
    parser.add_argument("--sleep", type=float, default=0.08)
    args = parser.parse_args()
    download_contracts("I", args.start_year, args.end_year, args.sleep)


if __name__ == "__main__":
    main()
