from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train_xgboost_compare import run_product


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CU XGBoost comparison.")
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()
    run_product("CU", args.horizon)


if __name__ == "__main__":
    main()
