from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train_xgboost_compare import (
    DatasetConfig,
    add_features,
    artifact_prefix,
    evaluate_forecasts,
    normalize_market_frame,
    prediction_column,
    purged_train_end,
    run_product,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the JM XGBoost comparison.")
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()
    run_product("JM", args.horizon)


if __name__ == "__main__":
    main()
