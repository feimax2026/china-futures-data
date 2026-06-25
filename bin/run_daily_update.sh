#!/bin/zsh

set -euo pipefail

PROJECT_ROOT="/Users/wangfeimin/Documents/Codex/2026-05-15/codex-mac-google-drive-codex-codex/china_futures_akshare"
LOG_DIR="$PROJECT_ROOT/logs"
STAMP="$(date '+%Y-%m-%d %H:%M:%S')"

mkdir -p "$LOG_DIR"

{
  echo "[$STAMP] Starting futures update"
  cd "$PROJECT_ROOT"
  .venv/bin/python src/download_futures.py
  .venv/bin/python src/build_duckdb.py
  .venv/bin/python src/download_jm_contracts.py --start-year 2019 --sleep 0.05
  .venv/bin/python src/build_jm_weighted_index.py
  .venv/bin/python src/train_jm_xgboost_compare.py --horizon 5
  .venv/bin/python src/train_jm_xgboost_compare.py --horizon 10
  .venv/bin/python src/download_i_contracts.py --start-year 2019 --sleep 0.05
  .venv/bin/python src/build_i_weighted_index.py
  .venv/bin/python src/train_i_xgboost_compare.py --horizon 5
  .venv/bin/python src/train_i_xgboost_compare.py --horizon 10
  .venv/bin/python src/download_sm_contracts.py --start-year 2019 --sleep 0.05
  .venv/bin/python src/build_sm_weighted_index.py
  .venv/bin/python src/train_sm_xgboost_compare.py --horizon 5
  .venv/bin/python src/train_sm_xgboost_compare.py --horizon 10
  .venv/bin/python src/download_cu_contracts.py --start-year 2019 --sleep 0.05
  .venv/bin/python src/build_cu_weighted_index.py
  .venv/bin/python src/train_cu_xgboost_compare.py --horizon 5
  .venv/bin/python src/train_cu_xgboost_compare.py --horizon 10
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Futures update completed"
} >> "$LOG_DIR/daily_update.log" 2>> "$LOG_DIR/daily_update.error.log"
