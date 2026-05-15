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
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Futures update completed"
} >> "$LOG_DIR/daily_update.log" 2>> "$LOG_DIR/daily_update.error.log"
