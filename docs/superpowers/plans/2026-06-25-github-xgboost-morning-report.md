# GitHub XGBoost Morning Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the four-product XGBoost forecasts on GitHub without a powered-on Mac and send a Telegram report at 09:30 JST only on Chinese futures trading days.

**Architecture:** The public `china-futures-data` repository refreshes the four main continuous contracts at 03:07 JST, trains 5-day and 10-day models, and commits a compact `latest_signals.json`. The private `commodity-sentinel` repository runs a separate 09:30 JST workflow, checks the Chinese exchange calendar, validates that the signal source date is the latest completed trading day, and sends Telegram output using its existing secrets.

**Tech Stack:** Python 3.12, AkShare, pandas, XGBoost, GitHub Actions, Telegram Bot API.

---

### Task 1: Export a stable signal payload

**Files:**
- Create: `src/export_latest_signals.py`
- Test: `tests/test_export_latest_signals.py`

- [ ] Write a failing test asserting that a payload contains `generated_at`, product code, source date, 5-day/10-day predictions, directions, and historical forecast-quality fields.
- [ ] Run `.venv/bin/python -m unittest tests.test_export_latest_signals -v` and confirm the export helper is missing.
- [ ] Implement payload construction from the main-continuous `DatasetResult` objects without loading weighted-contract data.
- [ ] Run the focused test and confirm it passes.

### Task 2: Move model generation to GitHub

**Files:**
- Modify: `.github/workflows/update_futures_data.yml`
- Modify: `.gitignore`
- Modify: `README.md`

- [ ] Schedule the workflow at `7 18 * * *`, corresponding to 03:07 JST on the following day.
- [ ] Install dependencies, download main-continuous data, run unit tests, export `latest_signals.json`, validate it, and commit only that compact payload.
- [ ] Ignore generated analysis and report directories while keeping `latest_signals.json` tracked.
- [ ] Run the exporter locally and validate its JSON structure.

### Task 3: Build the trading-day-aware Telegram sender

**Files:**
- Create: `/Users/wangfeimin/Documents/github管理/commodity-sentinel/xgboost_morning_report.py`
- Create: `/Users/wangfeimin/Documents/github管理/commodity-sentinel/tests/test_xgboost_morning_report.py`

- [ ] Write failing tests for holiday skipping, previous-trading-day validation, stale-data warnings, and Chinese Telegram formatting.
- [ ] Run `python -m unittest tests.test_xgboost_morning_report -v` and confirm the module is missing.
- [ ] Implement the calendar check with `ak.tool_trade_date_hist_sina()`, payload validation, message formatting, and Telegram delivery.
- [ ] Run the focused tests and confirm they pass.

### Task 4: Schedule and publish the morning report

**Files:**
- Create: `/Users/wangfeimin/Documents/github管理/commodity-sentinel/.github/workflows/xgboost_morning_report.yml`
- Modify: `/Users/wangfeimin/Documents/github管理/commodity-sentinel/README.md`

- [ ] Schedule the workflow at `30 0 * * *`, corresponding to 09:30 JST.
- [ ] Reuse `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`; do not authenticate to Vertex or invoke Gemini.
- [ ] Run both repositories' full unit-test suites and syntax checks.
- [ ] Commit and push both repositories, manually dispatch workflows, and inspect the GitHub Actions results.
- [ ] Pause the Codex automation `update-china-futures-data` after the GitHub workflows are verified.
