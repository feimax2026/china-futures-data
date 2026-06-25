# China Futures AkShare

This project stores China futures daily data separately from any stock-data work.

## Symbols

- `SM0`: manganese silicon continuous contract
- `JM0`: coking coal continuous contract
- `I0`: iron ore continuous contract

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Download

```bash
.venv/bin/python src/download_futures.py
```

CSV files are saved into `data/raw/`.

## Build Local DuckDB

```bash
.venv/bin/python src/build_duckdb.py
```

The DuckDB database is saved into `data/db/china_futures.duckdb`.

## Query Examples

Latest close for all symbols:

```bash
.venv/bin/python src/query_futures.py latest
```

Summary by symbol:

```bash
.venv/bin/python src/query_futures.py summary
```

Interval returns:

```bash
.venv/bin/python src/query_futures.py returns --start-date 2025-01-01 --end-date 2025-05-14
```

## First Research Workflow

Run the first black-chain research report:

```bash
.venv/bin/python src/analyze_black_chain.py
```

It reads `data/db/china_futures.duckdb`, builds daily return and volatility features, estimates a VAR model, exports IRF and FEVD charts, and writes a Markdown report into `reports/black_chain_var_report.md`.

## XGBoost Research Workflow

Run the first coking coal machine-learning report:

```bash
.venv/bin/python src/train_jm_xgboost.py
```

It reads `data/raw/coking_coal_JM0.csv`, maps `hold` to `open_interest`, builds lagged return, volatility, volume, and open-interest features, trains an XGBoost model with walk-forward splits, predicts future 5-day returns, and writes outputs into `analysis/ml/` and `reports/ml/`.

## Weighted Return Index

Download individual contracts and build product-level weighted return indexes:

```bash
.venv/bin/python src/download_jm_contracts.py
.venv/bin/python src/build_jm_weighted_index.py
.venv/bin/python src/download_i_contracts.py
.venv/bin/python src/build_i_weighted_index.py
.venv/bin/python src/download_sm_contracts.py
.venv/bin/python src/build_sm_weighted_index.py
.venv/bin/python src/download_cu_contracts.py
.venv/bin/python src/build_cu_weighted_index.py
```

The contract files are saved into `data/raw/contracts/<PRODUCT>/`, and the weighted indexes are saved into `data/processed/*_weighted_index.csv`. The model-facing OHLC columns are built by compounding previous-day open-interest-weighted contract returns, which avoids directly averaging different delivery-month price levels. The files also keep raw volume-weighted prices, open-interest-weighted prices, total volume, total open interest, dominant-contract shares, and roll-pressure features.

Compare the original continuous contracts against their weighted return indexes:

```bash
.venv/bin/python src/train_jm_xgboost_compare.py --horizon 5
.venv/bin/python src/train_i_xgboost_compare.py --horizon 5
.venv/bin/python src/train_sm_xgboost_compare.py --horizon 5
.venv/bin/python src/train_cu_xgboost_compare.py --horizon 5
```

The comparison reports are written to `reports/ml/jm_xgboost_compare_report.md`, `reports/ml/i_xgboost_compare_report.md`, `reports/ml/sm_xgboost_compare_report.md`, and `reports/ml/cu_xgboost_compare_report.md`.

Run the parallel 10-trading-day forecasts with:

```bash
.venv/bin/python src/train_jm_xgboost_compare.py --horizon 10
.venv/bin/python src/train_i_xgboost_compare.py --horizon 10
.venv/bin/python src/train_sm_xgboost_compare.py --horizon 10
.venv/bin/python src/train_cu_xgboost_compare.py --horizon 10
```

Five-day outputs retain their existing filenames for compatibility. Other horizons include the horizon in their names, for example `analysis/ml/jm_main_10d_xgboost_latest_signal.csv` and `reports/ml/jm_10d_xgboost_compare_report.md`. Walk-forward training purges the selected horizon between each training and test window to prevent overlapping forward-return labels from leaking into the test period.

Each run also writes a `*_xgboost_forecast_quality.csv` file. This table evaluates the model as a forecaster rather than as a trading strategy: it shows what the future return actually averaged after bullish and bearish predictions, plus the corresponding direction accuracy.

You can also use the generic commands:

```bash
.venv/bin/python src/download_contracts.py --product I
.venv/bin/python src/build_weighted_index.py --product I
.venv/bin/python src/train_xgboost_compare.py --product I --horizon 10
```

Supported products are `JM`, `I`, `SM`, and `CU`.

## Auto Update On GitHub

The public GitHub workflow in `.github/workflows/update_futures_data.yml` runs daily at 03:07 Asia/Tokyo. It refreshes the four main-continuous CSV files, trains the 5-day and 10-day main-contract models, and commits a compact `latest_signals.json` for downstream reporting. Generated datasets, model tables, and figures remain outside Git; only the latest signal payload is published.

Run the same export locally with:

```bash
.venv/bin/python src/download_futures.py
.venv/bin/python src/export_latest_signals.py
```
