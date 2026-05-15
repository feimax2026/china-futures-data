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

## Auto Update On GitHub

After pushing this folder into a GitHub repository, you can enable the workflow in `.github/workflows/update_futures_data.yml`.
It runs on a schedule or manual trigger, refreshes the CSV files, rebuilds the DuckDB database, and uploads the data files as workflow artifacts.
