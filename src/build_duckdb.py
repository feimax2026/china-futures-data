from __future__ import annotations

from pathlib import Path

import duckdb
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.product_config import PRODUCTS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DB_DIR = PROJECT_ROOT / "data" / "db"
DB_PATH = DB_DIR / "china_futures.duckdb"

DATASETS = [(product.continuous_symbol, product.lower_code, RAW_DATA_DIR / product.main_csv) for product in PRODUCTS.values()]


def main() -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(DB_PATH)) as con:
        con.execute("DROP TABLE IF EXISTS futures_daily")
        con.execute(
            """
            CREATE TABLE futures_daily (
                symbol VARCHAR,
                commodity_slug VARCHAR,
                date DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                hold BIGINT,
                settle DOUBLE
            )
            """
        )

        for symbol, slug, csv_path in DATASETS:
            if not csv_path.exists():
                continue
            con.execute(
                f"""
                INSERT INTO futures_daily
                SELECT
                    '{symbol}' AS symbol,
                    '{slug}' AS commodity_slug,
                    CAST(date AS DATE) AS date,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    hold,
                    settle
                FROM read_csv_auto('{csv_path.as_posix()}', header=true)
                """
            )

        con.execute("CREATE INDEX idx_futures_symbol_date ON futures_daily(symbol, date)")

        con.execute("DROP TABLE IF EXISTS futures_metadata")
        con.execute(
            """
            CREATE TABLE futures_metadata AS
            SELECT
                symbol,
                commodity_slug,
                MIN(date) AS start_date,
                MAX(date) AS end_date,
                COUNT(*) AS row_count
            FROM futures_daily
            GROUP BY 1, 2
            ORDER BY 1
            """
        )

        print(f"built database -> {DB_PATH}")
        print(con.sql("SELECT * FROM futures_metadata").df().to_string(index=False))
        for kind in ("futures_chain", "options_chain", "power_external"):
            paths = list((PROJECT_ROOT / "data" / "lake" / "snapshots" / kind).glob("**/*.parquet"))
            if paths:
                # Observed snapshots deliberately retain reruns and corrections.
                glob = (PROJECT_ROOT / "data" / "lake" / "snapshots" / kind / "**" / "*.parquet").as_posix()
                con.execute(f"CREATE OR REPLACE VIEW {kind}_observations AS SELECT * FROM read_parquet('{glob}', union_by_name=true, filename=true)")


if __name__ == "__main__":
    main()
