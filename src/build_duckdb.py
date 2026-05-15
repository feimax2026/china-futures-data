from __future__ import annotations

from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DB_DIR = PROJECT_ROOT / "data" / "db"
DB_PATH = DB_DIR / "china_futures.duckdb"

DATASETS = [
    ("SM0", "manganese_silicon", RAW_DATA_DIR / "manganese_silicon_SM0.csv"),
    ("JM0", "coking_coal", RAW_DATA_DIR / "coking_coal_JM0.csv"),
    ("I0", "iron_ore", RAW_DATA_DIR / "iron_ore_I0.csv"),
]


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


if __name__ == "__main__":
    main()
