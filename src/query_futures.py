from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "db" / "china_futures.duckdb"


def get_connection() -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. Run src/build_duckdb.py first."
        )
    return duckdb.connect(str(DB_PATH), read_only=True)


def query_summary(con: duckdb.DuckDBPyConnection) -> None:
    result = con.sql(
        """
        SELECT
            symbol,
            commodity_slug,
            COUNT(*) AS rows,
            MIN(date) AS start_date,
            MAX(date) AS end_date,
            ROUND(MIN(close), 2) AS min_close,
            ROUND(MAX(close), 2) AS max_close
        FROM futures_daily
        GROUP BY 1, 2
        ORDER BY symbol
        """
    ).df()
    print(result.to_string(index=False))


def query_latest(con: duckdb.DuckDBPyConnection) -> None:
    result = con.sql(
        """
        SELECT
            symbol,
            commodity_slug,
            date,
            close,
            settle,
            volume,
            hold
        FROM (
            SELECT
                *,
                ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) AS rn
            FROM futures_daily
        )
        WHERE rn = 1
        ORDER BY symbol
        """
    ).df()
    print(result.to_string(index=False))


def query_returns(
    con: duckdb.DuckDBPyConnection, start_date: str | None, end_date: str | None
) -> None:
    where_clause = []
    if start_date:
        where_clause.append(f"date >= DATE '{start_date}'")
    if end_date:
        where_clause.append(f"date <= DATE '{end_date}'")

    where_sql = f"WHERE {' AND '.join(where_clause)}" if where_clause else ""

    result = con.sql(
        f"""
        WITH filtered AS (
            SELECT *
            FROM futures_daily
            {where_sql}
        ),
        bounds AS (
            SELECT
                symbol,
                commodity_slug,
                MIN(date) AS start_date,
                MAX(date) AS end_date
            FROM filtered
            GROUP BY 1, 2
        )
        SELECT
            b.symbol,
            b.commodity_slug,
            b.start_date,
            s.close AS start_close,
            b.end_date,
            e.close AS end_close,
            ROUND((e.close / NULLIF(s.close, 0) - 1) * 100, 2) AS return_pct
        FROM bounds b
        JOIN filtered s
          ON b.symbol = s.symbol
         AND b.start_date = s.date
        JOIN filtered e
          ON b.symbol = e.symbol
         AND b.end_date = e.date
        ORDER BY b.symbol
        """
    ).df()
    print(result.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query the China futures DuckDB database")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("summary", help="Show summary statistics by symbol")
    subparsers.add_parser("latest", help="Show the latest daily record by symbol")

    returns_parser = subparsers.add_parser("returns", help="Show interval returns")
    returns_parser.add_argument("--start-date", help="Start date in YYYY-MM-DD format")
    returns_parser.add_argument("--end-date", help="End date in YYYY-MM-DD format")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    with get_connection() as con:
        if args.command == "summary":
            query_summary(con)
        elif args.command == "latest":
            query_latest(con)
        elif args.command == "returns":
            query_returns(con, args.start_date, args.end_date)


if __name__ == "__main__":
    main()
