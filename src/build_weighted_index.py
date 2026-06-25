from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.product_config import PROJECT_ROOT, get_product


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PRICE_COLUMNS = ["open", "high", "low", "close", "settle"]


def product_contract_dir(product_code: str) -> Path:
    return PROJECT_ROOT / "data" / "raw" / "contracts" / product_code.upper()


def product_output_path(product_code: str) -> Path:
    return PROCESSED_DIR / f"{product_code.lower()}_weighted_index.csv"


def read_contracts(product_code: str) -> pd.DataFrame:
    code = product_code.upper()
    contract_dir = product_contract_dir(code)
    paths = sorted(contract_dir.glob(f"{code}*.csv"))
    if not paths:
        raise FileNotFoundError(f"No contract files found in {contract_dir}. Run src/download_contracts.py first.")

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    for column in PRICE_COLUMNS + ["volume", "hold"]:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    return combined.dropna(subset=["date", "close"])


def weighted_average(group: pd.DataFrame, column: str, weight_column: str) -> float:
    valid = group[[column, weight_column]].dropna()
    valid = valid[valid[weight_column] > 0]
    if valid.empty:
        return np.nan
    return float(np.average(valid[column], weights=valid[weight_column]))


def weighted_return(group: pd.DataFrame, weight_column: str) -> float:
    valid = group[["contract_return_pct", weight_column]].dropna()
    valid = valid[valid[weight_column] > 0]
    if valid.empty:
        return np.nan
    return float(np.average(valid["contract_return_pct"], weights=valid[weight_column]))


def contract_rank_features(group: pd.DataFrame, weight_column: str, prefix: str) -> dict[str, object]:
    ranked = group[["contract", weight_column]].dropna()
    ranked = ranked[ranked[weight_column] > 0].sort_values(weight_column, ascending=False)
    total = ranked[weight_column].sum()
    if ranked.empty or total <= 0:
        return {
            f"dominant_{prefix}_contract": None,
            f"dominant_{prefix}_share": np.nan,
            f"second_{prefix}_share": np.nan,
            f"{prefix}_roll_pressure": np.nan,
        }

    dominant = ranked.iloc[0]
    second_weight = ranked.iloc[1][weight_column] if len(ranked) > 1 else 0.0
    dominant_weight = dominant[weight_column]
    return {
        f"dominant_{prefix}_contract": dominant["contract"],
        f"dominant_{prefix}_share": float(dominant_weight / total),
        f"second_{prefix}_share": float(second_weight / total),
        f"{prefix}_roll_pressure": float(second_weight / dominant_weight) if dominant_weight else np.nan,
    }


def add_contract_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["contract", "date"]).copy()
    out["prev_close"] = out.groupby("contract")["close"].shift(1)
    out["prev_volume"] = out.groupby("contract")["volume"].shift(1)
    out["prev_hold"] = out.groupby("contract")["hold"].shift(1)
    out["contract_return_pct"] = np.log(out["close"] / out["prev_close"]) * 100
    return out.replace([np.inf, -np.inf], np.nan)


def build_return_index(reference_close: pd.Series, return_pct: pd.Series) -> pd.Series:
    values: list[float] = []
    level = np.nan
    for reference, daily_return in zip(reference_close, return_pct, strict=False):
        if np.isnan(level):
            level = float(reference) if pd.notna(reference) else np.nan
        elif pd.notna(daily_return):
            level *= float(np.exp(daily_return / 100))
        elif pd.notna(reference):
            level = float(reference)
        values.append(level)
    return pd.Series(values, index=reference_close.index)


def build_index(df: pd.DataFrame) -> pd.DataFrame:
    df = add_contract_returns(df)
    rows: list[dict[str, object]] = []
    for current_date, group in df.groupby("date", sort=True):
        row: dict[str, object] = {"date": current_date}
        row["active_contracts"] = int(((group["volume"] > 0) | (group["hold"] > 0)).sum())
        row["total_volume"] = float(group["volume"].fillna(0).sum())
        row["total_open_interest"] = float(group["hold"].fillna(0).sum())
        row["volume_weighted_return_pct"] = weighted_return(group, "prev_volume")
        row["oi_weighted_return_pct"] = weighted_return(group, "prev_hold")

        for column in PRICE_COLUMNS:
            row[f"volume_weighted_{column}"] = weighted_average(group, column, "volume")
            row[f"oi_weighted_{column}"] = weighted_average(group, column, "hold")

        row.update(contract_rank_features(group, "volume", "volume"))
        row.update(contract_rank_features(group, "hold", "oi"))
        rows.append(row)

    index = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    index["dominant_contract_changed"] = (
        index["dominant_oi_contract"].ne(index["dominant_oi_contract"].shift(1)).fillna(False).astype(int)
    )
    index["dominant_volume_contract_changed"] = (
        index["dominant_volume_contract"].ne(index["dominant_volume_contract"].shift(1)).fillna(False).astype(int)
    )

    index = index.rename(columns={"total_volume": "volume", "total_open_interest": "open_interest"})
    if not index.empty:
        index.loc[index.index[0], ["dominant_contract_changed", "dominant_volume_contract_changed"]] = 0

    index["return_index_close"] = build_return_index(
        index["oi_weighted_close"], index["oi_weighted_return_pct"]
    )
    scale = index["return_index_close"] / index["volume_weighted_close"]
    for column in PRICE_COLUMNS:
        index[column] = index[f"volume_weighted_{column}"] * scale
    index["close"] = index["return_index_close"]
    return index


def build_product_index(product_code: str) -> pd.DataFrame:
    product = get_product(product_code)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    contracts = read_contracts(product.code)
    index = build_index(contracts)
    output_path = product_output_path(product.code)
    index.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"product -> {product.code} {product.name_en}")
    print(f"contracts rows -> {len(contracts)}")
    print(f"index rows -> {len(index)}")
    print(f"date range -> {index['date'].iloc[0].date()} to {index['date'].iloc[-1].date()}")
    print(f"output -> {output_path}")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a product-level weighted return index from contract CSVs.")
    parser.add_argument("--product", default="JM", help="Product code, e.g. JM or I.")
    args = parser.parse_args()
    build_product_index(args.product)


if __name__ == "__main__":
    main()
