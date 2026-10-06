"""Explicit external feed import; regional spot power is NOT smelter effective cost."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.research_storage import ResearchStore


def validate_power(frame: pd.DataFrame) -> pd.DataFrame:
    fields = {"region", "metric", "delivery_date", "published_at", "available_at", "value", "unit", "source_url"}
    if frame.empty or not fields.issubset(frame):
        raise ValueError(f"Power feed needs columns: {sorted(fields)}")
    out = frame.copy()
    for col in ["published_at", "available_at"]:
        # Sources must specify time zones to avoid guessed availability.
        if not out[col].astype(str).str.contains(r"(?:Z|[+-]\d{2}:\d{2})$").all():
            raise ValueError(f"{col} requires an explicit time zone")
        out[col] = pd.to_datetime(out[col], utc=True)
    out["delivery_date"] = pd.to_datetime(out["delivery_date"]).dt.date.astype(str)
    out["value"] = pd.to_numeric(out["value"], errors="raise")
    if not np.isfinite(out["value"].to_numpy()).all():
        raise ValueError("Power feed has non-finite values")
    if (out["available_at"] < out["published_at"]).any():
        raise ValueError("Available time cannot predate publication")
    if out[list(fields)].isna().any().any() or not out["source_url"].str.match(r"https?://").all():
        raise ValueError("Missing power metadata or source URL")
    if not out["metric"].isin(["spot_day_ahead", "spot_real_time", "industrial_tariff", "smelter_effective_cost", "load", "hydro", "coal_cost"]).all():
        raise ValueError("Unknown power metric; do not merge tariff and spot electricity")
    return out


def asof_power(frame: pd.DataFrame, decision_time: str) -> pd.DataFrame:
    return frame.loc[frame["available_at"] <= pd.Timestamp(decision_time)].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    frame = validate_power(pd.read_csv(args.csv))
    store = ResearchStore()
    for region, group in frame.groupby("region"):
        store.snapshot(group, "power_external", str(region), "user_provided_csv:source_url_per_row")
    print(f"Archived {len(frame)} power observations; not auto-joined into models")


if __name__ == "__main__":
    main()
