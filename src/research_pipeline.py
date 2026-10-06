"""Separate expanded data lake / research job, leaving the legacy JSON contract intact."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.contract_chain import fetch_chain
from src.download_futures import download_product
from src.export_latest_signals import current_trading_day_metadata
from src.option_data import OPTIONS, attach_underlyings, fetch_options, option_summary
from src.product_config import PRODUCTS, PROJECT_ROOT
from src.research_models import evaluate_product, forecast_product
from src.research_storage import ResearchStore


def run(args) -> dict:
    store = ResearchStore()
    metadata = current_trading_day_metadata()
    trade_date = args.trade_date or metadata["expected_source_date"]
    report = {"schema_version": 2, "generated_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
              **metadata, "expected_source_date": trade_date, "status": "research_only",
              "products": {}, "options": {}, "collection": {}, "evaluations": [],
              "external_power_data_status": "not_connected"}
    legacy_path = PROJECT_ROOT / "latest_signals.json"
    if legacy_path.exists():
        # Archive the already-issued legacy payload, not a reconstructed backtest.
        try:
            legacy = json.loads(legacy_path.read_text())
            report["legacy_archive"] = {"status": "archived", "path": store.archive_forecasts(legacy, "legacy"),
                                        "generated_at": legacy["generated_at"]}
        except Exception as error:
            report["legacy_archive"] = {"status": "failed", "error": str(error)}
    if not args.chains_only:
        for code in args.products:
            try:
                path = download_product(code, store, trade_date)
                frame = pd.read_csv(path)
                report["products"][code] = {"status": "ok", **forecast_product(frame, code, trade_date, store)}
                if args.evaluate:
                    for horizon in (5, 10):
                        summary, predictions = evaluate_product(frame, code, horizon)
                        store.snapshot(predictions, f"oos_{horizon}d", code, "research_models:weekly_purged")
                        report["evaluations"].append(summary)
            except Exception as error:
                report["products"][code] = {"status": "failed", "error": str(error)}
                print(f"{code}: failed ({type(error).__name__})", flush=True)
    chains = {}
    for exchange in sorted({PRODUCTS[code].exchange for code in args.products}):
        try:
            frame = fetch_chain(exchange, trade_date, store)
            for code in args.products:
                part = frame.loc[frame["product"] == code].copy()
                if PRODUCTS[code].exchange != exchange:
                    continue
                if part.empty:
                    report["collection"][code] = {"status": "failed", "error": "empty product chain"}
                    continue
                store.snapshot(part, "futures_chain", code, f"exchange:{exchange}:{trade_date}")
                chains[code] = part
                report["collection"][code] = {"status": "ok", "contracts": len(part), "source_date": trade_date}
        except Exception as error:
            for code in args.products:
                if PRODUCTS[code].exchange == exchange:
                    report["collection"][code] = {"status": "failed", "error": str(error)}
            print(f"{exchange} chain: failed ({type(error).__name__})", flush=True)
    for code in args.options:
        try:
            frame = attach_underlyings(fetch_options(code, trade_date, store), chains.get(code))
            store.snapshot(frame, "options_chain", code, f"exchange:{OPTIONS[code].exchange}:{trade_date}")
            matched = bool(frame["underlying_verified"].all())
            report["options"][code] = {"status": "ok" if matched else "degraded", "source_date": trade_date, **option_summary(frame)}
        except Exception as error:
            report["options"][code] = {"status": "failed", "error": str(error)}
            print(f"{code} options: failed ({type(error).__name__})", flush=True)
    failed = any(item["status"] != "ok" for section in ("products", "collection", "options") for item in report[section].values())
    failed = failed or report.get("legacy_archive", {}).get("status") == "failed"
    report["collection_status"] = "partial" if failed else "complete"
    # Timestamp actual publication after fitting/collection, not the start of a
    # potentially long run. A forecast must not appear issued before training.
    report["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    store.archive_forecasts(report)
    store.publish_json(report, "state/research/latest.json")
    (PROJECT_ROOT / "research_signals.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"collection_status": report["collection_status"], "source_date": trade_date,
                      "products": {k: v["status"] for k, v in report["products"].items()},
                      "options": {k: v["status"] for k, v in report["options"].items()}}, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", nargs="+", choices=sorted(PRODUCTS), default=list(PRODUCTS))
    parser.add_argument("--options", nargs="+", choices=sorted(OPTIONS), default=list(OPTIONS))
    parser.add_argument("--trade-date", help="Explicit historical YYYY-MM-DD for chain backfill")
    parser.add_argument("--chains-only", action="store_true", help="Skip main-series download and forecasts")
    parser.add_argument("--evaluate", action="store_true", help="Last-year weekly purged OOS comparisons; slower")
    parser.add_argument("--allow-partial", action="store_true", help="For diagnostics only: return success on incomplete feeds")
    args = parser.parse_args()
    if args.trade_date and not args.chains_only:
        parser.error("Historical --trade-date requires --chains-only (main feed returns latest history)")
    report = run(args)
    if report["collection_status"] != "complete" and not args.allow_partial:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
