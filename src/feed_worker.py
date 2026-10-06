"""Bound legacy AKShare network calls in an isolated process (no global app patch)."""
from __future__ import annotations

import argparse
import contextlib
import sys

import akshare as ak
import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["main", "chain", "calendar", "quotes"])
    parser.add_argument("code")
    parser.add_argument("--day")
    args = parser.parse_args()
    original = requests.sessions.Session.request
    def bounded_request(self, method, url, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = (10, 25)
        return original(self, method, url, **kwargs)
    requests.sessions.Session.request = bounded_request
    with contextlib.redirect_stdout(sys.stderr):
        if args.kind == "calendar":
            frame = ak.tool_trade_date_hist_sina()
        elif args.kind == "main":
            frame = ak.futures_zh_daily_sina(symbol=args.code)
        elif args.kind == "quotes":
            frame = ak.futures_zh_realtime(symbol=args.code)
        else:
            frame = ak.get_futures_daily(start_date=args.day, end_date=args.day, market=args.code)
    print(frame.to_json(orient="split", date_format="iso"))


if __name__ == "__main__":
    main()
