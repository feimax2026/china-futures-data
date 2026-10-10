from __future__ import annotations

import io
import subprocess
import sys

import pandas as pd

from src.product_config import PROJECT_ROOT


def bounded_ak_frame(kind: str, code: str, day: str | None = None, end_day: str | None = None) -> pd.DataFrame:
    command = [sys.executable, "-m", "src.feed_worker", kind, code]
    if day:
        command += ["--day", day]
    if end_day:
        command += ["--end-day", end_day]
    try:
        result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120, check=True)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"{kind}:{code} feed exceeded 120 seconds") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"{kind}:{code} source failed: {error.stderr[-1500:]}") from error
    return pd.read_json(io.StringIO(result.stdout), orient="split", convert_dates=False)
