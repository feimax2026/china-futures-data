"""Immutable observed snapshots in Parquet; GCS is optional for local work."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import pandas as pd

from src.product_config import PROJECT_ROOT


class ResearchStore:
    def __init__(self, root: Path | None = None, bucket: str | None = None):
        self.root = root or PROJECT_ROOT / "data" / "lake"
        self.bucket_name = bucket if bucket is not None else os.getenv("RESEARCH_BUCKET", "")
        self._bucket = None

    def local_path(self, key: str) -> Path:
        relative = PurePosixPath(key)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Object key must be a relative path")
        path = self.root / key
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("Object key escapes the data lake")
        return path

    @property
    def bucket(self):
        if self._bucket is None:
            from google.cloud import storage
            self._bucket = storage.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT")).bucket(self.bucket_name)
        return self._bucket

    def upload(self, path: Path, key: str, *, immutable: bool = True) -> str:
        relative = PurePosixPath(key)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Object key must be a relative path")
        if self.bucket_name:
            blob = self.bucket.blob(key)
            if immutable:
                from google.api_core.exceptions import PreconditionFailed
                try:
                    blob.upload_from_filename(str(path), if_generation_match=0)
                except PreconditionFailed:
                    # Existing object is valid only when its content is identical.
                    if blob.download_as_bytes() != path.read_bytes():
                        raise ValueError(f"Immutable object conflict: {key}")
            else:
                blob.upload_from_filename(str(path))
            return f"gs://{self.bucket_name}/{key}"
        return str(path)

    def snapshot(self, frame: pd.DataFrame, kind: str, product: str, source: str, *, observed_at: str | None = None) -> dict:
        observed_at = observed_at or datetime.now(timezone.utc).isoformat(timespec="microseconds")
        day = observed_at[:10]
        stamp = observed_at.replace(":", "").replace("+", "_")
        key = f"snapshots/{kind}/product={product}/observed_date={day}/{stamp}.parquet"
        path = self.local_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ValueError(f"Immutable local snapshot conflict: {key}")
        # Some exchange tables mix blank strings and numbers in object columns.
        # Keep original text in observed snapshots; normalized snapshots are typed.
        serializable = frame.copy()
        for column in serializable.columns:
            if pd.api.types.is_object_dtype(serializable[column].dtype):
                serializable[column] = serializable[column].astype("string")
        serializable.to_parquet(path, index=False)
        metadata = {
            "schema_version": 1, "kind": kind, "product": product,
            "source": source, "observed_at": observed_at,
            "rows": len(frame), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "data_path": self.upload(path, key),
        }
        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        self.upload(meta_path, str(PurePosixPath(key).with_suffix(".json")))
        return metadata

    def archive_forecasts(self, payload: dict, name: str = "research") -> str:
        # Never rewrite what was actually published, even if a later rerun changes it.
        stamp = payload["generated_at"].replace(":", "").replace("+", "_")
        key = f"forecasts/{name}/report_date={payload['report_date']}/{stamp}.json"
        path = self.local_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if path.exists() and path.read_text() != content:
            raise ValueError(f"Immutable local forecast conflict: {key}")
        path.write_text(content)
        return self.upload(path, key)

    def raw_response(self, content: bytes, source: str, kind: str, trade_date: str) -> dict:
        """Preserve original exchange response, not only an interpreted table."""
        observed = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        stamp = observed.replace(":", "").replace("+", "_")
        key = f"raw/{kind}/trade_date={trade_date}/{stamp}.json"
        path = self.local_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(content)
        meta = {"source": source, "observed_at": observed, "trade_date": trade_date,
                "sha256": hashlib.sha256(content).hexdigest(), "data_path": self.upload(path, key)}
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        self.upload(meta_path, str(PurePosixPath(key).with_suffix(".meta.json")))
        return meta

    def publish_json(self, payload: dict, key: str) -> str:
        relative = PurePosixPath(key)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Object key must be relative")
        path = self.local_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        return self.upload(path, key, immutable=False)

    def read_json(self, key: str) -> dict | None:
        path = self.local_path(key)
        if not path.exists() and self.bucket_name:
            blob = self.bucket.blob(key)
            if blob.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                blob.download_to_filename(str(path))
        return json.loads(path.read_text()) if path.exists() else None

    def immutable_json(self, payload: dict, key: str) -> str:
        path = self.local_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if path.exists() and path.read_text() != content:
            raise ValueError(f"Immutable JSON conflict: {key}")
        path.write_text(content)
        return self.upload(path, key)

    def restore_main(self, product) -> bool:
        if not self.bucket_name:
            return False
        blob = self.bucket.blob(f"state/main/{product.main_csv}")
        if not blob.exists():
            return False
        target = PROJECT_ROOT / "data" / "raw" / product.main_csv
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(target))
        return True

    def save_main_state(self, product, csv_path: Path) -> str:
        return self.upload(csv_path, f"state/main/{product.main_csv}", immutable=False)
