"""Resumable mapping from batch input file content hash to batch_id for `sference batch stream`."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _cache_file_path() -> Path:
    override = os.environ.get("SFERENCE_STREAM_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".sference" / "stream_cache.json"


def cache_key_from_file(path: Path) -> str:
    """SHA-256 hex digest of raw file bytes (same content => same key regardless of path)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_all() -> dict[str, Any]:
    p = _cache_file_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_all(data: dict[str, Any]) -> None:
    p = _cache_file_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def get_cached_batch_id(key: str, base_url: str) -> str | None:
    """Return batch_id if cache has an entry for this content hash and matching base_url."""
    entry = _load_all().get(key)
    if not isinstance(entry, dict):
        return None
    if entry.get("base_url") != base_url.rstrip("/"):
        return None
    bid = entry.get("batch_id")
    return str(bid) if bid else None


def set_cached_batch(key: str, batch_id: str, base_url: str) -> None:
    data = _load_all()
    data[key] = {
        "batch_id": batch_id,
        "base_url": base_url.rstrip("/"),
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    _save_all(data)


def remove_cached_batch(key: str) -> None:
    data = _load_all()
    if key in data:
        del data[key]
        _save_all(data)
