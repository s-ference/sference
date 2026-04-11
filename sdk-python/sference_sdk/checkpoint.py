from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _store_path() -> Path:
    raw = os.environ.get("SFERENCE_STREAM_CHECKPOINTS")
    if raw:
        return Path(raw)
    return Path.home() / ".sference" / "stream_checkpoints.json"


def _checkpoint_key(base_url: str, stream_id: str, consumer_name: str) -> str:
    return f"{base_url.rstrip('/')}:{stream_id}:{consumer_name}"


def _load_all(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _write_all(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_checkpoint(base_url: str, stream_id: str, consumer_name: str) -> str | None:
    path = _store_path()
    key = _checkpoint_key(base_url, stream_id, consumer_name)
    entry = _load_all(path).get(key)
    if not isinstance(entry, dict):
        return None
    last = entry.get("last_completion_id") or entry.get("last_event_id")
    return str(last) if last else None


def save_checkpoint(base_url: str, stream_id: str, consumer_name: str, last_completion_id: str) -> None:
    path = _store_path()
    key = _checkpoint_key(base_url, stream_id, consumer_name)
    data = _load_all(path)
    data[key] = {
        "last_completion_id": last_completion_id,
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    _write_all(path, data)


def clear_checkpoint(base_url: str, stream_id: str, consumer_name: str) -> None:
    path = _store_path()
    key = _checkpoint_key(base_url, stream_id, consumer_name)
    data = _load_all(path)
    data.pop(key, None)
    _write_all(path, data)
