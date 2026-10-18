"""Claude Code gateway model discovery support for ``sference launch claude``.

Claude Code populates ``/model`` from built-ins plus gateway discovery results
(``GET {ANTHROPIC_BASE_URL}/v1/models``), cached to
``~/.claude/cache/gateway-models.json``. Two client-side gates matter here:

* discovery only runs when ``CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1``
  and the base URL is not api.anthropic.com;
* the live fetch keeps only ids containing "claude" or "anthropic" — Sference
  catalog ids (``deepseek-ai/...``) match neither, so the fetch returns "0
  usable models" and leaves the cache untouched. The picker then falls back to
  whatever the cache already holds (verified on v2.1.302: the prime read is
  gated only on the flag + base URL, not on credentials).

So the launcher primes the cache with the live catalog before exec'ing claude.
Stdlib-only so this module imports anywhere.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

DISCOVERY_ENV_KEY = "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"

# Unknown (non-claude) model ids are believed to hold 200k tokens and
# auto-compact against that; Sference catalog models serve 1M. Disabling the
# unknown-model window enforcement lets CLAUDE_CODE_AUTO_COMPACT_WINDOW apply.
WINDOW_ENFORCEMENT_ENV_KEY = "CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT"
AUTO_COMPACT_WINDOW_ENV_KEY = "CLAUDE_CODE_AUTO_COMPACT_WINDOW"
DEFAULT_AUTO_COMPACT_WINDOW = "950000"


def fetch_sference_model_entries(base_url: str, api_key: str) -> list[dict[str, Any]]:
    """Pull the live text-generation model entries from GET {base_url}/v1/models."""
    api_root = base_url.rstrip("/") + "/v1"
    req = urllib.request.Request(
        f"{api_root}/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return [
        item
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id") and item.get("modality", "text_generation") == "text_generation"
    ]


def picker_cache_path() -> Path:
    """The discovery cache Claude Code primes its picker from (CLAUDE_CONFIG_DIR-aware)."""
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR", "").strip()
    if config_dir:
        return Path(config_dir) / "cache" / "gateway-models.json"
    return Path.home() / ".claude" / "cache" / "gateway-models.json"


def _picker_entry(item: dict[str, Any]) -> dict[str, str]:
    entry = {"id": str(item["id"])}
    name = str(item.get("display_name") or item["id"])
    entry["display_name"] = name if "sference" in name.lower() else f"{name} via Sference"
    ctx = item.get("context_tokens")
    if isinstance(ctx, int) and ctx > 0:
        entry["description"] = f"Sference inference — {ctx // 1000}k context"
    else:
        entry["description"] = "Sference inference"
    return entry


def prime_picker_cache(entries: list[dict[str, Any]], base_url: str) -> Path:
    """Write the picker cache atomically (temp+rename, 0600): a Claude Code
    starting mid-prime must never read a torn file."""
    models = [_picker_entry(item) for item in entries]
    body = {
        "baseUrl": base_url.rstrip("/"),
        "fetchedAt": int(time.time() * 1000),
        "models": models,
    }
    path = picker_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".gateway-models-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:  # mkstemp is already 0600
            f.write(json.dumps(body) + "\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path
