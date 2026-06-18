"""Shared helpers for sference integration examples."""

from __future__ import annotations

import os
import sys
import time
from typing import Any

from sference_sdk import SferenceClient
from sference_sdk.models import InferenceRequest

DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
COMPLETION_WINDOW = "24h"
BATCH_POLL_INTERVAL_S = float(os.environ.get("SFERENCE_BATCH_POLL_INTERVAL_S", "5.0"))
# Demo scripts poll until terminal; raise for real 24h windows in production.
BATCH_WAIT_TIMEOUT_S = float(os.environ.get("SFERENCE_BATCH_WAIT_TIMEOUT_S", "86400.0"))


def require_api_key() -> None:
    if not os.getenv("SFERENCE_API_KEY"):
        print("Error: SFERENCE_API_KEY is not set", file=sys.stderr)
        sys.exit(1)


def model_id() -> str:
    return os.environ.get("SFERENCE_MODEL", DEFAULT_MODEL)


def chat_batch_request(
    *,
    custom_id: str,
    user_content: str,
    model: str | None = None,
    system_content: str | None = None,
    temperature: float | None = None,
) -> InferenceRequest:
    messages: list[dict[str, str]] = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})
    body: dict[str, Any] = {"model": model or model_id(), "messages": messages}
    if temperature is not None:
        body["temperature"] = temperature
    return InferenceRequest(
        custom_id=custom_id,
        body=body,
    )


def wait_for_batch_terminal(
    client: SferenceClient,
    batch_id: str,
    *,
    poll_interval: float = BATCH_POLL_INTERVAL_S,
    timeout: float = BATCH_WAIT_TIMEOUT_S,
) -> Any:
    """Poll GET /v1/batches/{id} until status is terminal."""
    deadline = time.time() + timeout
    while True:
        batch = client.get_batch(batch_id)
        if batch.status in ("completed", "failed", "cancelled"):
            return batch
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for batch {batch_id}")
        time.sleep(poll_interval)


def completion_text_from_row(row: dict[str, Any]) -> str:
    """Extract assistant text from a batch results row."""
    if row.get("status") != "completed":
        err = row.get("error_json")
        return f"[{row.get('status')}] {err}" if err else f"[{row.get('status')}]"
    result = row.get("result_json") or {}
    choices = result.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    return content if isinstance(content, str) else str(content or "")


def index_results_by_custom_id(results: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in results or []:
        cid = row.get("custom_id")
        if cid is not None:
            indexed[str(cid)] = row
    return indexed
