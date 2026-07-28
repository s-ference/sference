"""Pure routing helpers for the Sference↔Claude Code mitmproxy addon.

Stdlib-only (json, urllib) so this module imports in ANY venv — including
mitmproxy's isolated tool venv, which does NOT contain ``sference_cli``. The
addon (``_proxy_addon.py``) loads it via a sibling-path import after the
launcher extracts both files to a temp dir. Tests import it directly from the
``sference_cli`` package.

Design: Sference has a native Anthropic Messages endpoint at ``/v1/messages``
(``apps/api/sference_api/inference/anthropic_controller.py``; auth accepts
``x-api-key`` or ``Authorization: Bearer``). So routing is a pure URL + auth
rewrite — the request body is passed through byte-for-byte untranslated and
the response is already native Anthropic. No Anthropic↔OpenAI translation.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlsplit, urlunsplit

# Decision labels returned by ``decide_routing``.
PASSTHROUGH = "passthrough"  # leave untouched, goes to the real upstream
SFERENCE = "sference"  # rewrite to the Sference /v1/messages endpoint
BOOTSTRAP = "bootstrap"  # Claude Code /api/claude_cli/bootstrap — inject models


def decide_routing(
    host: str,
    method: str,
    path: str,
    body: Any,
    sference_models: set[str],
) -> str:
    """Decide how to route a single request the addon sees.

    ``body`` is the parsed JSON of a POST body (or ``None`` if not JSON / not a
    dict). ``sference_models`` is the set of catalog ids this account can serve
    (live-fetched from ``GET /v1/models``). Returns one of ``PASSTHROUGH``,
    ``SFERENCE``, ``BOOTSTRAP``.
    """
    if (host or "") != "api.anthropic.com":
        return PASSTHROUGH
    method = (method or "").upper()
    if method == "GET" and "/api/claude_cli/bootstrap" in (path or ""):
        return BOOTSTRAP
    if method == "POST" and "/v1/messages" in (path or ""):
        model = body.get("model") if isinstance(body, dict) else None
        if isinstance(model, str) and model in sference_models:
            return SFERENCE
    return PASSTHROUGH


def strip_unsigned_thinking_blocks(body: Any) -> tuple[Any, int]:
    """Drop ``thinking`` blocks carrying an empty ``signature`` from a request body.

    Returns ``(body, dropped_count)``; the body is mutated in place and also
    returned for convenience.

    Anthropic verifies the ``signature`` on every ``thinking`` block it is sent
    and rejects the whole request with a 400 (``Invalid `signature` in
    `thinking` block``) if one does not validate. Sference's ``/v1/messages``
    emits ``signature: ""``, so once a Sference model has answered in a session,
    every later request routed to *Anthropic* replays that unsignable history
    and 400s.

    Claude Code already recovers by stripping the thinking blocks and resending,
    so this is not a correctness fix — it is a cost fix: it avoids re-uploading
    a large request body (~180KB in practice) and burning a round-trip on every
    turn after a model switch. We drop exactly what the client would have
    dropped, one step earlier.

    Only empty-signature blocks are removed; genuine Anthropic-minted
    signatures are left untouched so real Claude reasoning still replays.
    """
    if not isinstance(body, dict):
        return body, 0
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body, 0

    dropped = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        # Content may be a plain string (no blocks to inspect).
        if not isinstance(content, list):
            continue
        kept = [
            block
            for block in content
            if not (
                isinstance(block, dict)
                and block.get("type") == "thinking"
                and not block.get("signature")
            )
        ]
        # Never empty a message: Anthropic rejects a zero-length content array,
        # which would turn a recoverable 400 into one we caused. Leaving the
        # blocks in place costs us that request (Anthropic 400s on the bad
        # signature and Claude Code retries, exactly as it did before this fix)
        # rather than making things worse.
        if kept and len(kept) != len(content):
            dropped += len(content) - len(kept)
            message["content"] = kept
    return body, dropped


def _sference_origin(sference_base_url: str) -> str:
    """Normalize the Sference base URL to scheme + netloc (strip any ``/v1``).

    The Anthropic request path (``/v1/messages``) is preserved verbatim, so the
    base must contribute only the origin — otherwise we'd double the ``/v1``.
    """
    parts = urlsplit(sference_base_url.rstrip("/"))
    return f"{parts.scheme}://{parts.netloc}"


def rewrite_request_for_sference(
    url: str,
    sference_base_url: str,
    api_key: str,
) -> tuple[str, dict[str, str], list[str]]:
    """Rewrite an Anthropic request URL + auth to hit the Sference endpoint.

    Returns ``(new_url, headers_to_set, headers_to_remove)``. The request body
    is NOT touched — Sference speaks native Anthropic Messages, so the body
    passes through byte-for-byte. We swap the auth to the Sference API key
    (via ``x-api-key``, which Sference reads natively) and drop any stale
    Anthropic ``Authorization`` so it can't shadow the key.
    """
    origin = _sference_origin(sference_base_url)
    src = urlsplit(url)
    # Keep the Anthropic path + query (e.g. /v1/messages?beta=true). Sference's
    # controller accepts and ignores ?beta=true.
    new_url = urlunsplit((src.scheme, urlsplit(origin).netloc, src.path, src.query, src.fragment))
    headers_to_set = {"x-api-key": api_key}
    headers_to_remove = ["authorization"]
    return new_url, headers_to_set, headers_to_remove


def inject_models_into_bootstrap(body: Any, sference_models: set[str]) -> Any:
    """Append Sference models to a bootstrap response's ``additional_model_options``.

    Claude Code populates its ``/model`` picker from built-ins plus this array;
    each entry's ``model`` is sent verbatim on ``/v1/messages``, which the addon
    then routes to Sference. Best-effort: if the shape is wrong, return the body
    unchanged rather than crashing (the picker is a nicety, not a requirement).
    """
    if not isinstance(body, dict):
        return body
    options = body.get("additional_model_options")
    if not isinstance(options, list):
        options = []
    existing = {o.get("model") for o in options if isinstance(o, dict)}
    for model in sorted(sference_models):
        if model in existing:
            continue
        options.append(
            {
                "model": model,
                "name": f"[Sference] {model}",
                "description": "Sference (routed via local proxy)",
                "disabled_reason": None,
            }
        )
    body["additional_model_options"] = options
    return body


def parse_models_env(raw: str) -> set[str]:
    """Parse the ``SFERENCE_PROXY_MODELS`` env var (JSON list) into a set."""
    if not raw:
        return set()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return set()
    if isinstance(data, list):
        return {str(m) for m in data if isinstance(m, str) and m}
    return set()
