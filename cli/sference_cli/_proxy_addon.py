"""mitmproxy addon: route Sference models to api.sference.com, pass Claude through.

Loaded by ``mitmdump -s <this file>``. Runs inside mitmproxy's isolated tool
venv, which does NOT contain ``sference_cli`` — so we import the pure routing
helpers via a sibling path (the launcher extracts this file and
``_proxy_routing.py`` to the same temp dir), not via ``from sference_cli…``.

Config comes from env (set by the launcher):
  SFERENCE_PROXY_MODELS  JSON list of routable catalog ids (text-generation)
  SFERENCE_API_KEY       Sference API key (sent as x-api-key)
  SFERENCE_BASE_URL      Sference API origin, e.g. https://api.sference.com

Routing is a pure URL + auth rewrite; the request body passes through
byte-for-byte (Sference has a native /v1/messages endpoint) and the response is
already native Anthropic — no translation in either direction.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from mitmproxy import ctx, http

# Sibling import: _proxy_routing.py sits next to this file in the temp dir.
sys.path.insert(0, os.path.dirname(str(Path(__file__).resolve())))
from _proxy_routing import (  # noqa: E402
    BOOTSTRAP,
    PASSTHROUGH,
    SFERENCE,
    decide_routing,
    inject_models_into_bootstrap,
    parse_models_env,
    rewrite_request_for_sference,
    strip_unsigned_thinking_blocks,
)

SFERENCE_BASE_URL = os.environ.get("SFERENCE_BASE_URL", "https://api.sference.com").rstrip("/")
SFERENCE_API_KEY = os.environ.get("SFERENCE_API_KEY", "")
SFERENCE_MODELS = parse_models_env(os.environ.get("SFERENCE_PROXY_MODELS", ""))

# flow.id -> original model (for response handling — currently a no-op since
# the Sference response is already native Anthropic, but kept for clarity).
_sference_flows: dict[str, str] = {}
# flow.ids of bootstrap GETs whose response we post-process.
_bootstrap_flows: set[str] = set()


class SferenceRouter:
    def request(self, flow: "http.HTTPFlow") -> None:
        if flow.request is None:
            return
        host = flow.request.pretty_host
        method = flow.request.method
        path = flow.request.path

        body: object = None
        if method == "POST":
            try:
                body = json.loads(flow.request.get_text())
            except Exception:
                body = None

        decision = decide_routing(host, method, path, body, SFERENCE_MODELS)
        if decision == PASSTHROUGH:
            # Anthropic rejects thinking blocks whose signature it can't verify,
            # and Sference emits an empty one — so a session that switched away
            # from a Sference model replays unsignable history and 400s. Claude
            # Code recovers by stripping and resending; do it here instead and
            # save the wasted upload + round-trip. Sference-bound requests keep
            # the blocks (byte-for-byte passthrough).
            if (
                host == "api.anthropic.com"
                and method == "POST"
                and "/v1/messages" in (path or "")
            ):
                body, dropped = strip_unsigned_thinking_blocks(body)
                if dropped:
                    new_body = json.dumps(body)
                    flow.request.set_text(new_body)
                    flow.request.headers["content-length"] = str(len(new_body))
                    ctx.log.info(
                        f"sference: stripped {dropped} unsigned thinking block(s) "
                        "from Anthropic-bound request"
                    )
            return
        if decision == BOOTSTRAP:
            _bootstrap_flows.add(flow.id)
            return
        # SFERENCE
        model = body.get("model") if isinstance(body, dict) else None
        ctx.log.info(f"sference: routing model={model!r} -> Sference")
        _sference_flows[flow.id] = str(model)
        new_url, set_headers, remove_headers = rewrite_request_for_sference(
            flow.request.url, SFERENCE_BASE_URL, SFERENCE_API_KEY
        )
        flow.request.url = new_url
        for key, value in set_headers.items():
            flow.request.headers[key] = value
        for key in remove_headers:
            flow.request.headers.pop(key, None)

    def response(self, flow: "http.HTTPFlow") -> None:
        if flow.response is None:
            return
        if flow.id in _bootstrap_flows:
            _bootstrap_flows.discard(flow.id)
            self._inject_models(flow)
            return
        if flow.id in _sference_flows:
            _sference_flows.pop(flow.id)
            # Native Anthropic response: pass through untouched.
            return

    def _inject_models(self, flow: "http.HTTPFlow") -> None:
        if flow.response.status_code != 200:
            return
        try:
            body = json.loads(flow.response.get_text())
        except Exception:
            return
        before = len(body.get("additional_model_options", []) if isinstance(body, dict) else [])
        body = inject_models_into_bootstrap(body, SFERENCE_MODELS)
        flow.response.set_text(json.dumps(body))
        # Body length changed; drop stale framing so mitmproxy recomputes it.
        flow.response.headers.pop("content-length", None)
        flow.response.headers.pop("Content-Length", None)
        after = len(body.get("additional_model_options", []) if isinstance(body, dict) else [])
        ctx.log.info(f"sference: bootstrap injected {after - before} model(s) into /model picker")


addons = [SferenceRouter()]
