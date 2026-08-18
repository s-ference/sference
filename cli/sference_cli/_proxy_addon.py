"""mitmproxy addon: route Sference models to api.sference.com, pass Claude through.

Loaded by ``mitmdump -s <this file>``. Runs inside mitmproxy's isolated tool
venv, which does NOT contain ``sference_cli`` — so we import the pure routing
helpers via a sibling path (the launcher extracts this file,
``_proxy_routing.py`` and ``device_auth.py`` to the same temp dir), not via
``from sference_cli…``.

Config comes from env (set by the launcher):
  SFERENCE_PROXY_MODELS      JSON list of routable catalog ids (text-generation)
  SFERENCE_API_KEY           Sference access token (sent as x-api-key)
  SFERENCE_BASE_URL          Sference API origin, e.g. https://api.sference.com
  SFERENCE_REFRESH_TOKEN     device-flow refresh token (optional; enables 401 refresh)
  SFERENCE_CREDENTIALS_PATH  ~/.sference/credentials.json (optional; refresh target)

Routing is a pure URL + auth rewrite; the request body passes through
byte-for-byte (Sference has a native /v1/messages endpoint) and the response is
already native Anthropic — no translation in either direction.

Device-flow access tokens live 24 h, so a long Claude session can outlive its
token. When a Sference-routed request comes back 401 and a refresh token is
configured, the addon refreshes (re-reading the credentials file first — a
sibling CLI/proxy may already have rotated it), persists the rotated pair,
and replays the request once with the new token.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from mitmproxy import ctx, http

# Sibling imports: _proxy_routing.py and device_auth.py sit next to this file
# in the temp dir.
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
from device_auth import CLIENT_ID_SFERENCE_CLI, DeviceAuthError, refresh_tokens  # noqa: E402

SFERENCE_BASE_URL = os.environ.get("SFERENCE_BASE_URL", "https://api.sference.com").rstrip("/")
SFERENCE_API_KEY = os.environ.get("SFERENCE_API_KEY", "")
SFERENCE_MODELS = parse_models_env(os.environ.get("SFERENCE_PROXY_MODELS", ""))
SFERENCE_REFRESH_TOKEN = os.environ.get("SFERENCE_REFRESH_TOKEN", "")
SFERENCE_CREDENTIALS_PATH = os.environ.get("SFERENCE_CREDENTIALS_PATH", "")

# flow.id -> original model (for response handling — currently a no-op since
# the Sference response is already native Anthropic, but kept for clarity).
_sference_flows: dict[str, str] = {}
# flow.ids of bootstrap GETs whose response we post-process.
_bootstrap_flows: set[str] = set()


def _read_file_credentials() -> dict:
    """Best-effort read of the v2 credentials file; {} on any problem."""
    if not SFERENCE_CREDENTIALS_PATH:
        return {}
    try:
        data = json.loads(Path(SFERENCE_CREDENTIALS_PATH).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_file_credentials(access_token: str, refresh_token: str, expires_in: float) -> None:
    """Persist the rotated pair (0700 dir / 0600 file, matching the CLI)."""
    if not SFERENCE_CREDENTIALS_PATH:
        return
    path = Path(SFERENCE_CREDENTIALS_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "expires_at": time.time() + float(expires_in),
                }
            ),
            encoding="utf-8",
        )
        os.chmod(path, 0o600)
    except OSError as exc:
        ctx.log.warn(f"sference: could not persist refreshed credentials: {exc}")


def _refresh_access_token() -> str | None:
    """Return a fresh access token, or None when refresh is impossible/failed.

    Re-reads the credentials file BEFORE refreshing: a sibling CLI command or
    proxy may already have rotated the grant (refresh tokens are single-use —
    presenting a rotated-out one revokes the whole grant server-side). Only
    when the file still holds OUR refresh token do we rotate it ourselves.
    """
    global SFERENCE_API_KEY, SFERENCE_REFRESH_TOKEN

    file_creds = _read_file_credentials()
    file_access = file_creds.get("access_token")
    file_refresh = file_creds.get("refresh_token")
    if isinstance(file_access, str) and file_access and file_access != SFERENCE_API_KEY:
        # Someone else refreshed since we launched; adopt the file's token.
        ctx.log.info("sference: adopted fresher access token from credentials file")
        SFERENCE_API_KEY = file_access
        if isinstance(file_refresh, str) and file_refresh:
            SFERENCE_REFRESH_TOKEN = file_refresh
        return file_access

    refresh_token = file_refresh if isinstance(file_refresh, str) and file_refresh else SFERENCE_REFRESH_TOKEN
    if not refresh_token:
        return None
    try:
        tokens = refresh_tokens(SFERENCE_BASE_URL, CLIENT_ID_SFERENCE_CLI, refresh_token)
    except DeviceAuthError as exc:
        ctx.log.warn(f"sference: token refresh failed ({exc.description}); re-run `sference auth login`")
        return None
    SFERENCE_API_KEY = str(tokens["access_token"])
    SFERENCE_REFRESH_TOKEN = str(tokens["refresh_token"])
    _write_file_credentials(SFERENCE_API_KEY, SFERENCE_REFRESH_TOKEN, float(tokens["expires_in"]))
    ctx.log.info("sference: refreshed the device access token after a 401")
    return SFERENCE_API_KEY


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
                    # set_text() recomputes content-length itself (correctly for
                    # any content-encoding); do NOT touch the header afterwards.
                    flow.request.set_text(json.dumps(body))
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
            if flow.response.status_code == 401:
                self._maybe_refresh_and_replay(flow)
            # Native Anthropic response: pass through untouched.
            return

    def _maybe_refresh_and_replay(self, flow: "http.HTTPFlow") -> None:
        """On a Sference 401, refresh the device token and replay the request once.

        Guards: only with a refresh token configured (device-flow credentials),
        only once per flow (a replayed request that 401s again surfaces the 401
        to the client), and only when the mitmproxy version supports replay.
        """
        if not SFERENCE_REFRESH_TOKEN and not SFERENCE_CREDENTIALS_PATH:
            return
        if flow.metadata.get("sference_auth_retried"):
            return
        if not hasattr(flow, "replay"):
            ctx.log.warn("sference: 401 but this mitmproxy version cannot replay flows")
            return
        new_token = _refresh_access_token()
        if not new_token:
            return
        flow.metadata["sference_auth_retried"] = True
        flow.request.headers["x-api-key"] = new_token
        ctx.log.info("sference: replaying request with refreshed token")
        flow.replay()

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
