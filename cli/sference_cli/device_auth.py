"""Device-flow (RFC 8628) client for ``sference auth login`` and token refresh.

Stdlib-only (json, urllib, time) so this module imports in ANY venv — the
mitmproxy addon (``_proxy_addon.py``) runs in mitmproxy's isolated tool venv
without ``sference_cli`` installed and sibling-imports this file to refresh an
expired device token mid-session. Keep it dependency-free.

Wire shapes come from the platform's device-flow endpoints (plan §4.3):
  POST /v1/oauth/device_code  -> {device_code, user_code, verification_uri, expires_in, interval}
  POST /v1/oauth/token        -> {access_token, token_type, expires_in, refresh_token}
                              or 400 {error, error_description}   (RFC 6749 shape)
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

# Registered on the platform; unknown client_ids get invalid_client.
CLIENT_ID_SFERENCE_CLI = "sference-cli"

GRANT_TYPE_DEVICE_CODE = "urn:ietf:params:oauth:grant-type:device_code"
GRANT_TYPE_REFRESH_TOKEN = "refresh_token"

# Refresh this far ahead of expiry so a token never dies mid-request.
EXPIRY_SKEW_SECONDS = 60


class DeviceAuthError(Exception):
    """A device-flow call failed. ``error`` is the RFC 6749 code when the
    server sent one (``invalid_grant``, ``expired_token``, …), else a local
    label (``network_error``, ``http_<status>``)."""

    def __init__(self, error: str, description: str) -> None:
        super().__init__(f"{error}: {description}")
        self.error = error
        self.description = description


def post_json(url: str, payload: dict[str, Any], timeout: float = 15.0) -> tuple[int, dict[str, Any]]:
    """POST a JSON body; return ``(status, parsed_json)`` for ANY status.

    The device-flow endpoints signal pending/slow_down/expired via 400 bodies,
    so HTTP errors are data, not exceptions. Network failures raise
    :class:`DeviceAuthError` — the caller cannot distinguish statuses it never
    received.
    """
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return exc.code, {}
    except (urllib.error.URLError, OSError) as exc:
        raise DeviceAuthError("network_error", str(exc)) from exc


def _api_root(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return root + "/v1"


def start_device_login(base_url: str, client_id: str, device_label: str | None = None) -> dict[str, Any]:
    """POST /v1/oauth/device_code; return the parsed DeviceCodeResponse."""
    payload: dict[str, Any] = {"client_id": client_id}
    if device_label:
        payload["device_label"] = device_label
    status, body = post_json(f"{_api_root(base_url)}/oauth/device_code", payload)
    if status != 200:
        raise DeviceAuthError(
            str(body.get("error") or f"http_{status}"),
            str(body.get("error_description") or f"device_code request failed ({status})"),
        )
    return body


def poll_for_tokens(
    base_url: str,
    client_id: str,
    device_code: str,
    *,
    interval: float,
    expires_in: float,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Poll /v1/oauth/token until the user approves; return the token response.

    Honors the RFC 8628 pacing contract: ``authorization_pending`` waits
    ``interval``; ``slow_down`` adds 5 s to it (the server also enforces pacing
    via ``last_polled_at`` — a client that ignores slow_down gets 400s).
    ``sleep``/``monotonic`` are injectable for tests.
    """
    url = f"{_api_root(base_url)}/oauth/token"
    deadline = monotonic() + expires_in
    wait = interval
    while True:
        status, body = post_json(
            url,
            {"grant_type": GRANT_TYPE_DEVICE_CODE, "client_id": client_id, "device_code": device_code},
        )
        if status == 200:
            return body
        error = str(body.get("error") or f"http_{status}")
        description = str(body.get("error_description") or f"token poll failed ({status})")
        if error == "authorization_pending":
            if monotonic() + wait > deadline:
                raise DeviceAuthError("expired_token", "device code expired before approval")
            sleep(wait)
            continue
        if error == "slow_down":
            wait += 5
            if monotonic() + wait > deadline:
                raise DeviceAuthError("expired_token", "device code expired before approval")
            sleep(wait)
            continue
        raise DeviceAuthError(error, description)


def refresh_tokens(base_url: str, client_id: str, refresh_token: str) -> dict[str, Any]:
    """POST /v1/oauth/token with grant_type=refresh_token; return the token response.

    The server rotates on every refresh: the returned ``refresh_token`` REPLACES
    the presented one, and presenting a rotated-out token revokes the whole
    grant (reuse detection). Callers must persist the new pair atomically —
    see ``main._write_device_credentials``.
    """
    status, body = post_json(
        f"{_api_root(base_url)}/oauth/token",
        {"grant_type": GRANT_TYPE_REFRESH_TOKEN, "client_id": client_id, "refresh_token": refresh_token},
    )
    if status != 200:
        raise DeviceAuthError(
            str(body.get("error") or f"http_{status}"),
            str(body.get("error_description") or f"refresh failed ({status})"),
        )
    return body
