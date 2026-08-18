"""Unit tests for the RFC 8628 device-flow client (stdlib-only module)."""

from __future__ import annotations

import pytest

import sference_cli.device_auth as da


def _tokens() -> dict:
    return {
        "access_token": "jwt_x",
        "token_type": "bearer",
        "expires_in": 86400,
        "refresh_token": "rt_x",
    }


def test_poll_succeeds_after_pending(monkeypatch):
    calls: list[dict] = []
    responses = iter(
        [
            (400, {"error": "authorization_pending", "error_description": "not yet"}),
            (200, _tokens()),
        ]
    )
    monkeypatch.setattr(da, "post_json", lambda url, payload, **k: (calls.append(payload) or next(responses)))
    sleeps: list[float] = []
    out = da.poll_for_tokens(
        "https://api.sference.com",
        da.CLIENT_ID_SFERENCE_CLI,
        "dc",
        interval=5,
        expires_in=600,
        sleep=sleeps.append,
    )
    assert out["access_token"] == "jwt_x"
    assert sleeps == [5]
    assert calls[0]["grant_type"] == da.GRANT_TYPE_DEVICE_CODE
    assert calls[0]["device_code"] == "dc"


def test_poll_slow_down_extends_interval(monkeypatch):
    responses = iter(
        [
            (400, {"error": "slow_down", "error_description": "too fast"}),
            (400, {"error": "authorization_pending", "error_description": "not yet"}),
            (200, _tokens()),
        ]
    )
    monkeypatch.setattr(da, "post_json", lambda url, payload, **k: next(responses))
    sleeps: list[float] = []
    da.poll_for_tokens(
        "https://api.sference.com",
        da.CLIENT_ID_SFERENCE_CLI,
        "dc",
        interval=5,
        expires_in=600,
        sleep=sleeps.append,
    )
    # RFC 8628: slow_down adds 5s to the interval for ALL subsequent polls.
    assert sleeps == [10, 10]


def test_poll_expired_token_raises(monkeypatch):
    monkeypatch.setattr(
        da,
        "post_json",
        lambda url, payload, **k: (400, {"error": "expired_token", "error_description": "code expired"}),
    )
    with pytest.raises(da.DeviceAuthError) as excinfo:
        da.poll_for_tokens(
            "https://api.sference.com",
            da.CLIENT_ID_SFERENCE_CLI,
            "dc",
            interval=5,
            expires_in=600,
            sleep=lambda s: None,
        )
    assert excinfo.value.error == "expired_token"


def test_poll_pending_past_deadline_raises_expired(monkeypatch):
    monkeypatch.setattr(
        da,
        "post_json",
        lambda url, payload, **k: (400, {"error": "authorization_pending", "error_description": "not yet"}),
    )
    clock = iter([0.0, 1000.0])  # first poll at t=0, deadline check at t=1000 (> expires_in)
    with pytest.raises(da.DeviceAuthError) as excinfo:
        da.poll_for_tokens(
            "https://api.sference.com",
            da.CLIENT_ID_SFERENCE_CLI,
            "dc",
            interval=5,
            expires_in=600,
            sleep=lambda s: None,
            monotonic=lambda: next(clock),
        )
    assert excinfo.value.error == "expired_token"


def test_refresh_success(monkeypatch):
    seen: list[dict] = []
    monkeypatch.setattr(da, "post_json", lambda url, payload, **k: (seen.append(payload) or (200, _tokens())))
    out = da.refresh_tokens("https://api.sference.com", da.CLIENT_ID_SFERENCE_CLI, "rt_old")
    assert out["refresh_token"] == "rt_x"
    assert seen[0]["grant_type"] == da.GRANT_TYPE_REFRESH_TOKEN
    assert seen[0]["refresh_token"] == "rt_old"


def test_refresh_invalid_grant_raises(monkeypatch):
    monkeypatch.setattr(
        da,
        "post_json",
        lambda url, payload, **k: (400, {"error": "invalid_grant", "error_description": "revoked"}),
    )
    with pytest.raises(da.DeviceAuthError) as excinfo:
        da.refresh_tokens("https://api.sference.com", da.CLIENT_ID_SFERENCE_CLI, "rt_old")
    assert excinfo.value.error == "invalid_grant"
    assert "revoked" in excinfo.value.description


def test_start_device_login_success(monkeypatch):
    body = {
        "device_code": "dc",
        "user_code": "ABCD-EFGH",
        "verification_uri": "https://app.sference.com/device",
        "expires_in": 600,
        "interval": 5,
    }
    seen: list[dict] = []
    monkeypatch.setattr(da, "post_json", lambda url, payload, **k: (seen.append(payload) or (200, body)))
    out = da.start_device_login("https://api.sference.com", da.CLIENT_ID_SFERENCE_CLI, "my-laptop")
    assert out["user_code"] == "ABCD-EFGH"
    assert seen[0]["client_id"] == da.CLIENT_ID_SFERENCE_CLI
    assert seen[0]["device_label"] == "my-laptop"


def test_start_device_login_invalid_client_raises(monkeypatch):
    monkeypatch.setattr(
        da,
        "post_json",
        lambda url, payload, **k: (400, {"error": "invalid_client", "error_description": "unknown client_id"}),
    )
    with pytest.raises(da.DeviceAuthError) as excinfo:
        da.start_device_login("https://api.sference.com", "bogus-client")
    assert excinfo.value.error == "invalid_client"


def test_base_url_v1_suffix_not_doubled(monkeypatch):
    urls: list[str] = []
    monkeypatch.setattr(da, "post_json", lambda url, payload, **k: (urls.append(url) or (200, _tokens())))
    da.refresh_tokens("https://api.sference.com/v1", da.CLIENT_ID_SFERENCE_CLI, "rt")
    assert urls == ["https://api.sference.com/v1/oauth/token"]
