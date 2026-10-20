"""Tests for ``sference launch cursor`` (Cursor BYOK configuration)."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

import sference_cli.cursor_config as cc
import sference_cli.launch as launch_mod
import sference_cli.main as cli_main

runner = CliRunner()

REACTIVE = cc.REACTIVE_KEY
FAKE_KEY = "sk_test_sference_key"
FAKE_JWT = "header.payload.signature"
DEFAULT_MODEL = launch_mod.CURSOR_DEFAULT_LAUNCH_MODEL

FAKE_CATALOG = [
    {"id": "moonshotai/Kimi-K3", "display_name": "Kimi K3", "modality": "text_generation"},
    {"id": "zai-org/GLM-5.3", "display_name": "GLM 5.3", "modality": "text_generation"},
]


def make_cursor_home(tmp_path: Path, *, reactive: dict | None, rows: dict[str, str] | None = None) -> Path:
    """Create a fake Cursor globalStorage with a state.vscdb; return its dir."""
    config_dir = tmp_path / "Cursor" / "User" / "globalStorage"
    config_dir.mkdir(parents=True)
    conn = sqlite3.connect(str(config_dir / "state.vscdb"))
    conn.execute("CREATE TABLE ItemTable (key TEXT UNIQUE ON CONFLICT REPLACE, value BLOB)")
    if reactive is not None:
        conn.execute(
            "INSERT INTO ItemTable (key, value) VALUES (?, ?)",
            (REACTIVE, json.dumps(reactive, separators=(",", ":"))),
        )
    for key, value in (rows or {}).items():
        conn.execute("INSERT INTO ItemTable (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()
    return config_dir


def signed_in_rows(**overrides) -> dict[str, str]:
    rows = {
        cc.ACCESS_TOKEN_ROW: "cursor-access-token",
        cc.MEMBERSHIP_ROW: "pro",
    }
    rows.update(overrides)
    return rows


def read_db(config_dir: Path) -> dict[str, str]:
    conn = sqlite3.connect(str(config_dir / "state.vscdb"))
    out = dict(conn.execute("SELECT key, value FROM ItemTable").fetchall())
    conn.close()
    return out


def reactive_of(config_dir: Path) -> dict:
    return json.loads(read_db(config_dir)[REACTIVE])


@pytest.fixture
def cursor_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A signed-in, paid-plan Cursor home with no saved key; not running."""
    config_dir = make_cursor_home(
        tmp_path,
        reactive={
            "openAIBaseUrl": "https://api.openai.com/v1",
            "useOpenAIKey": False,
            "aiSettings": {
                "userAddedModels": ["my-other-custom-model"],
                "modelOverrideEnabled": ["my-other-custom-model"],
                "modelOverrideDisabled": ["gpt-5.2"],
                "modelConfig": {
                    "composer": {
                        "modelName": "grok-4.7",
                        "maxMode": False,
                        "selectedModels": [{"modelId": "grok-4.7", "parameters": []}],
                    }
                },
            },
            "availableDefaultModels2": [
                {"name": "default", "defaultOn": True},
                {"name": "grok-4.7", "defaultOn": True},
                {"name": "gpt-5.2", "defaultOn": False},
                {"name": "zai-org/GLM-5.2", "isUserAdded": True},
            ],
        },
        rows=signed_in_rows(),
    )
    monkeypatch.setattr(cc, "default_config_dir", lambda: config_dir)
    monkeypatch.setattr(launch_mod.cursor_config, "default_config_dir", lambda: config_dir)
    monkeypatch.setattr(cc, "_keychain_item_present", lambda: False)
    monkeypatch.setattr(cc, "_keychain_delete", lambda: False)
    # Cursor is not running (no code.lock) and the backup lives in tmp HOME.
    monkeypatch.setattr(cc, "backup_path", lambda: tmp_path / ".sference" / "cursor" / "backup.json")
    monkeypatch.setattr(launch_mod.cursor_config, "backup_path", lambda: tmp_path / ".sference" / "cursor" / "backup.json")
    # Network: live catalog fetch and control-plane calls are fakes.
    monkeypatch.setattr(
        launch_mod, "fetch_sference_model_entries", lambda base, key: list(FAKE_CATALOG)
    )
    monkeypatch.setattr(cli_main, "_read_device_credentials", lambda: None)
    monkeypatch.setenv("SFERENCE_API_KEY", FAKE_KEY)
    return config_dir


def invoke(args: list[str]):
    return runner.invoke(cli_main.app, args)


# --------------------------------------------------------------------------- on


def test_on_writes_config_and_backup(cursor_home: Path):
    result = invoke(["launch", "cursor"])
    assert result.exit_code == 0, result.output
    reactive = reactive_of(cursor_home)
    assert reactive["openAIBaseUrl"] == "https://api.sference.com/v1"
    assert reactive["useOpenAIKey"] is True
    ai = reactive["aiSettings"]
    for model in ("moonshotai/Kimi-K3", "zai-org/GLM-5.3"):
        assert model in ai["userAddedModels"]
        assert model in ai["modelOverrideEnabled"]
        assert model not in ai["modelOverrideDisabled"]
    # The user's own custom model and other overrides survive.
    assert "my-other-custom-model" in ai["userAddedModels"]
    assert "gpt-5.2" in ai["modelOverrideDisabled"]
    # New chats default to the Sference model; maxMode is preserved.
    assert ai["modelConfig"]["composer"]["modelName"] == DEFAULT_MODEL
    assert ai["modelConfig"]["composer"]["selectedModels"] == [
        {"modelId": DEFAULT_MODEL, "parameters": []}
    ]
    assert ai["modelConfig"]["composer"]["maxMode"] is False
    # The key row is our sk_ credential; no key was provisioned over HTTP.
    assert read_db(cursor_home)[cc.OPENAI_KEY_ROW] == FAKE_KEY
    backup = cc.read_backup()
    assert backup is not None
    assert backup["reactive"]["openAIBaseUrl"] == {"present": True, "value": "https://api.openai.com/v1"}
    assert backup["reactive"]["useOpenAIKey"] == {"present": True, "value": False}
    assert backup["reactive"]["aiSettings"]["modelOverrideDisabled"] == ["gpt-5.2"]
    assert backup["openai_key_row"]["existed"] is False
    assert backup["openai_key_row"]["written_sha256"] == cc.sha256_text(FAKE_KEY)
    assert backup["provisioned_key"] is None


def test_on_idempotent_second_run(cursor_home: Path):
    first = invoke(["launch", "cursor"])
    assert first.exit_code == 0, first.output
    second = invoke(["launch", "cursor"])
    assert second.exit_code == 0, second.output
    reactive = reactive_of(cursor_home)
    assert reactive["aiSettings"]["userAddedModels"].count("moonshotai/Kimi-K3") == 1
    assert reactive["aiSettings"]["modelOverrideEnabled"].count("moonshotai/Kimi-K3") == 1


def test_on_refuses_when_cursor_running(cursor_home: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cc, "cursor_running_pid", lambda _dir: os.getpid())
    result = invoke(["launch", "cursor"])
    assert result.exit_code == 1
    assert "Cursor is running" in (result.stderr or result.output)
    # Nothing was written.
    assert reactive_of(cursor_home)["useOpenAIKey"] is False
    assert cc.read_backup() is None


def test_on_force_writes_when_running(cursor_home: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cc, "cursor_running_pid", lambda _dir: os.getpid())
    result = invoke(["launch", "cursor", "--force"])
    assert result.exit_code == 0, result.output
    assert reactive_of(cursor_home)["useOpenAIKey"] is True
    assert "fully restart" in result.output


def test_on_refuses_free_plan(cursor_home: Path):
    rows = read_db(cursor_home)
    conn = cc.connect(cursor_home)
    conn.execute("INSERT OR REPLACE INTO ItemTable (key, value) VALUES (?, ?)", (cc.MEMBERSHIP_ROW, "free"))
    conn.commit()
    conn.close()
    result = invoke(["launch", "cursor"])
    assert result.exit_code == 1
    assert "paid Cursor plan" in (result.stderr or result.output)
    assert rows[REACTIVE] == read_db(cursor_home)[REACTIVE]


def test_on_refuses_signed_out(cursor_home: Path):
    conn = cc.connect(cursor_home)
    conn.execute("DELETE FROM ItemTable WHERE key = ?", (cc.ACCESS_TOKEN_ROW,))
    conn.commit()
    conn.close()
    result = invoke(["launch", "cursor"])
    assert result.exit_code == 1
    assert "not signed in" in (result.stderr or result.output)


def test_on_refuses_foreign_plaintext_key(cursor_home: Path):
    conn = cc.connect(cursor_home)
    conn.execute(
        "INSERT OR REPLACE INTO ItemTable (key, value) VALUES (?, ?)", (cc.OPENAI_KEY_ROW, "sk-openai-personal")
    )
    conn.commit()
    conn.close()
    result = invoke(["launch", "cursor"])
    assert result.exit_code == 1
    out = result.stderr or result.output
    assert "already has an OpenAI API key" in out
    assert "--replace-existing-key" in out
    assert read_db(cursor_home)[cc.OPENAI_KEY_ROW] == "sk-openai-personal"


def test_on_replace_existing_key_overwrites(cursor_home: Path):
    conn = cc.connect(cursor_home)
    conn.execute(
        "INSERT OR REPLACE INTO ItemTable (key, value) VALUES (?, ?)", (cc.OPENAI_KEY_ROW, "sk-openai-personal")
    )
    conn.execute(
        "INSERT OR REPLACE INTO ItemTable (key, value) VALUES (?, ?)", (cc.SECRET_KEY_ROW, "v10-encrypted")
    )
    conn.commit()
    conn.close()
    result = invoke(["launch", "cursor", "--replace-existing-key"])
    assert result.exit_code == 0, result.output
    rows = read_db(cursor_home)
    assert rows[cc.OPENAI_KEY_ROW] == FAKE_KEY
    assert cc.SECRET_KEY_ROW not in rows
    backup = cc.read_backup()
    assert backup["openai_key_row"]["existed"] is True
    assert backup["secret_row_deleted"] is True


def test_on_refuses_secret_row_without_flag(cursor_home: Path):
    conn = cc.connect(cursor_home)
    conn.execute(
        "INSERT OR REPLACE INTO ItemTable (key, value) VALUES (?, ?)", (cc.SECRET_KEY_ROW, "v10-encrypted")
    )
    conn.commit()
    conn.close()
    result = invoke(["launch", "cursor"])
    assert result.exit_code == 1
    assert "encrypted settings store" in (result.stderr or result.output)


def test_on_refuses_keychain_item_without_flag(cursor_home: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cc, "_keychain_item_present", lambda: True)
    result = invoke(["launch", "cursor"])
    assert result.exit_code == 1
    assert "macOS Keychain" in (result.stderr or result.output)


def test_on_deletes_keychain_item_with_flag(cursor_home: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cc, "_keychain_item_present", lambda: True)
    monkeypatch.setattr(cc, "_keychain_delete", lambda: True)
    result = invoke(["launch", "cursor", "--replace-existing-key"])
    assert result.exit_code == 0, result.output
    assert cc.read_backup()["keychain_item_deleted"] is True


def test_on_aborts_when_keychain_delete_fails(cursor_home: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cc, "_keychain_item_present", lambda: True)
    monkeypatch.setattr(cc, "_keychain_delete", lambda: False)
    result = invoke(["launch", "cursor", "--replace-existing-key"])
    assert result.exit_code == 1
    assert "could not remove" in (result.stderr or result.output)
    assert reactive_of(cursor_home)["useOpenAIKey"] is False
    assert cc.read_backup() is None


def test_on_sference_only_disables_catalog_models(cursor_home: Path):
    result = invoke(["launch", "cursor", "--sference-only"])
    assert result.exit_code == 0, result.output
    disabled = reactive_of(cursor_home)["aiSettings"]["modelOverrideDisabled"]
    assert "grok-4.7" in disabled
    assert "default" in disabled
    assert "gpt-5.2" in disabled
    assert "moonshotai/Kimi-K3" not in disabled
    # User-added entries are never disabled by the mode.
    assert "zai-org/GLM-5.2" not in disabled


def test_on_sference_only_requires_catalog(cursor_home: Path):
    conn = cc.connect(cursor_home)
    reactive = cursor_home and cc.read_reactive(conn)
    reactive.pop("availableDefaultModels2")
    conn.execute(
        "INSERT OR REPLACE INTO ItemTable (key, value) VALUES (?, ?)",
        (REACTIVE, json.dumps(reactive, separators=(",", ":"))),
    )
    conn.commit()
    conn.close()
    result = invoke(["launch", "cursor", "--sference-only"])
    assert result.exit_code == 1
    assert "cached model catalog" in (result.stderr or result.output)


def test_on_models_flag_skips_catalog_fetch(cursor_home: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        launch_mod,
        "fetch_sference_model_entries",
        lambda *a: (_ for _ in ()).throw(AssertionError("must not fetch")),
    )
    result = invoke(["launch", "cursor", "--models", "a/A,b/B", "--model", "a/A"])
    assert result.exit_code == 0, result.output
    ai = reactive_of(cursor_home)["aiSettings"]
    assert set(ai["userAddedModels"]) == {"my-other-custom-model", "a/A", "b/B"}
    assert ai["modelConfig"]["composer"]["modelName"] == "a/A"


def test_on_catalog_fetch_failure_falls_back_to_default_model(cursor_home: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        launch_mod, "fetch_sference_model_entries", lambda *a: (_ for _ in ()).throw(OSError("down"))
    )
    result = invoke(["launch", "cursor"])
    assert result.exit_code == 0, result.output
    ai = reactive_of(cursor_home)["aiSettings"]
    assert DEFAULT_MODEL in ai["userAddedModels"]


def test_dry_run_writes_nothing(cursor_home: Path):
    before = read_db(cursor_home)
    result = invoke(["launch", "cursor", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "dry run: nothing was written" in result.output
    assert read_db(cursor_home) == before
    assert cc.read_backup() is None


def test_on_missing_cursor_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cc, "default_config_dir", lambda: tmp_path / "nope")
    monkeypatch.setattr(launch_mod.cursor_config, "default_config_dir", lambda: tmp_path / "nope")
    monkeypatch.setenv("SFERENCE_API_KEY", FAKE_KEY)
    result = invoke(["launch", "cursor"])
    assert result.exit_code == 1
    assert "Cursor settings not found" in (result.stderr or result.output)


# ------------------------------------------------------------------ key source


def test_on_provisions_key_for_device_login(cursor_home: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SFERENCE_API_KEY", raising=False)
    monkeypatch.setattr(cli_main, "_read_token", lambda: FAKE_JWT)
    calls: list[dict] = []

    def fake_control(base, method, path, token, payload=None):
        calls.append({"method": method, "path": path, "token": token, "payload": payload})
        return 201, {
            "id": "11111111-1111-1111-1111-111111111111",
            "name": payload["name"],
            "plaintext": "sk_provisioned",
            "prefix": "sk_prov",
            "created_at": "2026-10-20T00:00:00Z",
        }

    monkeypatch.setattr(launch_mod, "control_request", fake_control)
    result = invoke(["launch", "cursor"])
    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "method": "POST",
            "path": "/control/v1/auth/api-keys",
            "token": FAKE_JWT,
            "payload": {"name": calls[0]["payload"]["name"], "tag": launch_mod.CURSOR_API_KEY_TAG},
        }
    ]
    assert calls[0]["payload"]["name"].startswith("sference-cli cursor (")
    assert read_db(cursor_home)[cc.OPENAI_KEY_ROW] == "sk_provisioned"
    backup = cc.read_backup()
    assert backup["provisioned_key"]["id"] == "11111111-1111-1111-1111-111111111111"
    assert backup["provisioned_key"]["tag"] == launch_mod.CURSOR_API_KEY_TAG


def test_on_provision_failure_exits(cursor_home: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SFERENCE_API_KEY", raising=False)
    monkeypatch.setattr(cli_main, "_read_token", lambda: FAKE_JWT)
    monkeypatch.setattr(launch_mod, "control_request", lambda *a, **k: (403, {"detail": "nope"}))
    result = invoke(["launch", "cursor"])
    assert result.exit_code == 1
    assert "could not create an API key" in (result.stderr or result.output)
    assert cc.read_backup() is None


def test_on_explicit_api_key_wins(cursor_home: Path):
    result = invoke(["launch", "cursor", "--api-key", "sk_explicit"])
    assert result.exit_code == 0, result.output
    assert read_db(cursor_home)[cc.OPENAI_KEY_ROW] == "sk_explicit"
    assert "--api-key" in result.output


# -------------------------------------------------------------------------- off


def test_off_restores_prior_state_verbatim(cursor_home: Path):
    before = read_db(cursor_home)
    assert invoke(["launch", "cursor"]).exit_code == 0
    result = invoke(["launch", "cursor", "off"])
    assert result.exit_code == 0, result.output
    after = read_db(cursor_home)
    assert after[REACTIVE] == before[REACTIVE]
    assert cc.OPENAI_KEY_ROW not in after
    assert cc.read_backup() is None


def test_off_keeps_user_replaced_key(cursor_home: Path):
    assert invoke(["launch", "cursor"]).exit_code == 0
    conn = cc.connect(cursor_home)
    conn.execute(
        "INSERT OR REPLACE INTO ItemTable (key, value) VALUES (?, ?)", (cc.OPENAI_KEY_ROW, "sk-user-new-choice")
    )
    conn.commit()
    conn.close()
    result = invoke(["launch", "cursor", "off"])
    assert result.exit_code == 0, result.output
    assert read_db(cursor_home)[cc.OPENAI_KEY_ROW] == "sk-user-new-choice"
    assert "leaving it untouched" in result.output


def test_off_without_backup_errors(cursor_home: Path):
    result = invoke(["launch", "cursor", "off"])
    assert result.exit_code == 1
    assert "never turned on" in (result.stderr or result.output)


def test_off_revoke_key_deletes_server_key(cursor_home: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SFERENCE_API_KEY", raising=False)
    monkeypatch.setattr(cli_main, "_read_token", lambda: FAKE_JWT)
    calls: list[dict] = []

    def fake_control(base, method, path, token, payload=None):
        calls.append({"method": method, "path": path})
        if method == "POST":
            return 201, {
                "id": "11111111-1111-1111-1111-111111111111",
                "name": payload["name"],
                "plaintext": "sk_provisioned",
                "prefix": "sk_prov",
                "created_at": "2026-10-20T00:00:00Z",
            }
        return 204, {}

    monkeypatch.setattr(launch_mod, "control_request", fake_control)
    assert invoke(["launch", "cursor"]).exit_code == 0
    result = invoke(["launch", "cursor", "off", "--revoke-key"])
    assert result.exit_code == 0, result.output
    assert {"method": "DELETE", "path": "/control/v1/auth/api-keys/11111111-1111-1111-1111-111111111111"} in calls
    assert "Revoked API key" in result.output


def test_off_revoke_key_with_api_key_credential_notes_manual(cursor_home: Path, monkeypatch: pytest.MonkeyPatch):
    # Provisioned key in the backup, but the current credential is an sk_ key,
    # which cannot call the JWT-only revoke endpoint.
    assert invoke(["launch", "cursor"]).exit_code == 0
    backup = cc.read_backup()
    backup["provisioned_key"] = {"id": "11111111-1111-1111-1111-111111111111", "tag": "t", "name": "k"}
    cc.write_backup(backup)
    monkeypatch.setattr(
        launch_mod, "control_request", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no HTTP expected"))
    )
    result = invoke(["launch", "cursor", "off", "--revoke-key"])
    assert result.exit_code == 0, result.output
    assert "Console → API Keys" in result.output


def test_off_dry_run(cursor_home: Path):
    assert invoke(["launch", "cursor"]).exit_code == 0
    snapshot = read_db(cursor_home)
    result = invoke(["launch", "cursor", "off", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "dry run" in result.output
    assert read_db(cursor_home) == snapshot
    assert cc.read_backup() is not None


# ----------------------------------------------------------------------- status


def test_status_reports_routing_on(cursor_home: Path):
    assert invoke(["launch", "cursor"]).exit_code == 0
    result = invoke(["launch", "cursor", "status"])
    assert result.exit_code == 0, result.output
    assert "plan: pro" in result.output
    assert "Sference routing: ON" in result.output
    assert "https://api.sference.com/v1" in result.output
    assert f"default model: {DEFAULT_MODEL}" in result.output


def test_status_reports_routing_off(cursor_home: Path):
    result = invoke(["launch", "cursor", "status"])
    assert result.exit_code == 0, result.output
    assert "Sference routing: off" in result.output
    assert "https://api.openai.com/v1" in result.output


def test_on_rerun_after_cursor_migrated_the_key(cursor_home: Path, monkeypatch: pytest.MonkeyPatch):
    """Cursor migrates our plaintext row into its secret store on first start.
    A second `launch cursor` must not refuse over 'its own' stored key, must not
    provision a second key, and must keep the ORIGINAL backup for off."""
    monkeypatch.delenv("SFERENCE_API_KEY", raising=False)
    monkeypatch.setattr(cli_main, "_read_token", lambda: FAKE_JWT)
    created: list[dict] = []

    def fake_control(base, method, path, token, payload=None):
        if method == "POST":
            created.append(payload)
            return 201, {
                "id": "11111111-1111-1111-1111-111111111111",
                "name": payload["name"],
                "plaintext": "sk_provisioned",
                "prefix": "sk_prov",
                "created_at": "2026-10-20T00:00:00Z",
            }
        raise AssertionError("unexpected HTTP")

    monkeypatch.setattr(launch_mod, "control_request", fake_control)
    assert invoke(["launch", "cursor"]).exit_code == 0
    assert len(created) == 1
    original_backup = cc.read_backup()

    # Simulate Cursor's first start: plaintext row migrated into the secret store.
    conn = cc.connect(cursor_home)
    conn.execute("DELETE FROM ItemTable WHERE key = ?", (cc.OPENAI_KEY_ROW,))
    conn.execute(
        "INSERT OR REPLACE INTO ItemTable (key, value) VALUES (?, ?)", (cc.SECRET_KEY_ROW, "v10-migrated")
    )
    conn.commit()
    conn.close()

    result = invoke(["launch", "cursor"])
    assert result.exit_code == 0, result.output
    assert len(created) == 1  # no second key provisioned
    assert "already configured in Cursor" in result.output
    rows = read_db(cursor_home)
    assert cc.OPENAI_KEY_ROW not in rows  # not rewritten behind the secret store's back
    assert rows[cc.SECRET_KEY_ROW] == "v10-migrated"
    backup = cc.read_backup()
    assert backup["reactive"] == original_backup["reactive"]
    assert backup["openai_key_row"] == original_backup["openai_key_row"]
    assert backup["provisioned_key"]["id"] == "11111111-1111-1111-1111-111111111111"


def test_off_after_rerun_restores_the_original_state(cursor_home: Path):
    before = read_db(cursor_home)
    assert invoke(["launch", "cursor"]).exit_code == 0
    # Cursor/user activity between the two runs.
    conn = cc.connect(cursor_home)
    conn.execute("DELETE FROM ItemTable WHERE key = ?", (cc.OPENAI_KEY_ROW,))
    conn.execute(
        "INSERT OR REPLACE INTO ItemTable (key, value) VALUES (?, ?)", (cc.SECRET_KEY_ROW, "v10-migrated")
    )
    conn.commit()
    conn.close()
    assert invoke(["launch", "cursor", "--model", "zai-org/GLM-5.3"]).exit_code == 0
    assert invoke(["launch", "cursor", "off"]).exit_code == 0
    after = read_db(cursor_home)
    assert after[REACTIVE] == before[REACTIVE]
    assert cc.OPENAI_KEY_ROW not in after


def test_status_reports_on_when_key_lives_in_keychain(cursor_home: Path, monkeypatch: pytest.MonkeyPatch):
    assert invoke(["launch", "cursor"]).exit_code == 0
    # Cursor migrated the row into the macOS Keychain (no row, no secret:// row).
    conn = cc.connect(cursor_home)
    conn.execute("DELETE FROM ItemTable WHERE key = ?", (cc.OPENAI_KEY_ROW,))
    conn.commit()
    conn.close()
    monkeypatch.setattr(cc, "_keychain_item_present", lambda: True)
    result = invoke(["launch", "cursor", "status"])
    assert result.exit_code == 0, result.output
    assert "Sference routing: ON" in result.output
    assert "macOS Keychain" in result.output
