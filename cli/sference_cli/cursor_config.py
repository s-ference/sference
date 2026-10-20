"""Cursor settings management for ``sference launch cursor``.

Cursor keeps its settings in a SQLite database (``state.vscdb``). The
BYOK-relevant rows in ``ItemTable`` are:

- ``...persistentStorage.applicationUser``: one JSON blob holding
  ``openAIBaseUrl``, ``useOpenAIKey`` and ``aiSettings``
  (``userAddedModels``, ``modelOverrideEnabled``/``modelOverrideDisabled``,
  ``modelConfig``). Cursor rewrites the whole blob from memory on exit, so
  edits must happen while Cursor is closed (``code.lock`` holds the running
  main-process pid) or they are silently lost.
- ``cursorAuth/openAIKey``: the legacy plaintext OpenAI key. Cursor reads it
  at startup, migrates it into its secret store (macOS Keychain, or the
  safe-storage ``secret://cursorAuth/openAIKey`` row), then removes the
  plaintext row. A key already in the secret store SHADOWS a plaintext row we
  write, so a pre-existing stored key has to be removed first — which we only
  do with explicit consent (``--replace-existing-key``), because it cannot be
  restored afterwards.

Writes happen in a single IMMEDIATE transaction: either the whole
configuration lands or none of it does. Stdlib-only (sqlite3, json) so this
module imports anywhere.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

REACTIVE_KEY = (
    "src.vs.platform.reactivestorage.browser.reactiveStorageServiceImpl"
    ".persistentStorage.applicationUser"
)
OPENAI_KEY_ROW = "cursorAuth/openAIKey"
SECRET_KEY_ROW = "secret://cursorAuth/openAIKey"
ACCESS_TOKEN_ROW = "cursorAuth/accessToken"
MEMBERSHIP_ROW = "cursorAuth/stripeMembershipType"

# Cursor's BYOK (own OpenAI key) requires a paid plan. Everything but "free"
# counts; a missing row means we cannot prove a paid plan either.
FREE_MEMBERSHIPS = {"free"}

# macOS Keychain location of Cursor's secret store (VS Code safe-storage
# convention). Deleting the item lets Cursor re-migrate our plaintext row into
# a fresh item it owns; we never read the stored secret.
KEYCHAIN_SERVICE = "Cursor Safe Storage"
KEYCHAIN_ACCOUNT = "cursorAuth/openAIKey"

# Picker labels are stamped with this prefix so users can tell which entries
# route through Sference; the API accepts and strips it at ingress (a Cursor
# picker's label IS the wire model id — there is no separate display field).
SFERENCE_NAMESPACE_PREFIX = "sference/"

# Version 1 backups predate the prefixed picker labels and carry no
# ``model_ids``; they restore fine (off replays the snapshot verbatim) and are
# migrated on the next ``on`` run.
BACKUP_VERSION = 2
READABLE_BACKUP_VERSIONS = (1, 2)


def picker_model_id(canonical: str) -> str:
    """The picker label for a catalog id: ``sference/<canonical>``."""
    return SFERENCE_NAMESPACE_PREFIX + canonical


def strip_picker_prefix(model_id: str) -> str:
    """The canonical id for a possibly-prefixed picker label (idempotent)."""
    if model_id.startswith(SFERENCE_NAMESPACE_PREFIX):
        return model_id[len(SFERENCE_NAMESPACE_PREFIX):]
    return model_id


class CursorConfigError(Exception):
    """A refusal that aborts ``launch cursor`` before anything was written."""


def default_config_dir() -> Path:
    """The directory holding Cursor's ``state.vscdb`` for this platform."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
        return base / "Cursor" / "User" / "globalStorage"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Cursor" / "User" / "globalStorage"
    return Path.home() / ".config" / "Cursor" / "User" / "globalStorage"


def state_db_path(config_dir: Path) -> Path:
    return config_dir / "state.vscdb"


def lock_file_path(config_dir: Path) -> Path:
    return config_dir.parent.parent / "code.lock"


def backup_path() -> Path:
    return Path.home() / ".sference" / "cursor" / "backup.json"


def cursor_running_pid(config_dir: Path) -> Optional[int]:
    """The pid of the running Cursor main process, or None.

    Cursor writes its pid to ``code.lock`` next to ``User/``; a stale lock
    (dead pid) is treated as not running.
    """
    try:
        pid = int(lock_file_path(config_dir).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    if pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        return pid
    return pid


def connect(config_dir: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(state_db_path(config_dir)))
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def _read_item(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM ItemTable WHERE key = ?", (key,)).fetchone()
    if row is None or row[0] is None:
        return None
    value = row[0]
    return value if isinstance(value, str) else str(value)


def read_reactive(conn: sqlite3.Connection) -> dict[str, Any]:
    raw = _read_item(conn, REACTIVE_KEY)
    if raw is None:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CursorConfigError(
            f"Could not parse Cursor's settings blob in {REACTIVE_KEY}: {exc}. "
            "Refusing to modify it."
        ) from exc
    if not isinstance(data, dict):
        raise CursorConfigError("Cursor's settings blob is not a JSON object; refusing to modify it.")
    return data


def read_plan_state(conn: sqlite3.Connection) -> dict[str, Any]:
    """The facts the plan gate and status command report on."""
    return {
        "signed_in": _read_item(conn, ACCESS_TOKEN_ROW) is not None,
        "membership": _read_item(conn, MEMBERSHIP_ROW),
        "openai_key_row": _read_item(conn, OPENAI_KEY_ROW),
        "secret_row_present": _read_item(conn, SECRET_KEY_ROW) is not None,
    }


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _keychain_item_present() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        proc = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-a", KEYCHAIN_ACCOUNT],
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _keychain_delete() -> bool:
    """Delete Cursor's saved OpenAI key from the macOS Keychain.

    macOS may ask the user to confirm (the item is owned by Cursor). We never
    read the secret. Returns False when the item is absent or deletion failed.
    """
    if sys.platform != "darwin":
        return False
    try:
        proc = subprocess.run(
            ["security", "delete-generic-password", "-s", KEYCHAIN_SERVICE, "-a", KEYCHAIN_ACCOUNT],
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def stored_key_situation(plan_state: dict[str, Any]) -> str:
    """Which stores currently hold an OpenAI key (for status/refusal messages)."""
    places = []
    if plan_state.get("openai_key_row") is not None:
        places.append("settings database")
    if plan_state.get("secret_row_present"):
        places.append("encrypted settings store")
    if _keychain_item_present():
        places.append("macOS Keychain")
    return " + ".join(places) if places else "none"


def _union_ordered(existing: list[Any], additions: list[str]) -> list[str]:
    out = [str(x) for x in existing if isinstance(x, str)]
    for model in additions:
        if model not in out:
            out.append(model)
    return out


def build_on_changes(
    reactive: dict[str, Any],
    *,
    openai_base_url: str,
    model_ids: list[str],
    default_model: str,
    sference_only: bool,
    previously_written: list[str] | None = None,
) -> dict[str, Any]:
    """The new reactive-blob fields for turning Sference routing on.

    Pure function of the current blob: ``aiSettings`` arrays are unioned so the
    user's other custom models survive, and the composer default is pointed at
    ``default_model`` so a fresh chat uses Sference without a picker visit.
    ``previously_written`` carries the ids an earlier ``launch cursor`` wrote —
    they are swapped out, not unioned, so a re-run migrates labels (e.g. to the
    prefixed form) instead of accumulating duplicates.
    """
    removed = set(previously_written or [])
    ai = reactive.get("aiSettings") if isinstance(reactive.get("aiSettings"), dict) else {}
    user_added = _union_ordered(
        [m for m in (ai.get("userAddedModels") or []) if m not in removed], model_ids
    )
    enabled = _union_ordered(
        [m for m in (ai.get("modelOverrideEnabled") or []) if m not in removed], model_ids
    )
    disabled = [m for m in (ai.get("modelOverrideDisabled") or []) if m not in model_ids]
    if sference_only:
        catalog_names = [
            m.get("name")
            for m in (reactive.get("availableDefaultModels2") or [])
            if isinstance(m, dict) and m.get("name") and not m.get("isUserAdded")
        ]
        if not catalog_names:
            raise CursorConfigError(
                "--sference-only needs Cursor's cached model catalog to know what to disable, "
                "but it is empty. Open Cursor once, let it load the model list, and re-run."
            )
        disabled = sorted({*disabled, *(n for n in catalog_names if n not in model_ids)})
    composer = ai.get("modelConfig", {}).get("composer", {}) if isinstance(ai.get("modelConfig"), dict) else {}
    model_config = dict(ai.get("modelConfig") or {})
    model_config["composer"] = {
        "modelName": default_model,
        "maxMode": bool(composer.get("maxMode", False)),
        "selectedModels": [{"modelId": default_model, "parameters": []}],
    }
    return {
        "openAIBaseUrl": openai_base_url,
        "useOpenAIKey": True,
        "aiSettings": {
            **ai,
            "userAddedModels": user_added,
            "modelOverrideEnabled": enabled,
            "modelOverrideDisabled": disabled,
            "modelConfig": model_config,
        },
    }


def snapshot_for_backup(reactive: dict[str, Any]) -> dict[str, Any]:
    """The prior values ``off`` restores verbatim (absence recorded explicitly)."""
    ai = reactive.get("aiSettings") if isinstance(reactive.get("aiSettings"), dict) else {}
    return {
        "openAIBaseUrl": {"present": "openAIBaseUrl" in reactive, "value": reactive.get("openAIBaseUrl")},
        "useOpenAIKey": {"present": "useOpenAIKey" in reactive, "value": reactive.get("useOpenAIKey")},
        "aiSettings": {
            "userAddedModels": list(ai.get("userAddedModels") or []),
            "modelOverrideEnabled": list(ai.get("modelOverrideEnabled") or []),
            "modelOverrideDisabled": list(ai.get("modelOverrideDisabled") or []),
            "modelConfig": ai.get("modelConfig") or {},
        },
    }


def preflight_on(
    config_dir: Path,
    plan_state: dict[str, Any],
    *,
    api_key: str,
    replace_existing_key: bool,
    force: bool,
    backup: Optional[dict[str, Any]] = None,
    already_on: bool = False,
) -> None:
    """Raise CursorConfigError when turning on is not safe right now.

    ``already_on`` (a previous ``launch cursor`` left Sference routing active)
    relaxes the stored-key refusals: the key in Cursor's secret store is the
    one WE put there (Cursor migrated it), not a foreign key.
    """
    pid = cursor_running_pid(config_dir)
    if pid is not None and not force:
        raise CursorConfigError(
            f"Cursor is running (pid {pid}). Cursor rewrites its settings from memory on exit, "
            "which would discard this change. Quit Cursor and re-run, or pass --force to write "
            "anyway (you must then fully restart Cursor for it to take effect)."
        )
    if not plan_state["signed_in"]:
        raise CursorConfigError(
            "Cursor is not signed in. Sference routing uses Cursor's own-API-key (BYOK) mode, "
            "which requires a signed-in paid Cursor plan (Pro/Business/Enterprise)."
        )
    membership = plan_state.get("membership")
    if membership in FREE_MEMBERSHIPS or membership is None:
        raise CursorConfigError(
            "Sference routing uses Cursor's own-API-key (BYOK) mode, which needs a paid Cursor "
            f"plan (Pro/Business/Enterprise); this account reports '{membership or 'unknown'}'."
        )
    if replace_existing_key or already_on:
        return
    stored = plan_state.get("openai_key_row")
    written_hash = (backup or {}).get("openai_key_row", {}).get("written_sha256")
    stored_is_ours = stored is not None and (
        stored == api_key or (written_hash is not None and sha256_text(stored) == written_hash)
    )
    if stored is not None and not stored_is_ours:
        raise CursorConfigError(
            "Cursor already has an OpenAI API key saved in its settings database, and it is not "
            "the Sference key. Re-run with --replace-existing-key to overwrite it (the old key "
            "cannot be restored), or remove it in Cursor Settings → Models first."
        )
    foreign = []
    if plan_state.get("secret_row_present"):
        foreign.append("Cursor's encrypted settings store")
    if _keychain_item_present():
        foreign.append("your macOS Keychain")
    if foreign:
        raise CursorConfigError(
            f"Cursor already has an OpenAI API key saved in {' and '.join(foreign)}. "
            "Re-run with --replace-existing-key to replace it (the old key cannot be "
            "restored; on macOS you will be asked to confirm the Keychain change), or "
            "remove it in Cursor Settings → Models first."
        )


def apply_on(
    conn: sqlite3.Connection,
    *,
    openai_base_url: str,
    api_key: Optional[str],
    model_ids: list[str],
    default_model: str,
    sference_only: bool,
    replace_existing_key: bool,
    previously_written: list[str] | None = None,
) -> dict[str, Any]:
    """Write the Sference configuration in one transaction; return the backup dict.

    ``api_key`` is None when Sference routing is already on and the key lives
    in Cursor's secret store (a re-run refreshing models/defaults): the key row
    is then left alone and the caller carries the prior backup's key metadata
    forward. ``model_ids`` are recorded in the backup so the next run can swap
    exactly the set it wrote (see ``build_on_changes``).
    """
    reactive = read_reactive(conn)
    backup = {
        "version": BACKUP_VERSION,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "openai_base_url": openai_base_url,
        "default_model": default_model,
        "sference_only": sference_only,
        "model_ids": list(model_ids),
        "reactive": snapshot_for_backup(reactive),
        "openai_key_row": {
            "existed": _read_item(conn, OPENAI_KEY_ROW) is not None,
            "written_sha256": sha256_text(api_key) if api_key is not None else None,
        },
        "secret_row_deleted": _read_item(conn, SECRET_KEY_ROW) is not None and replace_existing_key,
        "keychain_item_deleted": False,
        "provisioned_key": None,
    }
    changes = build_on_changes(
        reactive,
        openai_base_url=openai_base_url,
        model_ids=model_ids,
        default_model=default_model,
        sference_only=sference_only,
        previously_written=previously_written,
    )
    new_reactive = {**reactive, **changes}
    conn.isolation_level = None
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT OR REPLACE INTO ItemTable (key, value) VALUES (?, ?)",
            (REACTIVE_KEY, json.dumps(new_reactive, separators=(",", ":"))),
        )
        if api_key is not None:
            conn.execute(
                "INSERT OR REPLACE INTO ItemTable (key, value) VALUES (?, ?)",
                (OPENAI_KEY_ROW, api_key),
            )
        if replace_existing_key:
            conn.execute("DELETE FROM ItemTable WHERE key = ?", (SECRET_KEY_ROW,))
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return backup


def write_backup(backup: dict[str, Any]) -> Path:
    path = backup_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    path.write_text(json.dumps(backup, indent=2) + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def read_backup() -> Optional[dict[str, Any]]:
    path = backup_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) and data.get("version") in READABLE_BACKUP_VERSIONS else None


def apply_off(conn: sqlite3.Connection, backup: dict[str, Any]) -> list[str]:
    """Restore the state recorded in ``backup``; return human-readable notes."""
    notes: list[str] = []
    reactive = read_reactive(conn)
    prior = backup["reactive"]
    for field in ("openAIBaseUrl", "useOpenAIKey"):
        saved = prior[field]
        if saved["present"]:
            reactive[field] = saved["value"]
        else:
            reactive.pop(field, None)
    ai = reactive.get("aiSettings") if isinstance(reactive.get("aiSettings"), dict) else {}
    saved_ai = prior["aiSettings"]
    reactive["aiSettings"] = {
        **ai,
        "userAddedModels": saved_ai["userAddedModels"],
        "modelOverrideEnabled": saved_ai["modelOverrideEnabled"],
        "modelOverrideDisabled": saved_ai["modelOverrideDisabled"],
        "modelConfig": saved_ai["modelConfig"],
    }
    key_info = backup["openai_key_row"]
    current_key = _read_item(conn, OPENAI_KEY_ROW)
    delete_key_row = False
    if key_info["existed"]:
        notes.append(
            "The OpenAI API key that Cursor had before `launch cursor` was overwritten and cannot "
            "be restored; the Sference key stays in the settings database. Remove it in Cursor "
            "Settings → Models if you want it gone."
        )
    elif current_key is not None and sha256_text(current_key) == key_info["written_sha256"]:
        delete_key_row = True
    elif current_key is not None:
        notes.append(
            "The OpenAI API key in Cursor's settings database was changed after `launch cursor`; "
            "leaving it untouched."
        )
    conn.isolation_level = None
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT OR REPLACE INTO ItemTable (key, value) VALUES (?, ?)",
            (REACTIVE_KEY, json.dumps(reactive, separators=(",", ":"))),
        )
        if delete_key_row:
            conn.execute("DELETE FROM ItemTable WHERE key = ?", (OPENAI_KEY_ROW,))
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    if backup.get("secret_row_deleted") or backup.get("keychain_item_deleted"):
        notes.append(
            "If Cursor already moved the Sference key into its secret store, the key remains "
            "there but is unused (own-key mode is off). Remove it in Cursor Settings → Models "
            "if you want it gone."
        )
    return notes
