"""Launch external agent tools against the Sference API."""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
from pathlib import Path
from typing import Optional

import typer

from . import cursor_config
from ._discovery import (
    AUTO_COMPACT_WINDOW_ENV_KEY,
    DEFAULT_AUTO_COMPACT_WINDOW,
    DISCOVERY_ENV_KEY,
    WINDOW_ENFORCEMENT_ENV_KEY,
    fetch_sference_model_entries,
    prime_picker_cache,
)
from .device_auth import DeviceAuthError, control_request

DEFAULT_LAUNCH_MODEL = "zai-org/GLM-5.2"
# Claude Code's default request shape is thinking-disabled, which GLM-5.3-served
# models 400 on (verified vs the API). Kimi-K3 tolerates it and is the stronger
# coding model — claude-only default; the other launchers keep GLM-5.2.
CLAUDE_DEFAULT_LAUNCH_MODEL = "moonshotai/Kimi-K3"
# Cursor chats on Kimi-K3 out of the box for the same reason (thinking-capable,
# tool-calling, image input).
CURSOR_DEFAULT_LAUNCH_MODEL = "moonshotai/Kimi-K3"
DEFAULT_API_BASE_URL = "https://api.sference.com"

# Tag on API keys the cursor launcher provisions, so users can find them in
# Console → API Keys (and we can reconcile them later).
CURSOR_API_KEY_TAG = "sference-cli-cursor"

# Thinking-effort variants stamped on Sference model entries in the opencode config.
# The levels are a subset of OpenAI's ReasoningEffort enum that the platform accepts
# (``reasoning_effort`` on chat completions); ``none`` disables thinking, the rest
# enable it and each model's chat template decides what the level means.
#
# opencode merges a variant's keys straight into the model options, and the AI SDK
# openai-compatible provider forwards a top-level ``reasoningEffort`` option as
# ``reasoning_effort`` in the request body. So a variant is the flat
# ``{"reasoningEffort": level}`` — nesting it under ``"options"`` (as 0.3.1 did) leaves
# ``reasoningEffort`` unread and every variant sends the catalog default.
OPENCODE_THINKING_VARIANTS = ("none", "medium", "high", "xhigh")


def resolve_api_base_url(explicit: Optional[str]) -> str:
    if explicit:
        return explicit.rstrip("/")
    env = os.environ.get("SFERENCE_BASE_URL")
    if env:
        return env.strip().rstrip("/")
    return DEFAULT_API_BASE_URL


def resolve_launch_model(explicit: Optional[str], *, default: str = DEFAULT_LAUNCH_MODEL) -> str:
    """Default catalog model for ``launch`` subcommands (--model overrides)."""
    if explicit:
        return explicit
    env = os.environ.get("SFERENCE_MODEL")
    if env and env.strip():
        return env.strip()
    return default


def resolve_openai_base_url(explicit: Optional[str]) -> str:
    """OpenAI-compatible base URL with ``/v1`` for tool config files (Pi, opencode)."""
    base = resolve_api_base_url(explicit)
    base = base.rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def find_claude_executable() -> str | None:
    return shutil.which("claude")


def find_pi_executable() -> str | None:
    return shutil.which("pi")


def build_claude_env(
    *,
    api_key: str,
    base_url: str,
    model: str,
    enable_tool_search: bool,
) -> dict[str, str]:
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = base_url
    env["ANTHROPIC_AUTH_TOKEN"] = api_key
    env["ANTHROPIC_MODEL"] = model
    # Subagents and background calls resolve model aliases (opus/sonnet/haiku/fable)
    # independently of ANTHROPIC_MODEL; pin every alias to the catalog model so no
    # request goes out with a claude-* model id the API can't serve.
    env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = model
    env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = model
    env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = model
    env["ANTHROPIC_DEFAULT_FABLE_MODEL"] = model
    env["CLAUDE_CODE_SUBAGENT_MODEL"] = model
    # Deprecated alias for ANTHROPIC_DEFAULT_HAIKU_MODEL; still read by older
    # Claude Code versions, and a user-set value would leak a claude-* id.
    env["ANTHROPIC_SMALL_FAST_MODEL"] = model
    # Claude Code prefers AUTH_TOKEN; drop API_KEY to avoid precedence confusion.
    env.pop("ANTHROPIC_API_KEY", None)
    # Gateway model discovery: the picker reads the primed cache (written
    # before exec) without any further credential.
    env[DISCOVERY_ENV_KEY] = "1"
    # Sference catalog ids are "unknown" to Claude Code, which would clamp
    # them to a 200k window; let the auto-compact window below apply instead
    # (catalog models serve 1M).
    env[WINDOW_ENFORCEMENT_ENV_KEY] = "1"
    env.setdefault(AUTO_COMPACT_WINDOW_ENV_KEY, DEFAULT_AUTO_COMPACT_WINDOW)
    if enable_tool_search:
        env["ENABLE_TOOL_SEARCH"] = "true"
    else:
        env.pop("ENABLE_TOOL_SEARCH", None)
    return env


def build_claude_settings_json(env: dict[str, str]) -> str:
    """The launcher's env block as a ``--settings`` JSON string.

    Why not just the process env: a settings.json ``env`` entry (e.g. written
    by sference-switch's ``claude on``) beats the process environment per key,
    so a switch-managed machine would silently re-route the session through
    the switch. ``--settings`` loads AFTER user settings and wins per key.

    ANTHROPIC_AUTH_TOKEN is deliberately EXCLUDED: argv is visible to every
    process on the machine (``ps``), and nothing in a settings.json env block
    competes for that key — the process env carries it safely alone.
    """
    keys = (
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_DEFAULT_FABLE_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL",
        "ANTHROPIC_SMALL_FAST_MODEL",
        DISCOVERY_ENV_KEY,
        WINDOW_ENFORCEMENT_ENV_KEY,
        AUTO_COMPACT_WINDOW_ENV_KEY,
        "ENABLE_TOOL_SEARCH",
    )
    return json.dumps({"env": {k: env[k] for k in keys if k in env}})


def launch_claude_code(
    *,
    api_key: str,
    base_url: str,
    model: str,
    model_entries: list[dict],
    enable_tool_search: bool,
    claude_args: list[str],
    dry_run: bool,
) -> None:
    claude_bin = find_claude_executable()
    if claude_bin is None:
        typer.echo(
            "Claude Code CLI not found on PATH.\n"
            "Install from https://docs.anthropic.com/en/docs/claude-code "
            "and ensure the `claude` command is available.",
            err=True,
        )
        raise typer.Exit(code=1)

    env = build_claude_env(
        api_key=api_key,
        base_url=base_url,
        model=model,
        enable_tool_search=enable_tool_search,
    )
    settings_json = build_claude_settings_json(env)
    cmd = [claude_bin, "--settings", settings_json, *claude_args]

    if dry_run:
        typer.echo(f"ANTHROPIC_BASE_URL={env['ANTHROPIC_BASE_URL']}")
        typer.echo(f"ANTHROPIC_MODEL={env['ANTHROPIC_MODEL']}")
        for alias_var in (
            "ANTHROPIC_DEFAULT_OPUS_MODEL",
            "ANTHROPIC_DEFAULT_SONNET_MODEL",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL",
            "ANTHROPIC_DEFAULT_FABLE_MODEL",
            "CLAUDE_CODE_SUBAGENT_MODEL",
        ):
            typer.echo(f"{alias_var}={env[alias_var]}")
        typer.echo("ANTHROPIC_AUTH_TOKEN=<redacted>")
        typer.echo(f"{DISCOVERY_ENV_KEY}=1")
        typer.echo(f"{WINDOW_ENFORCEMENT_ENV_KEY}=1")
        typer.echo(f"{AUTO_COMPACT_WINDOW_ENV_KEY}={env[AUTO_COMPACT_WINDOW_ENV_KEY]}")
        typer.echo(f"picker cache: {len(model_entries)} model(s) to prime")
        if enable_tool_search:
            typer.echo("ENABLE_TOOL_SEARCH=true")
        typer.echo(f"--settings {settings_json}")
        typer.echo(f"command: {claude_bin} --settings <json> {' '.join(claude_args)}")
        return

    # Prime the picker's discovery cache before exec. Warn-only: a failed
    # prime costs the /model entries, not the session.
    try:
        cache_path = prime_picker_cache(model_entries, base_url)
        typer.echo(f"sference: /model picker cache primed ({len(model_entries)} model(s) -> {cache_path})", err=True)
    except (OSError, ValueError, TypeError) as e:
        typer.echo(f"sference: warning: could not prime the /model picker cache: {e}", err=True)

    if sys.platform == "win32":
        raise SystemExit(subprocess.call(cmd, env=env))

    os.execvpe(claude_bin, cmd, env)


def _pi_models_json_path() -> Path:
    return Path.home() / ".pi" / "agent" / "models.json"


def _pi_model_config(entry: dict) -> dict:
    """Map a ``GET /v1/models`` entry to a pi ProviderModelConfig.

    The Sference provider uses pi's ``openai-responses`` API, so reasoning is
    driven by pi's native ``reasoning.effort`` field — the Sference
    ``/v1/responses`` endpoint accepts it directly and translates it to the
    worker's ``enable_thinking`` / ``reasoning_effort``. No per-model
    chat-completions ``thinkingFormat`` compat is needed (that was an
    ``openai-completions`` concern; the Responses API has one native reasoning
    knob). Reasoning/input come from ``capabilities``, cost from ``pricing``;
    ``maxTokens`` is clamped to 16384 (pi's default cap).

    Reasoning-capable models get a ``thinkingLevelMap`` exposing ``xhigh`` only
    (Sference's ``/v1/responses`` accepts effort in ``(minimal,low,medium,high,
    xhigh)`` and 400s on ``max``). ``max`` is deliberately omitted from the map
    so a ``max`` selection clamps down to ``xhigh`` (pi's
    ``clampThinkingLevel`` walks down to the nearest supported level) instead of
    sending an effort the API rejects. ``high`` and below need no map entry — pi
    passes the requested level through verbatim for unmapped levels.
    """
    caps = entry.get("capabilities") or {}
    thinking = bool((caps.get("thinking") or {}).get("supported", False))
    image_in = bool((caps.get("image_input") or {}).get("supported", False))
    pricing = entry.get("pricing") or {}
    context_tokens = entry.get("context_tokens")
    context_window = context_tokens if context_tokens is not None else 128000
    max_tokens = min(context_tokens if context_tokens is not None else 8192, 16384)
    config = {
        "id": entry["id"],
        "name": entry.get("display_name") or entry["id"],
        "reasoning": thinking,
        "input": ["text", "image"] if image_in else ["text"],
        "cost": {
            "input": pricing.get("input_per_million_usd") or 0,
            "output": pricing.get("output_per_million_usd") or 0,
            "cacheRead": pricing.get("cached_input_per_million_usd") or 0,
            "cacheWrite": 0,
        },
        "contextWindow": context_window,
        "maxTokens": max_tokens,
    }
    if thinking:
        # Expose only xhigh (the top Sference-supported effort). max is
        # unsupported on the API (400), so omitting it lets pi clamp a ``max``
        # selection down to xhigh instead of sending an effort that errors.
        config["thinkingLevelMap"] = {"xhigh": "xhigh"}
    return config


def _build_pi_models(*, base_url: str, api_key: str, model: str) -> list[dict]:
    """Build the pi model config list for the sference provider.

    Fetches the live catalog (GET {base_url}/models) and maps every
    text-generation entry to a pi ProviderModelConfig with reasoning/input/
    cost/context metadata and the per-model thinking wire format, so all
    Sference-provided models appear in pi's ``/model`` picker — not just the one
    passed to ``--model``.

    The picker always mirrors the live catalog exactly: the list is rebuilt on
    every run, so a model that was removed/deprecated from the catalog is
    dropped rather than accumulated forever, and no model is injected that
    isn't currently available from Sference. ``--model`` only selects the
    active model for the launch; it is never added to the picker when absent
    from a reachable catalog. The only exception is an unreachable catalog
    (offline): we can't show what we can't fetch, so we register just ``model``
    as a bare fallback so ``launch pi`` still works.
    """
    try:
        entries = fetch_sference_model_entries(base_url, api_key)
    except Exception:
        typer.echo(
            f"note: could not fetch model catalog from {base_url}/models; "
            f"registered only --model {model}. The /model picker will show just "
            "this model.",
            err=True,
        )
        entries = []
        fetched = False
    else:
        fetched = True
    configs = [_pi_model_config(entry) for entry in entries]
    if not fetched and model not in {c["id"] for c in configs}:
        configs.append({"id": model, "name": model})
    return configs


def _write_pi_models_json(*, base_url: str, api_key: str, models: list[dict]) -> Path:
    """Write or update ``~/.pi/agent/models.json`` with a Sference provider.

    ``models`` is the full list of pi ProviderModelConfig dicts to register —
    the Sference-provided text-generation models from the live catalog (or just
    the chosen model as an offline fallback). Existing providers other than
    ``sference`` are preserved; only the ``sference`` block is overwritten.
    """
    path = _pi_models_json_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    provider = {
        "api": "openai-responses",
        "baseUrl": base_url,
        "apiKey": api_key,
        "models": models,
    }

    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    data.setdefault("providers", {})
    data["providers"]["sference"] = provider

    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")

    return path


def launch_pi(
    *,
    api_key: str,
    base_url: str,
    model: str,
    pi_args: list[str],
    dry_run: bool,
) -> None:
    pi_bin = find_pi_executable()
    if pi_bin is None:
        typer.echo(
            "Pi CLI not found on PATH.\n"
            "Install from https://pi.dev/docs and ensure the `pi` command is available.",
            err=True,
        )
        raise typer.Exit(code=1)

    models = _build_pi_models(base_url=base_url, api_key=api_key, model=model)
    models_json = _write_pi_models_json(base_url=base_url, api_key=api_key, models=models)
    env = os.environ.copy()
    # Pi reads providers from ~/.pi/agent/models.json; no env vars needed.
    # `--models sference/**` scopes the /model picker (and Ctrl+P cycling) to the
    # Sference provider only, so built-in / other configured providers are hidden
    # by default instead of cluttering the picker. Using `**` (globstar) is
    # required: pi matches `--models` patterns against `provider/modelId`, and
    # Sference model ids contain a slash (e.g. `zai-org/GLM-5.2`), so `sference/*`
    # would match nothing (`*` does not cross `/`) and the picker would fall back
    # to showing every configured provider.
    cmd = [pi_bin, "--provider", "sference", "--model", model, "--models", "sference/**", *pi_args]

    if dry_run:
        typer.echo(f"Wrote provider 'sference' to {models_json}")
        typer.echo(f"baseUrl: {base_url}")
        typer.echo("api: openai-responses (/v1/responses)")
        typer.echo(f"model: {model}")
        typer.echo(f"models registered: {len(models)}")
        typer.echo("model scope: sference/** (other providers hidden by default)")
        typer.echo("apiKey: <redacted>")
        typer.echo(f"command: {' '.join(cmd)}")
        return

    if sys.platform == "win32":
        raise SystemExit(subprocess.call(cmd, env=env))

    os.execvpe(pi_bin, cmd, env)


def find_opencode_executable() -> str | None:
    return shutil.which("opencode")


def _opencode_config_path() -> Path:
    return Path.home() / ".config" / "opencode" / "opencode.json"


def _thinks_always(entry: dict) -> bool:
    """Whether *entry* is served by a checkpoint that cannot turn thinking off.

    The GLM-5.3 family's chat template always opens a reasoning channel, so the API
    rejects ``reasoning_effort: "none"`` for it with a 400. ``/v1/models`` carries no
    field for that today, so mirror the API's own rule: the public id, then the id an
    alias dispatches to (``zai-org/GLM-5.2`` is an alias served by GLM-5.3 weights).
    """
    ids = (entry.get("id") or "", entry.get("alias_of") or "")
    return any("glm-5.3" in i.lower() for i in ids)


def _opencode_variants(entry: Optional[dict]) -> dict:
    """The thinking-effort variants for one model entry (empty: no variants).

    No entry (the model list could not be fetched, or the model was typed but not
    returned) → every level but ``none``: an effort level is accepted by any thinking
    model, while ``none`` 400s on one that cannot turn thinking off.
    """
    if entry is not None:
        thinking = (entry.get("capabilities") or {}).get("thinking") or {}
        if thinking.get("supported") is False:
            return {}
    levels = [
        level
        for level in OPENCODE_THINKING_VARIANTS
        if level != "none" or (entry is not None and not _thinks_always(entry))
    ]
    return {level: {"reasoningEffort": level} for level in levels}


def _opencode_model_entry(model_id: str, entry: Optional[dict]) -> dict:
    config: dict = {"name": model_id}
    variants = _opencode_variants(entry)
    if variants:
        config["variants"] = variants
    return config


def _write_opencode_config(
    *, base_url: str, model: str, models: set[str], entries: Optional[dict[str, dict]] = None
) -> Path:
    """Merge a ``sference`` OpenAI-compatible provider into the opencode config.

    Reads ``~/.config/opencode/opencode.json`` (creating it if absent), sets
    ``provider.sference`` to a block pointing at the Sference OpenAI-compatible
    endpoint (``/v1/chat/completions``), and writes it back. The user's existing
    config — other providers, keybinds, theme — is preserved; only
    ``provider.sference`` is (over)written. If the existing file is not
    valid JSON, we refuse to clobber it rather than silently destroying the
    user's opencode config.

    All models in ``models`` are written to the provider's ``models`` block so
    they appear in opencode's model picker, each with the thinking-effort
    variants that model accepts (``_opencode_variants``, from its ``/v1/models``
    entry in ``entries``; selectable as ``sference/<model>@<level>`` or via
    opencode's variant switcher; no variant means the server-side catalog
    default decides). ``model`` is the default —
    set as the top-level ``model`` key so opencode uses Sference by default on
    startup.

    The API key is referenced as ``{env:SFERENCE_API_KEY}`` so no secret is
    stored on disk; ``launch_opencode`` injects the resolved key into the
    opencode process env at exec time. opencode splits ``provider/model`` on
    the first slash, so a model id like ``zai-org/GLM-5.2`` is passed
    as ``--model sference/zai-org/GLM-5.2`` and resolves to provider
    ``sference`` + model id ``zai-org/GLM-5.2``.
    """
    path = _opencode_config_path()
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError:
                typer.echo(
                    f"Could not parse existing opencode config at {path} as JSON.\n"
                    "Refusing to overwrite your opencode config. Fix or remove the "
                    "file and re-run.",
                    err=True,
                )
                raise typer.Exit(code=1)
    else:
        data = {}

    data.setdefault("provider", {})
    all_models = models | {model}
    data["provider"]["sference"] = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Sference",
        "options": {
            "baseURL": base_url,
            "apiKey": "{env:SFERENCE_API_KEY}",
        },
        "models": {
            m: _opencode_model_entry(m, (entries or {}).get(m))
            for m in sorted(all_models)
        },
    }
    # Set Sference as the default model so opencode uses it on startup without
    # the user having to select it in the picker.
    data["model"] = f"sference/{model}"

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass

    return path


def launch_opencode(
    *,
    api_key: str,
    base_url: str,
    model: str,
    opencode_args: list[str],
    dry_run: bool,
) -> None:
    opencode_bin = find_opencode_executable()
    if opencode_bin is None:
        typer.echo(
            "opencode not found on PATH.\n"
            "Install from https://opencode.ai/docs and ensure the `opencode` command is available.",
            err=True,
        )
        raise typer.Exit(code=1)

    # Fetch all available text-generation models from /v1/models so they all
    # appear in opencode's model picker, not just the default. Falls back to
    # just the chosen model if the fetch fails (e.g. offline) so the launch
    # still works.
    try:
        entries = {e["id"]: e for e in fetch_sference_model_entries(base_url, api_key)}
    except Exception:
        entries = {}
    models = set(entries)

    config_path = _write_opencode_config(
        base_url=base_url, model=model, models=models, entries=entries
    )
    env = os.environ.copy()
    # The config references {env:SFERENCE_API_KEY}; inject the resolved key so
    # no secret is persisted to disk.
    env["SFERENCE_API_KEY"] = api_key
    cmd = [opencode_bin, "--model", f"sference/{model}", *opencode_args]

    if dry_run:
        typer.echo(f"Wrote provider 'sference' to {config_path}")
        typer.echo(f"baseURL: {base_url}")
        typer.echo(f"model: {model}")
        typer.echo(f"models in picker: {len(models) + 1}")
        typer.echo("apiKey: {env:SFERENCE_API_KEY} (injected at launch)")
        typer.echo(f"command: {' '.join(cmd)}")
        return

    if sys.platform == "win32":
        raise SystemExit(subprocess.call(cmd, env=env))

    os.execvpe(opencode_bin, cmd, env)


def find_codex_executable() -> str | None:
    return shutil.which("codex")


# Codex's own config may pin an effort Sference rejects (e.g. a level only
# OpenAI models accept), so the launcher always sets one. Forwarded
# ``-c model_reasoning_effort=...`` args come later on the command line and win.
CODEX_DEFAULT_REASONING_EFFORT = "medium"


def _toml_value(value: object) -> str:
    """Render a ``codex -c key=value`` value; Codex parses the right-hand side as TOML.

    A JSON string literal is a valid TOML basic string (and JSON ``true``/``false``
    and integers are valid TOML), so ``json.dumps`` covers every value we pass.
    """
    return json.dumps(value)


# Fields of a Codex catalog entry that switch on modes Sference's /v1/responses
# cannot serve. Set on every generated entry, whatever the template says:
# responses-lite moves the tools into an ``additional_tools`` input item, a
# freeform ``apply_patch`` is a non-function (custom) tool, and websockets,
# hosted search and service tiers are OpenAI-only.
_CODEX_CATALOG_ENTRY_OVERRIDES: dict = {
    "use_responses_lite": False,
    "apply_patch_tool_type": None,
    "prefer_websockets": False,
    "supports_search_tool": False,
    "service_tiers": [],
    "additional_speed_tiers": [],
    "default_service_tier": None,
    "upgrade": None,
    "availability_nux": None,
    "visibility": "list",
}


def bundled_codex_catalog(codex_bin: str) -> Optional[list[dict]]:
    """The model catalog shipped with the installed Codex, or None if unavailable.

    Read from ``codex debug models --bundled`` at launch rather than vendored, so the
    generated entries always match the schema of the Codex version being launched.
    """
    try:
        proc = subprocess.run(
            [codex_bin, "debug", "models", "--bundled"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        models = json.loads(proc.stdout).get("models")
    except (OSError, subprocess.SubprocessError, ValueError, AttributeError):
        return None
    return models if isinstance(models, list) else None


def codex_catalog_template(models: list[dict]) -> Optional[dict]:
    """The first bundled entry Codex doesn't run in a GPT-only mode, or None."""
    for entry in models:
        if (
            isinstance(entry, dict)
            and not entry.get("use_responses_lite")
            and not entry.get("tool_mode")
            and not entry.get("multi_agent_version")
        ):
            return entry
    return None


def build_codex_model_catalog(template: dict, entries: list[dict]) -> dict:
    """A Codex ``model_catalog_json`` listing only the Sference models in ``entries``.

    Without it Codex offers ``spawn_agent`` its bundled OpenAI models, and a sub-agent
    started on one of them (e.g. ``gpt-6-luna``) sends Sference a model id and a
    responses-lite request shape it cannot serve. Each entry is a copy of
    ``template`` (Codex's own entry, for its base instructions and tool settings)
    with the GPT-only modes switched off and the Sference model's name, context
    window and image input filled in.
    """
    models = []
    for priority, entry in enumerate(entries, start=1):
        caps = entry.get("capabilities") or {}
        image_in = bool((caps.get("image_input") or {}).get("supported", False))
        name = entry.get("display_name") or entry["id"]
        model = {
            **copy.deepcopy(template),
            **_CODEX_CATALOG_ENTRY_OVERRIDES,
            "slug": entry["id"],
            "display_name": name,
            "description": f"{name} on Sference",
            "input_modalities": ["text", "image"] if image_in else ["text"],
            "priority": priority,
        }
        if entry.get("context_tokens"):
            model["context_window"] = int(entry["context_tokens"])
            model["max_context_window"] = int(entry["context_tokens"])
        models.append(model)
    return {"models": models}


def codex_catalog_path() -> Path:
    return Path.home() / ".sference" / "codex" / "models.json"


def write_codex_catalog(catalog: dict) -> Path:
    """Write the catalog atomically (concurrent launches share the path)."""
    path = codex_catalog_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(catalog), encoding="utf-8")
    os.replace(tmp, path)
    return path


def build_codex_overrides(
    *, base_url: str, model: str, entry: Optional[dict], model_catalog: Optional[Path] = None
) -> list[str]:
    """``-c`` overrides that point Codex at Sference for this process only.

    Passing the provider on the command line instead of editing
    ``~/.codex/config.toml`` leaves the user's Codex setup untouched for normal
    (OpenAI) sessions, and ``env_key`` keeps the API key off disk.

    Codex speaks only the Responses wire API, served by Sference's
    ``/v1/responses``, which takes function and ``namespace`` tools (Codex sends
    sub-agents and MCP servers as namespaces) but no hosted tools, so Codex's
    hosted ``web_search`` tool is switched off.

    ``model_provider`` applies to the whole session, so sub-agents spawned with
    ``spawn_agent`` also run on Sference. ``model_catalog`` (see
    :func:`build_codex_model_catalog`) replaces Codex's OpenAI model list with the
    Sference models, so a sub-agent either inherits ``model`` or picks another
    Sference model.
    """
    overrides = {
        "model_provider": "sference",
        "model_providers.sference.name": "Sference",
        "model_providers.sference.base_url": base_url,
        "model_providers.sference.env_key": "SFERENCE_API_KEY",
        "model_providers.sference.wire_api": "responses",
        "model_reasoning_effort": CODEX_DEFAULT_REASONING_EFFORT,
        "web_search": "disabled",
    }
    # Codex has no metadata for Sference models and falls back to a generic
    # context window; give it the real one so auto-compaction fires in time.
    if entry is not None and entry.get("context_tokens"):
        overrides["model_context_window"] = int(entry["context_tokens"])
    if model_catalog is not None:
        overrides["model_catalog_json"] = str(model_catalog)
    args: list[str] = []
    for key, value in overrides.items():
        args += ["-c", f"{key}={_toml_value(value)}"]
    return [*args, "--model", model]


def _codex_config_path() -> Path:
    home = os.environ.get("CODEX_HOME")
    return Path(home) / "config.toml" if home else Path.home() / ".codex" / "config.toml"


def _codex_pins_subagent_model() -> bool:
    """Whether the user's Codex config sets ``agents.default_subagent_model``.

    That model is always an OpenAI id (Codex rejects anything off its own list),
    and with the Sference provider it 400s every sub-agent. A line scan rather than
    a TOML parse: ``tomllib`` needs Python 3.11 and this only drives a warning.
    """
    try:
        text = _codex_config_path().read_text(encoding="utf-8")
    except OSError:
        return False
    return re.search(r"^\s*(agents\.)?default_subagent_model\s*=", text, re.MULTILINE) is not None


def launch_codex(
    *,
    api_key: str,
    base_url: str,
    model: str,
    codex_args: list[str],
    dry_run: bool,
) -> None:
    codex_bin = find_codex_executable()
    if codex_bin is None:
        typer.echo(
            "Codex CLI not found on PATH.\n"
            "Install from https://developers.openai.com/codex/cli "
            "(e.g. `npm i -g @openai/codex`) and ensure the `codex` command is available.",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        entry_list = fetch_sference_model_entries(base_url, api_key)
    except Exception:
        entry_list = []
    entries = {e["id"]: e for e in entry_list}

    catalog: Optional[dict] = None
    if entry_list:
        bundled = bundled_codex_catalog(codex_bin)
        template = codex_catalog_template(bundled) if bundled else None
        if template is not None:
            catalog = build_codex_model_catalog(template, entry_list)
        else:
            typer.echo(
                "warning: could not read Codex's model catalog (`codex debug models --bundled`); "
                "sub-agents may be offered OpenAI models that Sference cannot serve.",
                err=True,
            )

    if _codex_pins_subagent_model():
        typer.echo(
            f"warning: {_codex_config_path()} sets agents.default_subagent_model, an OpenAI "
            "model Sference cannot serve; sub-agents will fail. Remove it to have "
            f"sub-agents inherit {model}.",
            err=True,
        )

    env = os.environ.copy()
    # model_providers.sference.env_key names this var; inject the resolved key
    # so no secret is persisted to disk.
    env["SFERENCE_API_KEY"] = api_key
    # A dry run reports the catalog path without writing it.
    catalog_path = None
    if catalog is not None:
        catalog_path = codex_catalog_path() if dry_run else write_codex_catalog(catalog)
    overrides = build_codex_overrides(
        base_url=base_url, model=model, entry=entries.get(model), model_catalog=catalog_path
    )
    cmd = [codex_bin, *overrides, *codex_args]

    if dry_run:
        typer.echo(f"base_url: {base_url}")
        typer.echo("wire_api: responses (/v1/responses)")
        typer.echo(f"model: {model}")
        typer.echo("apiKey: $SFERENCE_API_KEY (injected at launch)")
        if catalog is not None:
            typer.echo(f"model_catalog_json: {catalog_path} ({len(catalog['models'])} Sference models)")
        typer.echo(f"command: {' '.join(cmd)}")
        return

    if sys.platform == "win32":
        raise SystemExit(subprocess.call(cmd, env=env))

    os.execvpe(codex_bin, cmd, env)


def _provision_cursor_api_key(base_url: str, jwt_token: str) -> dict:
    """Create a Sference API key for Cursor via the control plane.

    Uses the device-login JWT (``require_jwt_only`` on the server side); the
    returned plaintext is shown exactly once, so it goes straight into Cursor's
    settings database and the key id into the backup for ``off --revoke-key``.
    """
    name = f"sference-cli cursor ({socket.gethostname()})"
    try:
        status, body = control_request(
            base_url,
            "POST",
            "/control/v1/auth/api-keys",
            jwt_token,
            {"name": name, "tag": CURSOR_API_KEY_TAG},
        )
    except DeviceAuthError as exc:
        typer.echo(f"ERROR: could not create an API key for Cursor: {exc.description}", err=True)
        raise typer.Exit(code=1) from None
    if status != 201 or not isinstance(body.get("plaintext"), str):
        detail = body.get("detail") if isinstance(body, dict) else None
        typer.echo(
            f"ERROR: could not create an API key for Cursor ({status}): {detail or body}",
            err=True,
        )
        raise typer.Exit(code=1)
    return {"id": str(body["id"]), "plaintext": body["plaintext"], "name": name, "tag": CURSOR_API_KEY_TAG}


def _revoke_cursor_api_key(base_url: str, credential: str, key_id: str) -> bool:
    """Best-effort revoke of a key we provisioned. Needs a JWT (the endpoint is
    session-only); returns False (with a note) when the current credential is an
    API key, which cannot revoke."""
    if credential.startswith("sk_"):
        return False
    try:
        status, _ = control_request(base_url, "DELETE", f"/control/v1/auth/api-keys/{key_id}", credential)
    except DeviceAuthError:
        return False
    return status in (204, 404)


def _cursor_resolve_key(
    *,
    explicit: Optional[str],
    credential: Optional[str],
    backup: Optional[dict],
    plan_state: dict,
    base_url: str,
    dry_run: bool,
) -> tuple[Optional[str], Optional[dict], str]:
    """Pick the Sference key Cursor will use; never writes anything.

    Returns ``(key, provisioned, source_label)``. ``key`` is None only in a dry
    run that would provision (the placeholder is reported instead).
    """
    if explicit:
        return explicit, None, "--api-key"
    if credential is not None and credential.startswith("sk_"):
        return credential, None, "your Sference API key"
    # Reuse the key a previous `launch cursor` wrote, if Cursor hasn't moved or
    # replaced it (it migrates the row into its secret store on next start, so
    # this only fires before that first restart).
    row = plan_state.get("openai_key_row")
    if (
        row is not None
        and backup is not None
        and cursor_config.sha256_text(row) == backup.get("openai_key_row", {}).get("written_sha256")
    ):
        return row, None, "the Sference key already configured in Cursor"
    if dry_run:
        return None, None, "a new Sference API key (created at launch)"
    if credential is None:
        typer.echo(
            "No Sference credential available to create an API key with.\n"
            "Run `sference auth login`, or pass --api-key 'sk_...'.",
            err=True,
        )
        raise typer.Exit(code=1)
    provisioned = _provision_cursor_api_key(base_url, credential)
    return provisioned["plaintext"], provisioned, f"new API key '{provisioned['name']}'"


def _cursor_on(
    *,
    model: Optional[str],
    models: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    sference_only: bool,
    replace_existing_key: bool,
    force: bool,
    dry_run: bool,
) -> None:
    from sference_cli.main import _read_token

    config_dir = cursor_config.default_config_dir()
    db_path = cursor_config.state_db_path(config_dir)
    if not db_path.exists():
        typer.echo(
            f"Cursor settings not found at {db_path}.\n"
            "Install Cursor (https://cursor.com) and sign in once, then re-run.",
            err=True,
        )
        raise typer.Exit(code=1)

    resolved_base = resolve_api_base_url(base_url)
    openai_base_url = resolve_openai_base_url(base_url)
    credential = _read_token()
    backup = cursor_config.read_backup()

    conn = cursor_config.connect(config_dir)
    try:
        plan_state = cursor_config.read_plan_state(conn)
        reactive = cursor_config.read_reactive(conn)
        try:
            # A re-run while a previous `launch cursor` is still active: the key
            # already lives in Cursor's secret store (Cursor migrated the row we
            # wrote), so there is nothing to provision or write — and nothing to
            # refuse over, because the stored key is ours, not a foreign one.
            # An explicit --api-key opts out: the user is asking to CHANGE the key,
            # which is the normal path (and needs --replace-existing-key).
            already_on = (
                not replace_existing_key
                and api_key is None
                and backup is not None
                and reactive.get("useOpenAIKey") is True
                and reactive.get("openAIBaseUrl") == openai_base_url
            )
            if already_on:
                key, provisioned, key_source = None, None, "already configured in Cursor"
            else:
                key, provisioned, key_source = _cursor_resolve_key(
                    explicit=api_key,
                    credential=credential,
                    backup=backup,
                    plan_state=plan_state,
                    base_url=resolved_base,
                    dry_run=dry_run,
                )
            cursor_config.preflight_on(
                config_dir,
                plan_state,
                api_key=key if key is not None else "",
                replace_existing_key=replace_existing_key,
                force=force,
                backup=backup,
                already_on=already_on,
            )

            default_model = resolve_launch_model(model, default=CURSOR_DEFAULT_LAUNCH_MODEL)
            if models:
                model_ids = [m.strip() for m in models.split(",") if m.strip()]
            else:
                fetch_key = key if key is not None else (credential or "")
                try:
                    entries = fetch_sference_model_entries(openai_base_url, fetch_key)
                    model_ids = [e["id"] for e in entries]
                except Exception as e:
                    typer.echo(
                        f"note: could not fetch the model catalog ({e}); registering only {default_model}.",
                        err=True,
                    )
                    model_ids = []
            if default_model not in model_ids:
                model_ids.append(default_model)

            # Validates the change set (raises for --sference-only without a
            # cached catalog) before anything is written; apply_on recomputes it.
            cursor_config.build_on_changes(
                reactive,
                openai_base_url=openai_base_url,
                model_ids=model_ids,
                default_model=default_model,
                sference_only=sference_only,
            )
        except cursor_config.CursorConfigError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from None

        if dry_run:
            typer.echo(f"Cursor settings: {db_path}")
            typer.echo(f"openAIBaseUrl: {openai_base_url}")
            typer.echo("useOpenAIKey: true")
            typer.echo(f"default model: {default_model}")
            typer.echo(f"models added to the picker: {len(model_ids)} ({', '.join(model_ids)})")
            if sference_only:
                typer.echo("sference-only: all Cursor-provided models disabled")
            typer.echo(f"key: <redacted> ({key_source})")
            if replace_existing_key:
                typer.echo("replace-existing-key: the currently saved OpenAI key is removed")
            typer.echo("dry run: nothing was written")
            return

        keychain_deleted = False
        if replace_existing_key and cursor_config._keychain_item_present():
            keychain_deleted = cursor_config._keychain_delete()
            if not keychain_deleted:
                typer.echo(
                    "ERROR: could not remove the OpenAI key saved in your macOS Keychain "
                    "(Cursor shadows the settings database with it). Remove it in "
                    "Cursor Settings → Models, then re-run.",
                    err=True,
                )
                raise typer.Exit(code=1)

        new_backup = cursor_config.apply_on(
            conn,
            openai_base_url=openai_base_url,
            api_key=key,
            model_ids=model_ids,
            default_model=default_model,
            sference_only=sference_only,
            replace_existing_key=replace_existing_key,
        )
        if backup is not None:
            # The backup exists to restore the state before the FIRST `launch
            # cursor`; a re-run must not re-snapshot our own config as "prior".
            new_backup["reactive"] = backup["reactive"]
            if key is None:
                new_backup["openai_key_row"] = backup["openai_key_row"]
            else:
                new_backup["openai_key_row"]["existed"] = backup["openai_key_row"]["existed"]
        new_backup["provisioned_key"] = (
            {"id": provisioned["id"], "tag": provisioned["tag"], "name": provisioned["name"]}
            if provisioned
            else (backup or {}).get("provisioned_key")
        )
        new_backup["keychain_item_deleted"] = keychain_deleted
        path = cursor_config.write_backup(new_backup)
    finally:
        conn.close()

    typer.echo("Sference routing is ON for Cursor:")
    typer.echo(f"  base URL: {openai_base_url}")
    typer.echo(f"  default model: {default_model}")
    typer.echo(f"  picker: {len(model_ids)} Sference model(s) enabled")
    typer.echo(f"  key: {key_source}")
    typer.echo(f"  backup for `launch cursor off`: {path}")
    if cursor_config.cursor_running_pid(config_dir) is not None:
        typer.echo("Cursor is running — fully restart it for the change to take effect.")
    else:
        typer.echo(f"Start Cursor: new chats use {default_model} on Sference.")


def _cursor_off(*, revoke_key: bool, force: bool, dry_run: bool) -> None:
    from sference_cli.main import _read_token

    config_dir = cursor_config.default_config_dir()
    db_path = cursor_config.state_db_path(config_dir)
    backup = cursor_config.read_backup()
    if backup is None:
        typer.echo(
            f"No `sference launch cursor` backup found at {cursor_config.backup_path()} — "
            "Sference routing was never turned on (or was already turned off).",
            err=True,
        )
        raise typer.Exit(code=1)
    if not db_path.exists():
        typer.echo(f"Cursor settings not found at {db_path}; nothing to restore.", err=True)
        raise typer.Exit(code=1)
    pid = cursor_config.cursor_running_pid(config_dir)
    if pid is not None and not force:
        typer.echo(
            f"Cursor is running (pid {pid}). Quit Cursor and re-run, or pass --force.",
            err=True,
        )
        raise typer.Exit(code=1)

    if dry_run:
        typer.echo(f"Would restore Cursor settings from {cursor_config.backup_path()} "
                   f"(written {backup.get('written_at', 'unknown')})")
        if revoke_key and backup.get("provisioned_key"):
            typer.echo(f"Would revoke API key '{backup['provisioned_key'].get('name')}'")
        typer.echo("dry run: nothing was written")
        return

    conn = cursor_config.connect(config_dir)
    try:
        try:
            notes = cursor_config.apply_off(conn, backup)
        except cursor_config.CursorConfigError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from None
    finally:
        conn.close()

    if revoke_key and backup.get("provisioned_key"):
        credential = _read_token()
        key_id = backup["provisioned_key"]["id"]
        revoke_base = backup.get("openai_base_url") or resolve_api_base_url(None)
        if credential is not None and _revoke_cursor_api_key(revoke_base, credential, key_id):
            notes.append(f"Revoked API key '{backup['provisioned_key'].get('name')}'.")
        else:
            notes.append(
                f"Could not revoke API key '{backup['provisioned_key'].get('name')}' here — "
                "revoke it in Console → API Keys."
            )

    cursor_config.backup_path().unlink(missing_ok=True)
    typer.echo("Sference routing is OFF for Cursor (settings restored).")
    for note in notes:
        typer.echo(f"  note: {note}")
    if cursor_config.cursor_running_pid(config_dir) is not None:
        typer.echo("Cursor is running — fully restart it for the change to take effect.")


def _cursor_status() -> None:
    config_dir = cursor_config.default_config_dir()
    db_path = cursor_config.state_db_path(config_dir)
    if not db_path.exists():
        typer.echo(f"Cursor settings not found at {db_path}.")
        raise typer.Exit(code=1)
    conn = cursor_config.connect(config_dir)
    try:
        plan_state = cursor_config.read_plan_state(conn)
        try:
            reactive = cursor_config.read_reactive(conn)
        except cursor_config.CursorConfigError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from None
    finally:
        conn.close()
    backup = cursor_config.read_backup()
    ai = reactive.get("aiSettings") if isinstance(reactive.get("aiSettings"), dict) else {}
    base_url = reactive.get("openAIBaseUrl")
    use_key = reactive.get("useOpenAIKey") is True
    has_key = (
        plan_state["openai_key_row"] is not None
        or plan_state["secret_row_present"]
        or cursor_config._keychain_item_present()
    )
    routing_on = bool(use_key and base_url and has_key)

    membership = plan_state.get("membership") or "unknown"
    typer.echo(f"Cursor: {'signed in' if plan_state['signed_in'] else 'signed out'} (plan: {membership})")
    typer.echo(f"Sference routing: {'ON' if routing_on else 'off'}")
    if base_url:
        typer.echo(f"  openAIBaseUrl: {base_url}")
    typer.echo(f"  useOpenAIKey: {use_key}")
    typer.echo(f"  saved key: {cursor_config.stored_key_situation(plan_state)}")
    composer = (ai.get("modelConfig") or {}).get("composer") or {}
    if composer.get("modelName"):
        typer.echo(f"  default model: {composer['modelName']}")
    added = ai.get("userAddedModels") or []
    if added:
        typer.echo(f"  custom models in picker: {', '.join(str(m) for m in added)}")
    if backup is not None:
        typer.echo(f"  `launch cursor off` backup: {cursor_config.backup_path()} (written {backup.get('written_at', 'unknown')})")


def register_launch_commands(app: typer.Typer) -> None:
    launch_app = typer.Typer(help="Launch external tools configured for Sference.", invoke_without_command=True)

    @launch_app.callback()
    def launch_root(ctx: typer.Context) -> None:
        if ctx.invoked_subcommand is None:
            typer.echo(ctx.get_help())
            raise typer.Exit(code=0)

    @launch_app.command(
        "claude",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
        help="Launch Claude Code against the Sference endpoint (gateway model discovery; no proxy).",
    )
    def claude(
        ctx: typer.Context,
        model: Optional[str] = typer.Option(
            None,
            "--model",
            "-m",
            help=f"Pinned catalog model (default: {CLAUDE_DEFAULT_LAUNCH_MODEL} or SFERENCE_MODEL).",
        ),
        base_url: Optional[str] = typer.Option(
            None,
            "--base-url",
            envvar="SFERENCE_BASE_URL",
            help=f"Sference API base URL (default: {DEFAULT_API_BASE_URL}).",
        ),
        no_anthropic: bool = typer.Option(
            False,
            "--no-anthropic",
            help="Deprecated no-op: the direct env-var mode is now the only mode.",
        ),
        models: Optional[str] = typer.Option(
            None,
            "--models",
            help="Comma-separated catalog ids for the /model picker (default: live GET /v1/models).",
        ),
        proxy_port: Optional[int] = typer.Option(
            None,
            "--proxy-port",
            help="Deprecated no-op: there is no proxy anymore.",
        ),
        enable_tool_search: bool = typer.Option(
            False,
            "--enable-tool-search",
            help="Set ENABLE_TOOL_SEARCH=true.",
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            help="Print the env config and command without launching anything.",
        ),
    ) -> None:
        """Run ``claude`` with Sference credentials.

        Points Claude Code at the Sference endpoint (native ``/v1/messages``)
        via ``ANTHROPIC_BASE_URL`` and enables gateway model discovery: the
        live catalog is primed into the picker cache before launch, so
        ``/model`` lists every Sference model the account can serve. No proxy,
        no mitmproxy. Unknown options and trailing args are forwarded to
        Claude Code, e.g. ``sference launch claude -p "fix the bug"``.
        """
        from sference_cli.main import (
            _ensure_api_credential,
            _read_device_credentials,
            _read_token,
            _write_device_credentials,
        )

        _ensure_api_credential()
        resolved_base = resolve_api_base_url(base_url)
        # Device-flow (v2) credentials: refresh the 24 h access token at launch
        # so long sessions start on a fresh grant. Skipped when SFERENCE_API_KEY
        # overrides the file — the env token is not the file's grant, and
        # refreshing would swap identities mid-session.
        device_creds = None if os.environ.get("SFERENCE_API_KEY") else _read_device_credentials()
        if device_creds and device_creds.get("refresh_token"):
            try:
                from sference_cli.device_auth import CLIENT_ID_SFERENCE_CLI, refresh_tokens

                tokens = refresh_tokens(
                    resolved_base, CLIENT_ID_SFERENCE_CLI, device_creds["refresh_token"]
                )
                _write_device_credentials(tokens)
            except Exception:  # noqa: BLE001 (best-effort; the current token may still be valid)
                # The launch proceeds on the file's token. When that token is
                # already expired, a failed refresh means the session WILL 401 —
                # say so now rather than mid-task.
                if float(device_creds.get("expires_at") or 0) <= time.time():
                    typer.echo(
                        "warning: could not refresh the expired device token; re-run `sference auth login`",
                        err=True,
                    )

        api_key = _read_token()
        if api_key is None:
            raise typer.Exit(code=1)

        if no_anthropic:
            typer.echo("note: --no-anthropic is deprecated; the direct env-var mode is now the only mode.", err=True)
        if proxy_port is not None:
            typer.echo("note: --proxy-port is deprecated; there is no proxy anymore.", err=True)

        forwarded = list(ctx.args)
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]

        # Resolve the picker model set.
        if models:
            model_entries = [{"id": m.strip()} for m in models.split(",") if m.strip()]
        else:
            try:
                model_entries = fetch_sference_model_entries(resolved_base, api_key)
            except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError) as e:
                typer.echo(f"ERROR: could not fetch models from {resolved_base}/v1/models: {e}", err=True)
                typer.echo("Pass --models explicitly to bypass the lookup.", err=True)
                raise typer.Exit(code=1) from None
            if not model_entries:
                typer.echo("ERROR: /v1/models returned no text-generation models for this account.", err=True)
                raise typer.Exit(code=1)

        launch_claude_code(
            api_key=api_key,
            base_url=resolved_base,
            model=resolve_launch_model(model, default=CLAUDE_DEFAULT_LAUNCH_MODEL),
            model_entries=model_entries,
            enable_tool_search=enable_tool_search,
            claude_args=forwarded,
            dry_run=dry_run,
        )

    @launch_app.command(
        "pi",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
        help="Launch Pi with Sference OpenAI-compatible routing.",
    )
    def pi(
        ctx: typer.Context,
        model: Optional[str] = typer.Option(
            None,
            "--model",
            "-m",
            help=f"Catalog model id (default: {DEFAULT_LAUNCH_MODEL} or SFERENCE_MODEL).",
        ),
        base_url: Optional[str] = typer.Option(
            None,
            "--base-url",
            envvar="SFERENCE_BASE_URL",
            help=f"Sference API base URL (default: {DEFAULT_API_BASE_URL}).",
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            help="Print provider config and command without launching Pi.",
        ),
    ) -> None:
        """Run ``pi`` with Sference credentials.

        Writes a ``sference`` provider block to ``~/.pi/agent/models.json`` using
        pi's ``openai-responses`` API (the Sference ``/v1/responses`` endpoint)
        with ``baseUrl``, ``apiKey``, and the full list of Sference-provided
        models (fetched live from ``GET /v1/models``), then execs
        ``pi --provider sference --model <id> --models sference/**``. The
        ``--models sference/**`` flag scopes the ``/model`` picker (and Ctrl+P
        cycling) to Sference models only, so other configured providers are
        hidden by default. ``**`` (globstar) is required because pi matches
        patterns against ``provider/modelId`` and Sference model ids contain a
        slash; ``sference/*`` would match nothing and show every provider. The
        picker mirrors the currently-available catalog exactly; ``--model``
        selects the active one and is never injected into the picker when absent
        from a reachable catalog. If the catalog fetch fails (offline), only the
        chosen model is registered.
        Unknown options and trailing args are forwarded to Pi, e.g.
        ``sference launch pi -- /path/to/project``.
        """
        from sference_cli.main import _ensure_api_credential, _read_token

        _ensure_api_credential()
        api_key = _read_token()
        if api_key is None:
            raise typer.Exit(code=1)

        forwarded = list(ctx.args)
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]

        launch_pi(
            api_key=api_key,
            base_url=resolve_openai_base_url(base_url),
            model=resolve_launch_model(model),
            pi_args=forwarded,
            dry_run=dry_run,
        )

    @launch_app.command(
        "opencode",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
        help="Launch opencode with Sference OpenAI-compatible routing.",
    )
    def opencode(
        ctx: typer.Context,
        model: Optional[str] = typer.Option(
            None,
            "--model",
            "-m",
            help=f"Catalog model id (default: {DEFAULT_LAUNCH_MODEL} or SFERENCE_MODEL).",
        ),
        base_url: Optional[str] = typer.Option(
            None,
            "--base-url",
            envvar="SFERENCE_BASE_URL",
            help=f"Sference API base URL (default: {DEFAULT_API_BASE_URL}).",
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            help="Print provider config and command without launching opencode.",
        ),
    ) -> None:
        """Run ``opencode`` with Sference credentials.

        Merges a ``sference`` provider (OpenAI-compatible, ``/v1/chat/completions``)
        into ``~/.config/opencode/opencode.json`` and execs
        ``opencode --model sference/<id>``. The API key is injected via the
        ``SFERENCE_API_KEY`` env var (referenced as ``{env:SFERENCE_API_KEY}`` in
        the config, so no secret is written to disk). Unknown options and trailing
        args are forwarded to opencode, e.g. ``sference launch opencode --prompt "..."``.
        """
        from sference_cli.main import _ensure_api_credential, _read_token

        _ensure_api_credential()
        api_key = _read_token()
        if api_key is None:
            raise typer.Exit(code=1)

        forwarded = list(ctx.args)
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]

        launch_opencode(
            api_key=api_key,
            base_url=resolve_openai_base_url(base_url),
            model=resolve_launch_model(model),
            opencode_args=forwarded,
            dry_run=dry_run,
        )

    @launch_app.command(
        "codex",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
        help="Launch OpenAI Codex CLI with Sference Responses-API routing.",
    )
    def codex(
        ctx: typer.Context,
        model: Optional[str] = typer.Option(
            None,
            "--model",
            "-m",
            help=f"Catalog model id (default: {DEFAULT_LAUNCH_MODEL} or SFERENCE_MODEL).",
        ),
        base_url: Optional[str] = typer.Option(
            None,
            "--base-url",
            envvar="SFERENCE_BASE_URL",
            help=f"Sference API base URL (default: {DEFAULT_API_BASE_URL}).",
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            help="Print provider config and command without launching Codex.",
        ),
    ) -> None:
        """Run ``codex`` with Sference credentials.

        Defines a ``sference`` model provider (``/v1/responses``) through
        ``codex -c`` overrides, so ``~/.codex/config.toml`` is never modified.
        The API key is injected via the ``SFERENCE_API_KEY`` env var. Unknown
        options and trailing args are forwarded to Codex, e.g.
        ``sference launch codex -- exec "fix the bug"`` or
        ``sference launch codex -- -c model_reasoning_effort=high``.
        """
        from sference_cli.main import _ensure_api_credential, _read_token

        _ensure_api_credential()
        api_key = _read_token()
        if api_key is None:
            raise typer.Exit(code=1)

        forwarded = list(ctx.args)
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]

        launch_codex(
            api_key=api_key,
            base_url=resolve_openai_base_url(base_url),
            model=resolve_launch_model(model),
            codex_args=forwarded,
            dry_run=dry_run,
        )

    cursor_app = typer.Typer(
        help="Configure Cursor to chat on Sference (its own-API-key mode).",
        invoke_without_command=True,
    )

    @cursor_app.callback()
    def cursor_root(
        ctx: typer.Context,
        model: Optional[str] = typer.Option(
            None,
            "--model",
            "-m",
            help=f"Default model for new chats (default: {CURSOR_DEFAULT_LAUNCH_MODEL} or SFERENCE_MODEL).",
        ),
        models: Optional[str] = typer.Option(
            None,
            "--models",
            help="Comma-separated catalog ids for the picker (default: live GET /v1/models).",
        ),
        base_url: Optional[str] = typer.Option(
            None,
            "--base-url",
            envvar="SFERENCE_BASE_URL",
            help=f"Sference API base URL (default: {DEFAULT_API_BASE_URL}).",
        ),
        api_key: Optional[str] = typer.Option(
            None,
            "--api-key",
            help="Sference API key to store in Cursor (default: your credential, or a newly created one).",
        ),
        sference_only: bool = typer.Option(
            False,
            "--sference-only",
            help="Disable all Cursor-provided models, leaving only Sference models in the picker.",
        ),
        replace_existing_key: bool = typer.Option(
            False,
            "--replace-existing-key",
            help="Replace an OpenAI key already saved in Cursor (the old key cannot be restored).",
        ),
        force: bool = typer.Option(
            False,
            "--force",
            help="Write even while Cursor is running (you must fully restart Cursor afterwards).",
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            help="Print what would change without writing anything.",
        ),
    ) -> None:
        """Turn Sference routing on for Cursor (default action)."""
        if ctx.invoked_subcommand is not None:
            return
        _cursor_on(
            model=model,
            models=models,
            base_url=base_url,
            api_key=api_key,
            sference_only=sference_only,
            replace_existing_key=replace_existing_key,
            force=force,
            dry_run=dry_run,
        )

    @cursor_app.command("on", help="Turn Sference routing on (same as `sference launch cursor`).")
    def cursor_on_cmd(
        model: Optional[str] = typer.Option(None, "--model", "-m"),
        models: Optional[str] = typer.Option(None, "--models"),
        base_url: Optional[str] = typer.Option(None, "--base-url", envvar="SFERENCE_BASE_URL"),
        api_key: Optional[str] = typer.Option(None, "--api-key"),
        sference_only: bool = typer.Option(False, "--sference-only"),
        replace_existing_key: bool = typer.Option(False, "--replace-existing-key"),
        force: bool = typer.Option(False, "--force"),
        dry_run: bool = typer.Option(False, "--dry-run"),
    ) -> None:
        _cursor_on(
            model=model,
            models=models,
            base_url=base_url,
            api_key=api_key,
            sference_only=sference_only,
            replace_existing_key=replace_existing_key,
            force=force,
            dry_run=dry_run,
        )

    @cursor_app.command("off", help="Restore Cursor's settings from before `launch cursor`.")
    def cursor_off_cmd(
        revoke_key: bool = typer.Option(
            False,
            "--revoke-key",
            help="Also revoke the Sference API key that was created for Cursor.",
        ),
        force: bool = typer.Option(False, "--force"),
        dry_run: bool = typer.Option(False, "--dry-run"),
    ) -> None:
        _cursor_off(revoke_key=revoke_key, force=force, dry_run=dry_run)

    @cursor_app.command("status", help="Show whether Cursor chats on Sference right now.")
    def cursor_status_cmd() -> None:
        _cursor_status()

    launch_app.add_typer(cursor_app, name="cursor")

    app.add_typer(launch_app, name="launch")
