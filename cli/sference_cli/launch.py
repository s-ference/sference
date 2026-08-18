"""Launch external agent tools against the Sference API."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.error
from pathlib import Path
from typing import Optional

import typer

from .proxy import fetch_sference_model_entries, fetch_sference_models, launch_claude_via_proxy

DEFAULT_LAUNCH_MODEL = "zai-org/GLM-5.2"
DEFAULT_API_BASE_URL = "https://api.sference.com"


def resolve_api_base_url(explicit: Optional[str]) -> str:
    if explicit:
        return explicit.rstrip("/")
    env = os.environ.get("SFERENCE_BASE_URL")
    if env:
        return env.strip().rstrip("/")
    return DEFAULT_API_BASE_URL


def resolve_launch_model(explicit: Optional[str]) -> str:
    """Default catalog model for ``launch`` subcommands (--model overrides)."""
    if explicit:
        return explicit
    env = os.environ.get("SFERENCE_MODEL")
    if env and env.strip():
        return env.strip()
    return DEFAULT_LAUNCH_MODEL


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
    if enable_tool_search:
        env["ENABLE_TOOL_SEARCH"] = "true"
    else:
        env.pop("ENABLE_TOOL_SEARCH", None)
    return env


def launch_claude_code(
    *,
    api_key: str,
    base_url: str,
    model: str,
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
    cmd = [claude_bin, *claude_args]

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
        if enable_tool_search:
            typer.echo("ENABLE_TOOL_SEARCH=true")
        typer.echo(f"command: {' '.join(cmd)}")
        return

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


def _write_opencode_config(*, base_url: str, model: str, models: set[str]) -> Path:
    """Merge a ``sference`` OpenAI-compatible provider into the opencode config.

    Reads ``~/.config/opencode/opencode.json`` (creating it if absent), sets
    ``provider.sference`` to a block pointing at the Sference OpenAI-compatible
    endpoint (``/v1/chat/completions``), and writes it back. The user's existing
    config — other providers, keybinds, theme — is preserved; only
    ``provider.sference`` is (over)written. If the existing file is not
    valid JSON, we refuse to clobber it rather than silently destroying the
    user's opencode config.

    All models in ``models`` are written to the provider's ``models`` block so
    they appear in opencode's model picker. ``model`` is the default — set as
    the top-level ``model`` key so opencode uses Sference by default on startup.

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
            m: {"name": m}
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
        fetched = fetch_sference_models(base_url, api_key)
    except Exception:
        fetched = set()
    models = fetched if fetched else set()

    config_path = _write_opencode_config(base_url=base_url, model=model, models=models)
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
        help="Launch Claude Code with a local Sference proxy (default). Use --no-anthropic for the direct env-var mode.",
    )
    def claude(
        ctx: typer.Context,
        model: Optional[str] = typer.Option(
            None,
            "--model",
            "-m",
            help="Proxy mode: catalog id to inject into the /model picker. --no-anthropic: the pinned model (default: "
            f"{DEFAULT_LAUNCH_MODEL} or SFERENCE_MODEL).",
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
            help="Disable the proxy and route everything directly to the Sference endpoint "
            "(today's ANTHROPIC_BASE_URL env-var behavior; no hybrid routing, no /model picker).",
        ),
        models: Optional[str] = typer.Option(
            None,
            "--models",
            help="Proxy mode: comma-separated catalog ids to route and inject into the picker "
            "(default: live GET /v1/models). Ignored with --no-anthropic.",
        ),
        proxy_port: Optional[int] = typer.Option(
            None,
            "--proxy-port",
            help="Local mitmproxy port (default: auto-pick a free port). Proxy mode only.",
        ),
        enable_tool_search: bool = typer.Option(
            False,
            "--enable-tool-search",
            help="Set ENABLE_TOOL_SEARCH=true (only meaningful with --no-anthropic; the proxy "
            "keeps first-party detection on, so tool search already works).",
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            help="Print the proxy/env config and command without launching anything.",
        ),
    ) -> None:
        """Run ``claude`` with Sference credentials.

        By default a local mitmproxy forward proxy routes Sference models to
        ``api.sference.com`` (native ``/v1/messages``, body untranslated) while
        Claude Code's first-party detection stays on — Sference models appear in
        the ``/model`` picker and real Claude models pass through to Anthropic.
        ``--no-anthropic`` switches to the direct env-var mode (everything to one
        Sference model). Unknown options and trailing args are forwarded to
        Claude Code, e.g. ``sference launch claude -p "fix the bug"``.
        """
        from sference_cli.main import (
            CREDENTIALS_PATH,
            _ensure_api_credential,
            _read_device_credentials,
            _read_token,
        )

        _ensure_api_credential()
        api_key = _read_token()
        if api_key is None:
            raise typer.Exit(code=1)
        # Device-flow (v2) credentials let the proxy refresh the 24 h access
        # token mid-session; legacy API-key credentials have no refresh token.
        # Skip when SFERENCE_API_KEY overrides the file — the env token is not
        # the file's grant, and refreshing would swap identities mid-session.
        device_creds = None if os.environ.get("SFERENCE_API_KEY") else _read_device_credentials()

        forwarded = list(ctx.args)
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]

        resolved_base = resolve_api_base_url(base_url)

        if no_anthropic:
            launch_claude_code(
                api_key=api_key,
                base_url=resolved_base,
                model=resolve_launch_model(model),
                enable_tool_search=enable_tool_search,
                claude_args=forwarded,
                dry_run=dry_run,
            )
            return

        if enable_tool_search:
            typer.echo(
                "note: --enable-tool-search is a no-op in proxy mode (first-party "
                "detection is on, so tool search already works).",
                err=True,
            )

        # Proxy mode: resolve the routable model set.
        if models:
            model_set = {m.strip() for m in models.split(",") if m.strip()}
        elif model:
            model_set = {model}
        else:
            try:
                model_set = fetch_sference_models(resolved_base, api_key)
            except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError) as e:
                typer.echo(f"ERROR: could not fetch models from {resolved_base}/v1/models: {e}", err=True)
                typer.echo("Pass --models or --model explicitly to bypass the lookup.", err=True)
                raise typer.Exit(code=1) from None
            if not model_set:
                typer.echo("ERROR: /v1/models returned no text-generation models for this account.", err=True)
                raise typer.Exit(code=1)

        launch_claude_via_proxy(
            api_key=api_key,
            base_url=resolved_base,
            models=model_set,
            claude_args=forwarded,
            dry_run=dry_run,
            proxy_port=proxy_port,
            refresh_token=device_creds["refresh_token"] if device_creds else None,
            credentials_path=str(CREDENTIALS_PATH) if device_creds else None,
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

    app.add_typer(launch_app, name="launch")
