"""Launch external agent tools against the Sference API."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

DEFAULT_LAUNCH_MODEL = "moonshotai/Kimi-K2.6"
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


def resolve_openai_base_url_for_pi(explicit: Optional[str]) -> str:
    """OpenAI-compatible base URL with ``/v1`` for Pi ``models.json``."""
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


def _write_pi_models_json(*, base_url: str, api_key: str, model: str) -> Path:
    """Write or update ``~/.pi/agent/models.json`` with a Sference provider."""
    path = _pi_models_json_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    provider = {
        "api": "openai-completions",
        "baseUrl": base_url,
        "apiKey": api_key,
        "models": [{"id": model, "name": model}],
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

    models_json = _write_pi_models_json(base_url=base_url, api_key=api_key, model=model)
    env = os.environ.copy()
    # Pi reads providers from ~/.pi/agent/models.json; no env vars needed.
    cmd = [pi_bin, "--provider", "sference", "--model", model, *pi_args]

    if dry_run:
        typer.echo(f"Wrote provider 'sference' to {models_json}")
        typer.echo(f"baseUrl: {base_url}")
        typer.echo(f"model: {model}")
        typer.echo("apiKey: <redacted>")
        typer.echo(f"command: {' '.join(cmd)}")
        return

    if sys.platform == "win32":
        raise SystemExit(subprocess.call(cmd, env=env))

    os.execvpe(pi_bin, cmd, env)


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
        help="Launch Claude Code with Sference API routing.",
    )
    def claude(
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
            help=f"Sference API base URL (default: {DEFAULT_API_BASE_URL} or SFERENCE_BASE_URL).",
        ),
        enable_tool_search: bool = typer.Option(
            False,
            "--enable-tool-search",
            help="Set ENABLE_TOOL_SEARCH=true for Claude Code on a custom API host.",
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            help="Print env and command without launching Claude Code.",
        ),
    ) -> None:
        """Run ``claude`` with Sference credentials.

        Uses ``~/.sference/credentials.json`` or ``SFERENCE_API_KEY``. Unknown options
        and trailing args are forwarded to Claude Code, e.g.
        ``sference launch claude --resume`` or ``sference launch claude -p "fix the bug"``.
        """
        from sference_cli.main import _ensure_api_credential, _read_token

        _ensure_api_credential()
        api_key = _read_token()
        if api_key is None:
            raise typer.Exit(code=1)

        forwarded = list(ctx.args)
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]

        launch_claude_code(
            api_key=api_key,
            base_url=resolve_api_base_url(base_url),
            model=resolve_launch_model(model),
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
            help=f"Sference API base URL (default: {DEFAULT_API_BASE_URL} or SFERENCE_BASE_URL).",
        ),
        dry_run: bool = typer.Option(
            False,
            "--dry-run",
            help="Print provider config and command without launching Pi.",
        ),
    ) -> None:
        """Run ``pi`` with Sference credentials.

        Writes a ``sference`` provider block to ``~/.pi/agent/models.json`` with
        ``baseUrl``, ``apiKey``, and the chosen ``model``, then execs
        ``pi --provider sference --model <id>``. Unknown options and trailing args
        are forwarded to Pi, e.g. ``sference launch pi -- /path/to/project``.
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
            base_url=resolve_openai_base_url_for_pi(base_url),
            model=resolve_launch_model(model),
            pi_args=forwarded,
            dry_run=dry_run,
        )

    # Hidden from `sference --help`; documented for Claude Code / Pi setup only.
    app.add_typer(launch_app, name="launch", hidden=True)
