from __future__ import annotations

import json
import logging
import os
import sys
import time
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional, TypeVar

import typer

import sference_sdk
from sference_sdk import COMPLETION_WINDOWS, ApiError, SferenceClient

from sference_cli import __version__ as _CLI_VERSION
from sference_sdk.checkpoint import clear_checkpoint, load_checkpoint, save_checkpoint

from . import stream_cache as stream_cache_mod
from .launch import register_launch_commands

_T = TypeVar("_T")

_logger = logging.getLogger(__name__)


app = typer.Typer(help="sference CLI", invoke_without_command=True)
auth_app = typer.Typer(help="Auth commands", invoke_without_command=True)
batch_app = typer.Typer(help="Batch commands", invoke_without_command=True)
stream_app = typer.Typer(help="Stream commands", invoke_without_command=True)
responses_app = typer.Typer(help="Responses commands", invoke_without_command=True)
models_app = typer.Typer(help="List models (GET /v1/models)", invoke_without_command=True)
embeddings_app = typer.Typer(help="Embeddings commands", invoke_without_command=True)
app.add_typer(auth_app, name="auth")
app.add_typer(batch_app, name="batch")
app.add_typer(stream_app, name="stream")
app.add_typer(responses_app, name="responses")
app.add_typer(models_app, name="models")
app.add_typer(embeddings_app, name="embeddings")
register_launch_commands(app)

CREDENTIALS_PATH = Path.home() / ".sference" / "credentials.json"

DEFAULT_CONSOLE_URL = "https://app.sference.com"


def _write_token(token: str) -> None:
    # The credential file holds an API key; keep it owner-only (0700 dir / 0600 file)
    # so other local users cannot read it. chmod is a no-op on Windows.
    CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(CREDENTIALS_PATH.parent, 0o700)
    except OSError:
        pass
    CREDENTIALS_PATH.write_text(json.dumps({"token": token}), encoding="utf-8")
    try:
        os.chmod(CREDENTIALS_PATH, 0o600)
    except OSError:
        pass


def _read_token() -> str | None:
    env = os.environ.get("SFERENCE_API_KEY")
    if env:
        return env.strip() or None
    if not CREDENTIALS_PATH.exists():
        return None
    try:
        payload = json.loads(CREDENTIALS_PATH.read_text(encoding="utf-8"))
        tok = payload.get("token")
        return tok.strip() if isinstance(tok, str) else None
    except Exception:
        return None


def _client(base_url: Optional[str] = None, *, timeout: float | None = None) -> SferenceClient:
    env_base_url = os.environ.get("SFERENCE_BASE_URL")
    if env_base_url and (base_url is None or base_url == "https://api.sference.com"):
        base_url = env_base_url
    return SferenceClient(base_url=base_url, api_key=_read_token(), timeout=timeout)


def _ensure_api_credential() -> None:
    if _read_token() is None:
        typer.echo(
            "No API credential configured.\n"
            "  • Run: sference auth login\n"
            "  • Or: sference auth login --api-key 'sk_...'\n"
            "  • Or set environment variable SFERENCE_API_KEY",
            err=True,
        )
        raise typer.Exit(code=1)


def _call_api(fn: Callable[[], _T]) -> _T:
    try:
        return fn()
    except ApiError as exc:
        err = str(exc)
        if err.startswith("401:"):
            typer.echo(
                "Unauthorized (401). Create a key in Console → API Keys (while signed in), then run:\n"
                "  sference auth login --api-key 'sk_...'\n"
                "If SFERENCE_API_KEY is set, it overrides ~/.sference/credentials.json.\n"
                "If you already saved a key, it may be revoked or not registered for this API/database.",
                err=True,
            )
            raise typer.Exit(code=1) from None
        typer.echo(err, err=True)
        raise typer.Exit(code=1) from None


def _batch_id_and_status(obj: object) -> tuple[str, str]:
    if hasattr(obj, "model_dump"):
        d = obj.model_dump()
        return str(d["id"]), str(d["status"])
    bid = getattr(obj, "id", None)
    st = getattr(obj, "status", None)
    return str(bid), str(st)


def _get_batch_or_missing(client: SferenceClient, batch_id: str) -> Any | None:
    try:
        return client.get_batch(batch_id)
    except ApiError as exc:
        err = str(exc)
        if err.startswith("404:"):
            return None
        if err.startswith("401:"):
            typer.echo(
                "Unauthorized (401). Create a key in Console → API Keys (while signed in), then run:\n"
                "  sference auth login --api-key 'sk_...'\n"
                "If SFERENCE_API_KEY is set, it overrides ~/.sference/credentials.json.\n"
                "If you already saved a key, it may be revoked or not registered for this API/database.",
                err=True,
            )
            raise typer.Exit(code=1) from None
        typer.echo(err, err=True)
        raise typer.Exit(code=1) from None


def _print(value: object, as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(value, indent=2, default=str))
    else:
        typer.echo(value)


def _list_models(*, client: SferenceClient, as_json: bool) -> None:
    payload = _call_api(client.list_models)
    if as_json:
        typer.echo(json.dumps(payload, indent=2, default=str))
        return
    _print_models_table(payload)


def _capability_flag(caps: object, key: str) -> str:
    if not isinstance(caps, dict):
        return "—"
    entry = caps.get(key)
    if isinstance(entry, dict) and "supported" in entry:
        return "yes" if entry["supported"] else "no"
    return "—"


def _modality_label(raw: object) -> str:
    if not isinstance(raw, str) or not raw:
        return "—"
    if raw.startswith("text_"):
        return raw.removeprefix("text_")
    return raw


def _print_models_table(payload: object) -> None:
    if not isinstance(payload, dict):
        typer.echo(str(payload))
        return
    rows = payload.get("data")
    if not isinstance(rows, list):
        typer.echo(json.dumps(payload, indent=2, default=str))
        return
    if not rows:
        typer.echo("No models.")
        return

    w_id, w_name, w_mod, w_ctx, w_vis, w_pdf, w_think, w_tools = 34, 18, 10, 8, 6, 4, 8, 5
    typer.echo(
        f"{'id':<{w_id}} {'display_name':<{w_name}} {'modality':<{w_mod}} {'context':>{w_ctx}} "
        f"{'vision':>{w_vis}} {'pdf':>{w_pdf}} {'thinking':>{w_think}} {'tools':>{w_tools}}"
    )
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("id", ""))
        if len(model_id) > w_id:
            model_id = model_id[: w_id - 3] + "..."
        display_name = str(row.get("display_name") or "")
        if len(display_name) > w_name:
            display_name = display_name[: w_name - 1] + "…"
        modality = _modality_label(row.get("modality"))
        if len(modality) > w_mod:
            modality = modality[: w_mod - 1] + "…"
        ctx_raw = row.get("context_tokens")
        context = str(ctx_raw) if isinstance(ctx_raw, int) else "—"
        caps = row.get("capabilities")
        vision = _capability_flag(caps, "image_input")
        pdf = _capability_flag(caps, "pdf_input")
        thinking = _capability_flag(caps, "thinking")
        tools = _capability_flag(caps, "tools")
        typer.echo(
            f"{model_id:<{w_id}} {display_name:<{w_name}} {modality:<{w_mod}} {context:>{w_ctx}} "
            f"{vision:>{w_vis}} {pdf:>{w_pdf}} {thinking:>{w_think}} {tools:>{w_tools}}"
        )


def _window_choice(value: str) -> str:
    if value not in COMPLETION_WINDOWS:
        allowed = ", ".join(f'"{w}"' for w in COMPLETION_WINDOWS)
        raise typer.BadParameter(f"Window must be one of {allowed}.", param_hint="--window")
    return value


@app.callback()
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Print version (from package metadata; same as the release tag on PyPI).",
        is_eager=True,
    ),
) -> None:
    """Print help when invoked without a command."""
    if version:
        typer.echo(f"sference-cli {_CLI_VERSION} (sference-sdk {sference_sdk.__version__})")
        raise typer.Exit(code=0)
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0)


@auth_app.callback()
def _auth_root(ctx: typer.Context) -> None:
    """Print help when invoked without a command."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0)


@batch_app.callback()
def _batch_root(ctx: typer.Context) -> None:
    """Print help when invoked without a command."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0)


@stream_app.callback()
def _stream_root(ctx: typer.Context) -> None:
    """Print help when invoked without a command."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0)


@responses_app.callback()
def _responses_root(ctx: typer.Context) -> None:
    """Print help when invoked without a command."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0)


@models_app.callback()
def _models_root(
    ctx: typer.Context,
    as_json: bool = typer.Option(False, "--json", help="Print GET /v1/models JSON."),
    base_url: str = typer.Option("https://api.sference.com"),
) -> None:
    """List models (OpenAI-compatible GET /v1/models)."""
    if ctx.invoked_subcommand is not None:
        return
    _ensure_api_credential()
    client = _client(base_url)
    _list_models(client=client, as_json=as_json)


@models_app.command("list")
def models_list(
    as_json: bool = typer.Option(False, "--json", help="Print GET /v1/models JSON."),
    base_url: str = typer.Option("https://api.sference.com"),
) -> None:
    """List models (OpenAI-compatible GET /v1/models)."""
    _ensure_api_credential()
    client = _client(base_url)
    _list_models(client=client, as_json=as_json)


@embeddings_app.command("create")
def embeddings_create(
    model: str = typer.Option(..., "--model"),
    input_text: list[str] = typer.Option(..., "--input", help="Text to embed (repeat for multiple strings)."),
    encoding_format: str = typer.Option("float", "--encoding-format", help='"float" or "base64".'),
    dimensions: int | None = typer.Option(None, "--dimensions"),
    timeout: float = typer.Option(600.0, "--timeout", help="HTTP read timeout in seconds."),
    base_url: str = typer.Option("https://api.sference.com"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Create embeddings (POST /v1/embeddings)."""
    _ensure_api_credential()
    if encoding_format not in ("float", "base64"):
        typer.echo('encoding_format must be "float" or "base64".', err=True)
        raise typer.Exit(code=1)
    inp: str | list[str] = input_text[0] if len(input_text) == 1 else input_text
    client = _client(base_url, timeout=timeout)
    resp = _call_api(
        lambda: client.create_embeddings(
            model=model,
            input=inp,
            encoding_format=encoding_format,  # type: ignore[arg-type]
            dimensions=dimensions,
        )
    )
    _print(resp.model_dump(), as_json or True)


@auth_app.command("login")
def auth_login(
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="API key (sk_...) or JWT. Non-interactive: saves immediately (e.g. CI).",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print authenticated user as JSON after validation."),
    validate: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Validate the credential by calling GET /v1/auth/me and print the authenticated user.",
    ),
    console_url: Optional[str] = typer.Option(
        None,
        "--console-url",
        envvar="SFERENCE_CONSOLE_URL",
        help="Console base URL for the browser step (default: https://app.sference.com).",
    ),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not open a browser; print the URL and prompt only."),
) -> None:
    """Authenticate: store an API key for API requests (Baseten-style: optional --api-key for non-interactive)."""
    if api_key is not None:
        key = api_key.strip()
        if not key:
            typer.echo("Empty --api-key.", err=True)
            raise typer.Exit(code=1)
        _write_token(key)
        typer.echo(f"Credentials saved to {CREDENTIALS_PATH}")
        if validate:
            client = _client(None)
            me = _call_api(client.get_me)
            if as_json:
                _print(me, True)
            else:
                if isinstance(me, dict) and "username" in me and "role" in me:
                    typer.echo(f"Authenticated as {me['username']} ({me['role']}).")
                else:
                    typer.echo(f"Authenticated. {me}")
        return

    base = (console_url or DEFAULT_CONSOLE_URL).rstrip("/")
    login_url = f"{base}/login"
    api_keys_url = f"{base}/api-keys"

    if not no_browser:
        typer.echo(f"Opening {login_url} in your browser...")
        try:
            webbrowser.open(login_url)
        except Exception:
            typer.echo("Could not open the browser automatically.")

    typer.echo("")
    typer.echo("After signing in:")
    typer.echo(f"  1. Open {api_keys_url}")
    typer.echo("  2. Create an API key")
    typer.echo("  3. Paste it below")
    typer.echo("")
    token = typer.prompt("API key", hide_input=True)
    if not token.strip():
        typer.echo("No API key provided.", err=True)
        raise typer.Exit(code=1)
    _write_token(token.strip())
    typer.echo(f"Credentials saved to {CREDENTIALS_PATH}")
    if validate:
        client = _client(None)
        me = _call_api(client.get_me)
        if as_json:
            _print(me, True)
        else:
            if isinstance(me, dict) and "username" in me and "role" in me:
                typer.echo(f"Authenticated as {me['username']} ({me['role']}).")
            else:
                typer.echo(f"Authenticated. {me}")


@auth_app.command("me")
def auth_me(
    base_url: str = typer.Option("https://api.sference.com"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    _ensure_api_credential()
    client = _client(base_url)
    me = _call_api(client.get_me)
    _print(me, as_json)


@responses_app.command("create")
def responses_create(
    model: str = typer.Option(..., "--model"),
    content: str = typer.Option(..., "--content"),
    include_reasoning: bool = typer.Option(
        True,
        "--include-reasoning/--no-include-reasoning",
        help="Include extracted reasoning in the response output (default: true).",
    ),
    enable_thinking: bool | None = typer.Option(
        None,
        "--enable-thinking/--disable-thinking",
        help="Qwen3: forward to tokenizer apply_chat_template. Omit for tokenizer default.",
    ),
    background: bool = typer.Option(
        False,
        "--background/--no-background",
        help="Submit asynchronously and return immediately. Default blocks until the response is terminal (matches POST /v1/responses).",
    ),
    timeout: float = typer.Option(
        600.0,
        "--timeout",
        help="HTTP read timeout in seconds.",
    ),
    base_url: str = typer.Option("https://api.sference.com"),
) -> None:
    _ensure_api_credential()
    client = _client(base_url, timeout=timeout)
    resp = _call_api(
        lambda: client.create_response(
            model=model,
            input=[{"role": "user", "content": content}],
            include_reasoning=include_reasoning,
            enable_thinking=enable_thinking,
            background=background,
        )
    )
    _print(resp.model_dump(), True)


@responses_app.command("result")
def responses_result(
    id: str = typer.Option(..., "--id"),
    poll_interval: float = typer.Option(
        0.5,
        "--poll-interval",
        help="Seconds between status polls while waiting for completion.",
    ),
    base_url: str = typer.Option("https://api.sference.com"),
) -> None:
    """Wait for a response to reach a terminal status and print the final JSON.

    Use Ctrl-C to stop waiting.
    """
    _ensure_api_credential()
    client = _client(base_url)
    try:
        while True:
            resp = _call_api(lambda: client.get_response(id))
            d = resp.model_dump()
            status = d.get("status") if isinstance(d, dict) else None
            if status in ("completed", "failed", "cancelled"):
                _print(d, True)
                return
            time.sleep(max(0.01, poll_interval))
    except KeyboardInterrupt:
        raise typer.Exit(code=130) from None


@responses_app.command("tail")
def responses_tail(
    stream_id: Optional[str] = typer.Option(
        None,
        "--stream-id",
        help="Optional stream UUID; omit to tail non-stream completion events only.",
    ),
    consumer: str = typer.Option("default", "--consumer", help="Checkpoint namespace for resume."),
    from_latest: bool = typer.Option(
        False,
        "--from-latest",
        help="Ignore saved checkpoint and start from the latest events page.",
    ),
    no_checkpoint: bool = typer.Option(False, "--no-checkpoint", help="Do not read or write checkpoints."),
    poll_interval: float = typer.Option(
        5.0,
        "--poll-interval",
        help="Seconds to long-poll while catching up (capped at 30s server-side).",
    ),
    base_url: str = typer.Option("https://api.sference.com"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Print completion events as JSONL (GET /v1/responses/events). Ctrl-C to stop.

    Omit --stream-id for standalone/batch completions; pass --stream-id to scope to a stream.
    """
    _ = as_json
    _ensure_api_credential()
    client = _client(base_url)
    bu = base_url.rstrip("/")
    ck = stream_id if stream_id else "__responses_events__"

    if from_latest and not no_checkpoint:
        clear_checkpoint(bu, ck, consumer)

    last: str | None = None
    if not no_checkpoint and not from_latest:
        last = load_checkpoint(bu, ck, consumer)

    wait_ms = int(min(poll_interval, 30.0) * 1000)
    try:
        while True:

            def fetch() -> Any:
                return client.list_responses_events(
                    stream_id=stream_id,
                    limit=50,
                    starting_after=last,
                    wait_ms=wait_ms,
                )

            page = _call_api(fetch)
            if not page.data:
                time.sleep(max(0.001, poll_interval))
                continue
            for ev in page.data:
                typer.echo(json.dumps(ev.model_dump(), default=str))
                last = ev.completion_id
                if not no_checkpoint:
                    save_checkpoint(bu, ck, consumer, ev.completion_id)
    except KeyboardInterrupt:
        typer.echo("Interrupted.", err=True)
        raise typer.Exit(code=130) from None

@batch_app.command("list")
def batch_list(
    base_url: str = typer.Option("https://api.sference.com"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    _ensure_api_credential()
    client = _client(base_url)
    payload = _call_api(lambda: client.list_batches().model_dump())
    if as_json:
        _print(payload, True)
        return
    if not isinstance(payload, dict):
        typer.echo(str(payload))
        return
    items = payload.get("items")
    if not isinstance(items, list):
        typer.echo(str(payload))
        return
    if not items:
        typer.echo("No batches.")
        return
    w_id, w_status, w_reqs, w_tok = 36, 12, 6, 10
    typer.echo(
        f"{'id':<{w_id}} {'status':<{w_status}} {'reqs':>{w_reqs}} {'tokens':>{w_tok}}"
    )
    for it in items:
        if not isinstance(it, dict):
            continue
        bid = str(it.get("id", ""))
        if len(bid) > w_id:
            bid = bid[: w_id - 3] + "..."
        status = str(it.get("status", ""))
        if len(status) > w_status:
            status = status[: w_status - 1] + "…"
        reqs = int(it.get("request_count") or 0)
        tokens = int(it.get("total_tokens") or 0)
        typer.echo(f"{bid:<{w_id}} {status:<{w_status}} {reqs:>{w_reqs}} {tokens:>{w_tok}}")


@batch_app.command("submit")
def batch_submit(
    input_file: str = typer.Option(..., "--input-file"),
    model: Optional[str] = typer.Option(None, "--model", help="Required for content-only JSONL; ignored per line for OpenAI-compatible JSONL."),
    window: str = typer.Option(
        "24h",
        "--window",
        help='Batch SLA window: "24h" is the only supported value.',
        callback=_window_choice,
    ),
    base_url: str = typer.Option("https://api.sference.com"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    _ensure_api_credential()
    client = _client(base_url)
    batch = _call_api(lambda: client.submit_batch(input_file=input_file, model=model, window=window))
    _print(batch.model_dump(), as_json)


@batch_app.command("status")
def batch_status(
    batch_id: str = typer.Option(..., "--batch-id"),
    base_url: str = typer.Option("https://api.sference.com"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    _ensure_api_credential()
    client = _client(base_url)
    batch = _call_api(lambda: client.get_batch(batch_id))
    _print(batch.model_dump(), as_json)


@batch_app.command("stream")
def batch_stream(
    input_file: str = typer.Option(..., "--input-file"),
    model: Optional[str] = typer.Option(None, "--model", help="Required for content-only JSONL; ignored per line for OpenAI-compatible JSONL."),
    window: str = typer.Option(
        "24h",
        "--window",
        help='Batch SLA window: "24h" is the only supported value.',
        callback=_window_choice,
    ),
    poll_interval: float = typer.Option(2.0, "--poll-interval", help="Seconds between status polls while the batch is running."),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Skip resumable cache; always submit a new batch.",
    ),
    base_url: str = typer.Option("https://api.sference.com"),
) -> None:
    """Submit a JSONL batch, wait until it finishes, print JSONL results to stdout (progress on stderr).

    The same input file content can be resumed after Ctrl+C: a cache maps file hash → batch_id
    under ~/.sference/stream_cache.json (override with SFERENCE_STREAM_CACHE).
    """
    _ensure_api_credential()
    path = Path(input_file)
    if not path.is_file():
        typer.echo(f"Not a file: {input_file}", err=True)
        raise typer.Exit(code=1)

    client = _client(base_url)
    key = stream_cache_mod.cache_key_from_file(path)
    bu = base_url.rstrip("/")

    batch: object | None = None
    batch_id: str | None = None
    terminal = ("completed", "failed", "cancelled")

    if not no_cache:
        cached_id = stream_cache_mod.get_cached_batch_id(key, bu)
        if cached_id:
            batch = _get_batch_or_missing(client, cached_id)
            if batch is None:
                stream_cache_mod.remove_cached_batch(key)
            else:
                batch_id, st_cached = _batch_id_and_status(batch)
                if st_cached not in terminal:
                    typer.echo(f"Resuming batch {batch_id}…", err=True)

    if batch_id is None:
        batch = _call_api(lambda: client.submit_batch(input_file=input_file, model=model, window=window))
        batch_id, _st = _batch_id_and_status(batch)
        if not no_cache:
            stream_cache_mod.set_cached_batch(key, batch_id, bu)

    assert batch_id is not None
    assert batch is not None
    start = time.monotonic()
    _, status = _batch_id_and_status(batch)
    while status not in terminal:
        elapsed = int(time.monotonic() - start)
        typer.echo(f"Batch {batch_id} status={status} ({elapsed}s)", err=True)
        time.sleep(poll_interval)
        batch = _call_api(lambda: client.get_batch(batch_id))
        _, status = _batch_id_and_status(batch)

    elapsed = int(time.monotonic() - start)
    typer.echo(f"Batch {batch_id} status={status} ({elapsed}s)", err=True)

    try:
        _call_api(lambda: client.download_results_jsonl(batch_id, sys.stdout.buffer))
    finally:
        if not no_cache:
            stream_cache_mod.remove_cached_batch(key)

    raise typer.Exit(code=0 if status == "completed" else 1)


@batch_app.command("wait")
def batch_wait(
    batch_id: str = typer.Option(..., "--batch-id"),
    poll_interval: float = typer.Option(1.0, "--poll-interval"),
    timeout: float = typer.Option(30.0, "--timeout"),
    base_url: str = typer.Option("https://api.sference.com"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    _ensure_api_credential()
    client = _client(base_url)
    batch = _call_api(
        lambda: client.wait_for_completion(batch_id, poll_interval=poll_interval, timeout=timeout)
    )
    _print(batch.model_dump(), as_json)


@batch_app.command("results")
def batch_results(
    batch_id: str = typer.Option(..., "--batch-id"),
    base_url: str = typer.Option("https://api.sference.com"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    _ensure_api_credential()
    client = _client(base_url)
    results = _call_api(lambda: client.get_results(batch_id))
    _print(results.model_dump(), as_json)


@batch_app.command("cancel")
def batch_cancel(
    batch_id: str = typer.Option(..., "--batch-id"),
    base_url: str = typer.Option("https://api.sference.com"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    _ensure_api_credential()
    client = _client(base_url)
    batch = _call_api(lambda: client.cancel_batch(batch_id))
    _print(batch.model_dump(), as_json)


@stream_app.command("create")
def stream_create(
    name: str = typer.Option(..., "--name"),
    window: str = typer.Option(
        "24h",
        "--window",
        help='Per-item SLA window: "24h" is the only supported value.',
        callback=_window_choice,
    ),
    base_url: str = typer.Option("https://api.sference.com"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    _ensure_api_credential()
    client = _client(base_url)
    stream = _call_api(lambda: client.create_stream(name=name, window=window))
    _print(stream.model_dump(), as_json)


@stream_app.command("list")
def stream_list(
    base_url: str = typer.Option("https://api.sference.com"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    _ensure_api_credential()
    client = _client(base_url)
    payload = _call_api(lambda: client.list_streams().model_dump())
    _print(payload, as_json)


@stream_app.command("status")
def stream_status(
    stream_id: str = typer.Option(..., "--stream-id"),
    base_url: str = typer.Option("https://api.sference.com"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    _ensure_api_credential()
    client = _client(base_url)
    st = _call_api(lambda: client.get_stream(stream_id))
    _print(st.model_dump(), as_json)


@stream_app.command("submit")
def stream_submit(
    stream_id: str = typer.Option(..., "--stream-id"),
    input_file: str = typer.Option(
        ...,
        "--input-file",
        help="JSONL per line: OpenAI-compatible {input: [{role, content}]} or content-only {content} (requires --model).",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Required for content-only JSONL; creates responses using /v1/responses with metadata.stream_id.",
    ),
    base_url: str = typer.Option("https://api.sference.com"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Submit items to a stream using the unified /v1/responses endpoint.

    Each line in the JSONL file creates a response associated with the stream
    via metadata.stream_id. The responses are queued and processed by the
    inference worker.
    """
    _ensure_api_credential()
    path = Path(input_file)
    if not path.is_file():
        typer.echo(f"Not a file: {input_file}", err=True)
        raise typer.Exit(code=1)

    if not model:
        typer.echo("--model is required for stream submit (e.g., Qwen/Qwen3.6-35B-A3B)", err=True)
        raise typer.Exit(code=1)

    client = _client(base_url)

    # Read and parse the JSONL file
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    created_responses = []

    for line in lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            typer.echo(f"Invalid JSON in input file: {e}", err=True)
            raise typer.Exit(code=1)

        # Handle content-only format: {"content": "..."}
        if "content" in data and "input" not in data:
            body = {
                "model": model,
                "input": [{"role": "user", "content": data["content"]}],
                "metadata": {"stream_id": stream_id},
            }
        # Handle OpenAI-compatible format with input
        elif "input" in data:
            body = {
                "model": model,
                "input": data["input"],
                "metadata": {"stream_id": stream_id, **data.get("metadata", {})},
            }
        else:
            typer.echo(f"Unrecognized line format: {line[:100]}", err=True)
            raise typer.Exit(code=1)

        # SDK signature is keyword-only; pass the payload as kwargs.
        resp = _call_api(lambda: client.create_response(**body))
        created_responses.append(resp)

    # Return stream detail-like response for compatibility
    result = {
        "stream_id": stream_id,
        "total_items": len(created_responses),
        "items": [r.model_dump() for r in created_responses],
    }
    _print(result, as_json)


@stream_app.command("archive")
def stream_archive(
    stream_id: str = typer.Option(..., "--stream-id"),
    base_url: str = typer.Option("https://api.sference.com"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    _ensure_api_credential()
    client = _client(base_url)
    st = _call_api(lambda: client.archive_stream(stream_id))
    _print(st.model_dump(), as_json)


@stream_app.command("cancel")
def stream_cancel(
    stream_id: str = typer.Option(..., "--stream-id"),
    base_url: str = typer.Option("https://api.sference.com"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Stop accepting new items and stop enqueueing pending work; in-flight items are not auto-cancelled."""
    _ensure_api_credential()
    client = _client(base_url)
    st = _call_api(lambda: client.cancel_stream(stream_id))
    _print(st.model_dump(), as_json)


@batch_app.command("download-results")
def batch_download_results(
    batch_id: str = typer.Option(..., "--batch-id"),
    out: Path = typer.Option(..., "--out", help="Output path for downloaded results."),
    format: str = typer.Option("jsonl", "--format", help="Download format (only jsonl is supported)."),
    base_url: str = typer.Option("https://api.sference.com"),
) -> None:
    _ensure_api_credential()
    if format.lower() != "jsonl":
        typer.echo("Only --format jsonl is supported.", err=True)
        raise typer.Exit(code=1)
    client = _client(base_url)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        _call_api(lambda: client.download_results_jsonl(batch_id, f))
    typer.echo(str(out))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
