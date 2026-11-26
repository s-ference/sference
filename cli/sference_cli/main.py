from __future__ import annotations

import json
import logging
import os
import platform
import sys
import time
import warnings
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn, Optional, TypeVar

import typer

import sference_sdk
from sference_sdk import COMPLETION_WINDOWS, ApiError, SferenceClient

from sference_cli import __version__ as _CLI_VERSION
from sference_sdk.checkpoint import clear_checkpoint, load_checkpoint, save_checkpoint

from . import stream_cache as stream_cache_mod
from .device_auth import (
    CLIENT_ID_SFERENCE_CLI,
    EXPIRY_SKEW_SECONDS,
    DeviceAuthError,
    poll_for_tokens,
    refresh_tokens,
    start_device_login,
)
from ._discovery import is_retired
from .launch import register_launch_commands
from .settings import (
    DEFAULT_API_BASE_URL,
    DEFAULT_MODEL,
    resolve_api_base_url,
    resolve_model,
)

_T = TypeVar("_T")

_logger = logging.getLogger(__name__)


app = typer.Typer(help="sference CLI", invoke_without_command=True)
auth_app = typer.Typer(
    help="Sign in and inspect the current credential.", invoke_without_command=True
)
batch_app = typer.Typer(
    help="Batches: submit a JSONL file, wait, fetch results.",
    invoke_without_command=True,
)
stream_app = typer.Typer(
    help="Streams: long-lived queues of responses.", invoke_without_command=True
)
responses_app = typer.Typer(
    help="Single responses (POST /v1/responses).", invoke_without_command=True
)
models_app = typer.Typer(
    help="List models (GET /v1/models).", invoke_without_command=True
)
embeddings_app = typer.Typer(
    help="Embeddings (POST /v1/embeddings).", invoke_without_command=True
)
decisions_app = typer.Typer(
    help="Realtime decisions (POST /v1/decisions).", invoke_without_command=True
)
app.add_typer(auth_app, name="auth")
app.add_typer(batch_app, name="batch")
app.add_typer(stream_app, name="stream")
app.add_typer(responses_app, name="responses")
app.add_typer(models_app, name="models")
app.add_typer(embeddings_app, name="embeddings")
app.add_typer(decisions_app, name="decisions")
register_launch_commands(app)

CREDENTIALS_PATH = Path.home() / ".sference" / "credentials.json"

# Shared option definitions so every command documents them the same way.
BASE_URL_OPTION = typer.Option(
    None,
    "--base-url",
    envvar="SFERENCE_BASE_URL",
    help=f"Sference API base URL (default: {DEFAULT_API_BASE_URL}).",
)
JSON_OPTION = typer.Option(False, "--json", help="Print the raw API response as JSON.")
MODEL_OPTION = typer.Option(
    None,
    "--model",
    "-m",
    help=f"Catalog model id (default: SFERENCE_MODEL or {DEFAULT_MODEL}).",
)
BATCH_MODEL_OPTION = typer.Option(
    None,
    "--model",
    "-m",
    help=(
        "Model for content-only JSONL lines (default: SFERENCE_MODEL or "
        f"{DEFAULT_MODEL}); envelope lines use their own body.model."
    ),
)


def _write_credentials_payload(payload: dict) -> None:
    # The credential file holds an API key or OAuth tokens; keep it owner-only
    # (0700 dir / 0600 file) so other local users cannot read it. chmod is a
    # no-op on Windows.
    CREDENTIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(CREDENTIALS_PATH.parent, 0o700)
    except OSError:
        pass
    CREDENTIALS_PATH.write_text(json.dumps(payload), encoding="utf-8")
    try:
        os.chmod(CREDENTIALS_PATH, 0o600)
    except OSError:
        pass


def _write_token(token: str) -> None:
    _write_credentials_payload({"token": token})


def _write_device_credentials(tokens: dict) -> None:
    """Persist a device-flow token response (v2 credentials).

    Stores an absolute ``expires_at`` (not the wire ``expires_in``) so readers
    never have to know when the file was written. The refresh token rotates on
    every refresh — the file must always hold the newest pair.
    """
    _write_credentials_payload(
        {
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "expires_at": time.time() + float(tokens["expires_in"]),
        }
    )


def _read_credentials_file() -> dict | None:
    if not CREDENTIALS_PATH.exists():
        return None
    try:
        payload = json.loads(CREDENTIALS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _device_flow_base_url() -> str:
    return resolve_api_base_url()


def _read_device_credentials() -> dict | None:
    """v2 credentials ``{access_token, refresh_token, expires_at}``, refreshing first.

    Returns None when the file holds no usable device grant (legacy ``{"token"}``
    shape, malformed, or the refresh failed — e.g. the grant was revoked, which
    is also how reuse detection surfaces here). A successful refresh rewrites
    the file with the rotated pair before returning.
    """
    payload = _read_credentials_file()
    if payload is None:
        return None
    access_token = payload.get("access_token")
    refresh_token = payload.get("refresh_token")
    expires_at = payload.get("expires_at")
    if not isinstance(access_token, str) or not isinstance(refresh_token, str):
        return None
    if not isinstance(expires_at, (int, float)):
        return None
    if time.time() < expires_at - EXPIRY_SKEW_SECONDS:
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_at": expires_at,
        }
    try:
        tokens = refresh_tokens(
            _device_flow_base_url(), CLIENT_ID_SFERENCE_CLI, refresh_token
        )
    except DeviceAuthError as exc:
        typer.echo(
            f"Device login could not be refreshed ({exc.description}).\n"
            "Run `sference auth login` to sign in again.",
            err=True,
        )
        return None
    _write_device_credentials(tokens)
    return _read_credentials_file()


_warned: set[str] = set()


def _warn_once(message: str) -> None:
    """Credential resolution runs several times per command; say it once."""
    if message not in _warned:
        _warned.add(message)
        typer.echo(message, err=True)


def _has_device_grant() -> bool:
    """Whether the file holds a device pair — checked without refreshing it."""
    payload = _read_credentials_file()
    if payload is None:
        return False
    return isinstance(payload.get("access_token"), str) and isinstance(
        payload.get("refresh_token"), str
    )


def _env_api_key() -> str | None:
    env = os.environ.get("SFERENCE_API_KEY")
    return env.strip() if env and env.strip() else None


# Set once a refresh has proved the grant is gone, so the rest of the command
# stops presenting a credential the server has already rejected.
_device_grant_dead = False


def _recover_from_unauthorized() -> str | None:
    """SDK 401 hook: the access token was rejected — try to replace it.

    The cached token is used until it nears expiry, so a grant revoked (or
    reuse-detected) server-side is invisible locally until something 401s. That
    is the moment to refresh: either the token had simply gone stale and the
    grant is fine, or the refresh fails and the grant really is dead — in which
    case an API key, if the caller has one, is the credential to continue on.
    """
    global _device_grant_dead
    if _device_grant_dead or not _has_device_grant():
        return None
    payload = _read_credentials_file() or {}
    refresh_token = payload.get("refresh_token")
    if not isinstance(refresh_token, str):
        return None
    try:
        tokens = refresh_tokens(
            _device_flow_base_url(), CLIENT_ID_SFERENCE_CLI, refresh_token
        )
    except DeviceAuthError as exc:
        _device_grant_dead = True
        _warn_once(
            f"This device is no longer signed in ({exc.description}).\n"
            "Run `sference auth login` to sign in again."
        )
        env_key = _env_api_key()
        if env_key is not None:
            _warn_once("Continuing with SFERENCE_API_KEY.")
        return env_key
    _write_device_credentials(tokens)
    return str(tokens["access_token"])


def _read_token() -> str | None:
    """Resolve the credential to send, device login first.

    `sference auth login` is the CLI's own way in, so a signed-in device beats
    ``SFERENCE_API_KEY``: an sk_ key is meant for non-interactive product
    integrations (litellm, CI) that never run the device flow. The env var used
    to win outright, which meant a key left over in a shell silently swallowed a
    login the user had just completed — including the validation step right
    after it, which then reported a bare 401.
    """
    env_key = _env_api_key()

    if _has_device_grant() and not _device_grant_dead:
        if env_key is not None:
            _warn_once(
                "Note: SFERENCE_API_KEY is set, but this device is signed in — using the "
                "device login. Unset SFERENCE_API_KEY to use the API key instead."
            )
        creds = _read_device_credentials()
        if creds is not None:
            return creds["access_token"]
        # The grant is gone (revoked, expired, reuse-detected). _read_device_credentials
        # has already said so; fall through so a usable key still works.
    if env_key is not None:
        return env_key
    payload = _read_credentials_file()
    if payload is None:
        return None
    tok = payload.get("token")  # legacy API-key shape (CI / --api-key)
    return tok.strip() if isinstance(tok, str) and tok.strip() else None


# The client the running command built. One invocation runs one command and
# builds one client, so this is unambiguous — it is what _call_api re-credentials
# when a device token turns out to be stale.
_active_client: SferenceClient | None = None


def _client(
    base_url: Optional[str] = None,
    *,
    timeout: float | None = None,
    token: Optional[str] = None,
) -> SferenceClient:
    """Build a client. ``token`` pins the credential instead of resolving one —
    used right after a login, which must validate what it just minted rather
    than whatever ``_read_token`` would pick."""
    global _active_client
    _active_client = SferenceClient(
        base_url=resolve_api_base_url(base_url),
        api_key=token or _read_token(),
        timeout=timeout,
    )
    return _active_client


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


def _fail(message: str) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(code=1)


def _exit_on_api_error(err: str) -> NoReturn:
    if err.startswith("401:"):
        # A dead grant has already been explained by the recovery step, in
        # more specific terms than this generic hint — don't say it twice.
        if not _device_grant_dead:
            typer.echo(
                "Unauthorized (401). Run `sference auth login` to sign in again.\n"
                "CI/non-interactive: sference auth login --api-key 'sk_...' "
                "(create a key in Console → API Keys while signed in).\n"
                "SFERENCE_API_KEY is used only when this device is not signed in.",
                err=True,
            )
        raise typer.Exit(code=1)
    typer.echo(err, err=True)
    if err.startswith(("504:", "408:")):
        typer.echo(
            "The synchronous wait timed out and the request was cancelled. "
            "For long-running requests, resubmit with --background and fetch "
            "the result with `sference responses result --id <id>`.",
            err=True,
        )
    raise typer.Exit(code=1)


def _call_api(fn: Callable[[], _T]) -> _T:
    try:
        return fn()
    except ApiError as exc:
        err = str(exc)
    if err.startswith("401:"):
        # The cached access token is used until it nears expiry, so a grant
        # refreshed, revoked or reuse-detected server-side surfaces only
        # here. Retrying is safe because a 401 means the handler never ran,
        # so nothing can be repeated — this is not a general-purpose retry.
        replacement = _recover_from_unauthorized()
        if replacement and _active_client is not None:
            _active_client.set_api_key(replacement)
            try:
                return fn()
            except ApiError as retry_exc:
                err = str(retry_exc)
    _exit_on_api_error(err)


def _batch_id_and_status(obj: object) -> tuple[str, str]:
    if hasattr(obj, "model_dump"):
        d = obj.model_dump()
        return str(d["id"]), str(d["status"])
    bid = getattr(obj, "id", None)
    st = getattr(obj, "status", None)
    return str(bid), str(st)


def _get_batch_or_missing(client: SferenceClient, batch_id: str) -> Any | None:
    """The batch, or None when the server no longer has it (404)."""

    def fetch() -> Any | None:
        try:
            return client.get_batch(batch_id)
        except ApiError as exc:
            if str(exc).startswith("404:"):
                return None
            raise

    return _call_api(fetch)


def _submit_batch(
    client: SferenceClient, *, input_file: str, model: Optional[str], window: str
) -> Any:
    """Submit a JSONL file, turning local parse errors into a clean exit."""
    with warnings.catch_warnings():
        # The SDK warns that --model is ignored for envelope lines; that is
        # expected when the model came from the default rather than the user.
        if model is None:
            warnings.simplefilter("ignore")
        try:
            return _call_api(
                lambda: client.submit_batch(
                    input_file=input_file, model=resolve_model(model), window=window
                )
            )
        except (ValueError, OSError) as exc:
            _fail(f"Could not read {input_file}: {exc}")


def _plain(value: object) -> object:
    """API payload as plain JSON data; typed SDK models dump only the fields the server sent."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_unset=True)
    return value


def _print(value: object, as_json: bool) -> None:
    """``--json`` prints the payload verbatim; otherwise one ``key: value`` line per field."""
    if as_json or not isinstance(value, dict):
        typer.echo(json.dumps(value, indent=2, default=str))
        return
    width = max((len(str(k)) for k in value), default=0)
    for key, val in value.items():
        rendered = (
            val
            if isinstance(val, (str, int, float)) or val is None
            else json.dumps(val, default=str)
        )
        typer.echo(f"{str(key).ljust(width)}  {'—' if rendered is None else rendered}")


def _list_models(*, client: SferenceClient, as_json: bool) -> None:
    payload = _plain(_call_api(client.list_models))
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
    labels = {"text_generation": "text", "text_embedding": "embedding"}
    if not isinstance(raw, str) or not raw:
        return "—"
    return labels.get(raw, raw)


def _format_tokens(n: object) -> str:
    if not isinstance(n, int) or n <= 0:
        return "—"
    if n % (1024 * 1024) == 0:
        return f"{n // (1024 * 1024)}M"
    if n % 1024 == 0:
        return f"{n // 1024}k"
    return str(n)


def _model_note(row: dict) -> str:
    """Alias target and retirement, the two things that make an id worth migrating off."""
    parts = []
    if row.get("alias_of"):
        parts.append(f"alias of {row['alias_of']}")
    if row.get("retires_on"):
        verb = "retired" if is_retired(row) else "retires"
        parts.append(f"{verb} {row['retires_on']}")
    return "; ".join(parts)


def _print_models_table(payload: object) -> None:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        typer.echo(json.dumps(payload, indent=2, default=str))
        return
    rows = [r for r in payload["data"] if isinstance(r, dict)]
    if not rows:
        typer.echo("No models.")
        return

    headers = (
        "id",
        "name",
        "modality",
        "context",
        "vision",
        "thinking",
        "tools",
        "note",
    )
    table = []
    for row in rows:
        ctx = row.get("context_tokens")
        caps = row.get("capabilities")
        table.append(
            (
                str(row.get("id", "")),
                str(row.get("display_name") or ""),
                _modality_label(row.get("modality")),
                _format_tokens(ctx),
                _capability_flag(caps, "image_input"),
                _capability_flag(caps, "thinking"),
                _capability_flag(caps, "tools"),
                _model_note(row),
            )
        )
    widths = [max(len(h), *(len(r[i]) for r in table)) for i, h in enumerate(headers)]
    for line in (headers, *table):
        typer.echo("  ".join(cell.ljust(w) for cell, w in zip(line, widths)).rstrip())


def _window_choice(value: str) -> str:
    if value not in COMPLETION_WINDOWS:
        allowed = ", ".join(f'"{w}"' for w in COMPLETION_WINDOWS)
        raise typer.BadParameter(
            f"Window must be one of {allowed}.", param_hint="--window"
        )
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
        typer.echo(
            f"sference-cli {_CLI_VERSION} (sference-sdk {sference_sdk.__version__})"
        )
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
    as_json: bool = JSON_OPTION,
    base_url: Optional[str] = BASE_URL_OPTION,
) -> None:
    """List models (GET /v1/models)."""
    if ctx.invoked_subcommand is not None:
        return
    _ensure_api_credential()
    _list_models(client=_client(base_url), as_json=as_json)


@models_app.command("list")
def models_list(
    as_json: bool = JSON_OPTION,
    base_url: Optional[str] = BASE_URL_OPTION,
) -> None:
    """List models (GET /v1/models)."""
    _ensure_api_credential()
    _list_models(client=_client(base_url), as_json=as_json)


@embeddings_app.command("create")
def embeddings_create(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Embedding model id (modality=embedding in `sference models`).",
    ),
    input_text: list[str] = typer.Option(
        ..., "--input", help="Text to embed (repeat for multiple strings)."
    ),
    encoding_format: str = typer.Option(
        "float", "--encoding-format", help='"float" or "base64".'
    ),
    dimensions: Optional[int] = typer.Option(None, "--dimensions"),
    timeout: float = typer.Option(
        600.0, "--timeout", help="HTTP read timeout in seconds."
    ),
    base_url: Optional[str] = BASE_URL_OPTION,
    _as_json: bool = typer.Option(
        False, "--json", hidden=True, help="No-op: output is always JSON."
    ),
) -> None:
    """Create embeddings (POST /v1/embeddings). Prints the response as JSON."""
    _ensure_api_credential()
    if encoding_format not in ("float", "base64"):
        _fail('--encoding-format must be "float" or "base64".')
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
    _print(resp.model_dump(), True)


def _read_json_option(raw: str, option: str) -> object:
    """Parse a JSON option value; ``@path`` reads the file and ``@-`` reads stdin."""
    if raw.startswith("@"):
        source = raw[1:]
        try:
            raw = (
                sys.stdin.read()
                if source == "-"
                else Path(source).read_text(encoding="utf-8")
            )
        except OSError as exc:
            _fail(f"{option}: could not read {source}: {exc}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        _fail(f"{option} is not valid JSON: {exc}")


def _format_answer(answer: dict) -> str:
    kind = answer.get("type")
    if kind == "choice":
        return f"{answer['choice']} (confidence {answer['confidence']:.2f})"
    if kind == "score":
        return f"{answer['score']:.2f} (confidence {answer['confidence']:.2f})"
    if kind == "noul":
        return f"p(true)={answer['noul']:.2f}"
    return json.dumps(answer)


@decisions_app.command("create")
def decisions_create(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Decisions model id (modality=decisions in `sference models`).",
    ),
    questions: str = typer.Option(
        ...,
        "--questions",
        help='JSON object {name: {"type": "choice"|"score"|"noul", ...}}; @path reads a file, @- stdin.',
    ),
    state: Optional[str] = typer.Option(None, "--state", help="Text to decide about."),
    state_json: Optional[str] = typer.Option(
        None,
        "--state-json",
        help="State as a JSON value instead of text; @path reads a file, @- stdin.",
    ),
    # Unset leaves the SDK's per-call read timeout, which outlasts the server's own deadline.
    timeout: Optional[float] = typer.Option(
        None,
        "--timeout",
        help="HTTP read timeout in seconds (default: as long as the server may take).",
    ),
    base_url: Optional[str] = BASE_URL_OPTION,
    as_json: bool = JSON_OPTION,
) -> None:
    """Answer up to 64 questions about one state in a single realtime call (POST /v1/decisions)."""
    _ensure_api_credential()
    if (state is None) == (state_json is None):
        _fail("Pass exactly one of --state or --state-json.")
    state_value = (
        state if state is not None else _read_json_option(state_json, "--state-json")
    )
    parsed_questions = _read_json_option(questions, "--questions")
    if not isinstance(parsed_questions, dict):
        _fail("--questions must be a JSON object keyed by question name.")
    client = _client(base_url, timeout=timeout)
    try:
        resp = _call_api(
            lambda: client.create_decision(
                model=model, state=state_value, questions=parsed_questions
            )
        )
    except ValueError as exc:
        _fail(f"Invalid --questions: {exc}")
    payload = resp.model_dump()
    if as_json:
        _print(payload, True)
        return
    width = max(len(name) for name in payload["answers"])
    for name, answer in payload["answers"].items():
        typer.echo(f"{name.ljust(width)}  {_format_answer(answer)}")
    typer.echo(f"input_tokens: {payload['usage']['input_tokens']}")


def _validate_login(base_url: Optional[str], token: str, as_json: bool) -> None:
    """Call GET /v1/auth/me with the credential just saved and say who it is."""
    me = _plain(_call_api(_client(base_url, token=token).get_me))
    if as_json:
        _print(me, True)
    elif isinstance(me, dict) and "username" in me and "role" in me:
        typer.echo(f"Authenticated as {me['username']} ({me['role']}).")
    else:
        typer.echo(f"Authenticated. {me}")


@auth_app.command("login")
def auth_login(
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="API key (sk_...) or JWT. Non-interactive: saves immediately (e.g. CI).",
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Print authenticated user as JSON after validation."
    ),
    validate: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Validate the credential by calling GET /v1/auth/me and print the authenticated user.",
    ),
    base_url: Optional[str] = BASE_URL_OPTION,
    console_url: Optional[str] = typer.Option(
        None,
        "--console-url",
        envvar="SFERENCE_CONSOLE_URL",
        help="Override the console URL the browser opens (default: the server-provided verification URI).",
    ),
    no_browser: bool = typer.Option(
        False,
        "--no-browser",
        help="Do not open a browser; print the URL and code only.",
    ),
) -> None:
    """Authenticate this device: approve a code in the console (RFC 8628 device flow).

    Interactive users never handle an API key: the CLI holds a 24 h access
    token plus a 30 d rotating refresh token in ~/.sference/credentials.json and
    refreshes it automatically. ``--api-key`` is the non-interactive path (CI).
    """
    if api_key is not None:
        key = api_key.strip()
        if not key:
            _fail("Empty --api-key.")
        _write_token(key)
        typer.echo(f"Credentials saved to {CREDENTIALS_PATH}")
        if validate:
            _validate_login(base_url, key, as_json)
        return

    api_base = resolve_api_base_url(base_url)
    try:
        device_label = platform.node() or None
    except Exception:
        device_label = None
    try:
        start = start_device_login(api_base, CLIENT_ID_SFERENCE_CLI, device_label)
    except DeviceAuthError as exc:
        _fail(f"Could not start device login ({exc.description}).")

    user_code = str(start["user_code"])
    verification_uri = str(start["verification_uri"])
    if console_url:
        # Local dev / port-forwards: point the browser at the overridden console
        # rather than the server-configured one.
        verification_uri = f"{console_url.rstrip('/')}/device"
    approve_url = f"{verification_uri}?code={user_code}"

    if not no_browser:
        typer.echo(f"Opening {approve_url} in your browser...")
        try:
            webbrowser.open(approve_url)
        except Exception:
            typer.echo("Could not open the browser automatically.")
    typer.echo("")
    typer.echo(f"Approve this device at: {approve_url}")
    typer.echo(f"Code: {user_code}")
    typer.echo("Waiting for approval...")

    try:
        tokens = poll_for_tokens(
            api_base,
            CLIENT_ID_SFERENCE_CLI,
            str(start["device_code"]),
            interval=float(start["interval"]),
            expires_in=float(start["expires_in"]),
        )
    except DeviceAuthError as exc:
        _fail(f"Device login failed ({exc.description}).")

    _write_device_credentials(tokens)
    typer.echo(f"Credentials saved to {CREDENTIALS_PATH}")
    if validate:
        _validate_login(api_base, str(tokens["access_token"]), as_json)


@auth_app.command("me")
def auth_me(
    base_url: Optional[str] = BASE_URL_OPTION,
    as_json: bool = JSON_OPTION,
) -> None:
    """Show the user the current credential belongs to (GET /v1/auth/me)."""
    _ensure_api_credential()
    _print(_plain(_call_api(_client(base_url).get_me)), as_json)


@responses_app.command("create")
def responses_create(
    content: str = typer.Option(..., "--content", help="User message text."),
    model: Optional[str] = MODEL_OPTION,
    instructions: Optional[str] = typer.Option(
        None, "--instructions", help="System prompt."
    ),
    max_output_tokens: Optional[int] = typer.Option(None, "--max-output-tokens"),
    include_reasoning: bool = typer.Option(
        True,
        "--include-reasoning/--no-include-reasoning",
        help="Include extracted reasoning in the response output.",
    ),
    enable_thinking: Optional[bool] = typer.Option(
        None,
        "--enable-thinking/--disable-thinking",
        help="Turn thinking on/off for thinking-capable models (omit for the model default). "
        "GLM-5.3 models always think and reject --disable-thinking.",
    ),
    background: bool = typer.Option(
        False,
        "--background/--no-background",
        help=(
            "Submit asynchronously and return immediately; fetch later with "
            "`sference responses result`. Default blocks until the response is "
            "terminal; if the server-side sync wait times out the request is "
            "cancelled and the command fails — use --background for long requests."
        ),
    ),
    # Unset leaves the SDK's per-call read timeout, which outlasts the server's own deadline.
    timeout: Optional[float] = typer.Option(
        None,
        "--timeout",
        help="HTTP read timeout in seconds (default: as long as the server may take).",
    ),
    base_url: Optional[str] = BASE_URL_OPTION,
) -> None:
    """Create one response (POST /v1/responses) and print it as JSON."""
    _ensure_api_credential()
    client = _client(base_url, timeout=timeout)
    resp = _call_api(
        lambda: client.create_response(
            model=resolve_model(model),
            input=[{"role": "user", "content": content}],
            instructions=instructions,
            max_output_tokens=max_output_tokens,
            include_reasoning=include_reasoning,
            enable_thinking=enable_thinking,
            background=background,
        )
    )
    _print(resp.model_dump(), True)


_TERMINAL_RESPONSE_STATUSES = ("completed", "failed", "cancelled", "incomplete")


@responses_app.command("result")
def responses_result(
    id: str = typer.Option(..., "--id", help="Response id from `responses create`."),
    poll_interval: float = typer.Option(
        0.5,
        "--poll-interval",
        help="Seconds between status polls while waiting for completion.",
    ),
    base_url: Optional[str] = BASE_URL_OPTION,
) -> None:
    """Wait for a response to reach a terminal status and print the final JSON. Ctrl-C stops waiting."""
    _ensure_api_credential()
    client = _client(base_url)
    try:
        while True:
            d = _call_api(lambda: client.get_response(id)).model_dump()
            if d.get("status") in _TERMINAL_RESPONSE_STATUSES:
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
    consumer: str = typer.Option(
        "default", "--consumer", help="Checkpoint namespace for resume."
    ),
    from_latest: bool = typer.Option(
        False,
        "--from-latest",
        help="Ignore saved checkpoint and start from the latest events page.",
    ),
    no_checkpoint: bool = typer.Option(
        False, "--no-checkpoint", help="Do not read or write checkpoints."
    ),
    poll_interval: float = typer.Option(
        5.0,
        "--poll-interval",
        help="Seconds to long-poll while catching up (capped at 30s server-side).",
    ),
    base_url: Optional[str] = BASE_URL_OPTION,
    _as_json: bool = typer.Option(
        False, "--json", hidden=True, help="No-op: output is always JSONL."
    ),
) -> None:
    """Print completion events as JSONL (GET /v1/responses/events). Ctrl-C to stop.

    Omit --stream-id for standalone/batch completions; pass --stream-id to scope to a stream.
    """
    _ensure_api_credential()
    client = _client(base_url)
    bu = resolve_api_base_url(base_url)
    ck = stream_id if stream_id else "__responses_events__"

    if from_latest and not no_checkpoint:
        clear_checkpoint(bu, ck, consumer)

    last: str | None = None
    if not no_checkpoint and not from_latest:
        last = load_checkpoint(bu, ck, consumer)

    wait_ms = int(min(poll_interval, 30.0) * 1000)
    try:
        while True:
            page = _call_api(
                lambda: client.list_responses_events(
                    stream_id=stream_id, limit=50, starting_after=last, wait_ms=wait_ms
                )
            )
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
    base_url: Optional[str] = BASE_URL_OPTION,
    as_json: bool = JSON_OPTION,
) -> None:
    """List batches."""
    _ensure_api_credential()
    client = _client(base_url)
    payload = _call_api(lambda: client.list_batches().model_dump())
    items = payload.get("items") if isinstance(payload, dict) else None
    if as_json or not isinstance(items, list):
        _print(payload, True)
        return
    if not items:
        typer.echo("No batches.")
        return
    w_id = max(len("id"), *(len(str(it.get("id", ""))) for it in items))
    typer.echo(f"{'id':<{w_id}}  {'status':<12} {'reqs':>6} {'tokens':>10}")
    for it in items:
        typer.echo(
            f"{it.get('id', '')!s:<{w_id}}  {it.get('status', '')!s:<12} "
            f"{int(it.get('request_count') or 0):>6} {int(it.get('total_tokens') or 0):>10}"
        )


_WINDOW_OPTION = typer.Option(
    "24h",
    "--window",
    help='Completion window: "24h" is the only supported value.',
    callback=_window_choice,
)


@batch_app.command("submit")
def batch_submit(
    input_file: str = typer.Option(
        ..., "--input-file", help="JSONL file (see README: JSONL input formats)."
    ),
    model: Optional[str] = BATCH_MODEL_OPTION,
    window: str = _WINDOW_OPTION,
    base_url: Optional[str] = BASE_URL_OPTION,
    as_json: bool = JSON_OPTION,
) -> None:
    """Submit a JSONL file as one batch (POST /v1/batches)."""
    _ensure_api_credential()
    client = _client(base_url)
    batch = _submit_batch(client, input_file=input_file, model=model, window=window)
    _print(batch.model_dump(), as_json)


@batch_app.command("status")
def batch_status(
    batch_id: str = typer.Option(..., "--batch-id"),
    base_url: Optional[str] = BASE_URL_OPTION,
    as_json: bool = JSON_OPTION,
) -> None:
    """Show one batch."""
    _ensure_api_credential()
    client = _client(base_url)
    _print(_call_api(lambda: client.get_batch(batch_id)).model_dump(), as_json)


_TERMINAL_BATCH_STATUSES = ("completed", "failed", "cancelled")


@batch_app.command("stream")
def batch_stream(
    input_file: str = typer.Option(
        ..., "--input-file", help="JSONL file (see README: JSONL input formats)."
    ),
    model: Optional[str] = BATCH_MODEL_OPTION,
    window: str = _WINDOW_OPTION,
    poll_interval: float = typer.Option(
        2.0,
        "--poll-interval",
        help="Seconds between status polls while the batch is running.",
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Skip resumable cache; always submit a new batch."
    ),
    base_url: Optional[str] = BASE_URL_OPTION,
) -> None:
    """Submit a JSONL batch, wait until it finishes, print JSONL results to stdout (progress on stderr).

    The same input file content can be resumed after Ctrl+C: a cache maps file hash → batch_id
    under ~/.sference/stream_cache.json (override with SFERENCE_STREAM_CACHE).
    """
    _ensure_api_credential()
    path = Path(input_file)
    if not path.is_file():
        _fail(f"Not a file: {input_file}")

    client = _client(base_url)
    key = stream_cache_mod.cache_key_from_file(path)
    bu = resolve_api_base_url(base_url)

    batch: Any = None
    if not no_cache:
        cached_id = stream_cache_mod.get_cached_batch_id(key, bu)
        if cached_id:
            batch = _get_batch_or_missing(client, cached_id)
            if batch is None:
                stream_cache_mod.remove_cached_batch(key)
            else:
                cached_batch_id, cached_status = _batch_id_and_status(batch)
                if cached_status not in _TERMINAL_BATCH_STATUSES:
                    typer.echo(f"Resuming batch {cached_batch_id}…", err=True)

    if batch is None:
        batch = _submit_batch(client, input_file=input_file, model=model, window=window)
        if not no_cache:
            stream_cache_mod.set_cached_batch(key, _batch_id_and_status(batch)[0], bu)

    batch_id, status = _batch_id_and_status(batch)
    start = time.monotonic()
    while status not in _TERMINAL_BATCH_STATUSES:
        typer.echo(
            f"Batch {batch_id} status={status} ({int(time.monotonic() - start)}s)",
            err=True,
        )
        time.sleep(poll_interval)
        _, status = _batch_id_and_status(_call_api(lambda: client.get_batch(batch_id)))
    typer.echo(
        f"Batch {batch_id} status={status} ({int(time.monotonic() - start)}s)", err=True
    )

    try:
        _call_api(lambda: client.download_results_jsonl(batch_id, sys.stdout.buffer))
    finally:
        if not no_cache:
            stream_cache_mod.remove_cached_batch(key)

    raise typer.Exit(code=0 if status == "completed" else 1)


@batch_app.command("wait")
def batch_wait(
    batch_id: str = typer.Option(..., "--batch-id"),
    poll_interval: float = typer.Option(
        1.0, "--poll-interval", help="Seconds between status polls."
    ),
    timeout: float = typer.Option(
        30.0, "--timeout", help="Give up after this many seconds."
    ),
    base_url: Optional[str] = BASE_URL_OPTION,
    as_json: bool = JSON_OPTION,
) -> None:
    """Poll a batch until it reaches a terminal status (or --timeout)."""
    _ensure_api_credential()
    client = _client(base_url)
    try:
        batch = _call_api(
            lambda: client.wait_for_completion(
                batch_id, poll_interval=poll_interval, timeout=timeout
            )
        )
    except TimeoutError as exc:
        _fail(
            f"{exc}. Raise --timeout, or use `sference batch stream`, which waits without a limit."
        )
    _print(batch.model_dump(), as_json)


@batch_app.command("results")
def batch_results(
    batch_id: str = typer.Option(..., "--batch-id"),
    base_url: Optional[str] = BASE_URL_OPTION,
    as_json: bool = JSON_OPTION,
) -> None:
    """Show a finished batch's results (GET /v1/batches/{id}/results)."""
    _ensure_api_credential()
    client = _client(base_url)
    _print(_call_api(lambda: client.get_results(batch_id)).model_dump(), as_json)


@batch_app.command("cancel")
def batch_cancel(
    batch_id: str = typer.Option(..., "--batch-id"),
    base_url: Optional[str] = BASE_URL_OPTION,
    as_json: bool = JSON_OPTION,
) -> None:
    """Cancel a batch."""
    _ensure_api_credential()
    client = _client(base_url)
    _print(_call_api(lambda: client.cancel_batch(batch_id)).model_dump(), as_json)


@batch_app.command("download-results")
def batch_download_results(
    batch_id: str = typer.Option(..., "--batch-id"),
    out: Path = typer.Option(..., "--out", help="Output path for downloaded results."),
    format: str = typer.Option(
        "jsonl", "--format", help="Download format (only jsonl is supported)."
    ),
    base_url: Optional[str] = BASE_URL_OPTION,
) -> None:
    """Download a batch's results as JSONL to a file."""
    _ensure_api_credential()
    if format.lower() != "jsonl":
        _fail("Only --format jsonl is supported.")
    client = _client(base_url)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        _call_api(lambda: client.download_results_jsonl(batch_id, f))
    typer.echo(str(out))


@stream_app.command("create")
def stream_create(
    name: str = typer.Option(..., "--name"),
    window: str = _WINDOW_OPTION,
    base_url: Optional[str] = BASE_URL_OPTION,
    as_json: bool = JSON_OPTION,
) -> None:
    """Create a stream."""
    _ensure_api_credential()
    client = _client(base_url)
    _print(
        _call_api(lambda: client.create_stream(name=name, window=window)).model_dump(),
        as_json,
    )


@stream_app.command("list")
def stream_list(
    base_url: Optional[str] = BASE_URL_OPTION,
    as_json: bool = JSON_OPTION,
) -> None:
    """List streams."""
    _ensure_api_credential()
    client = _client(base_url)
    _print(_call_api(lambda: client.list_streams().model_dump()), as_json)


@stream_app.command("status")
def stream_status(
    stream_id: str = typer.Option(..., "--stream-id"),
    base_url: Optional[str] = BASE_URL_OPTION,
    as_json: bool = JSON_OPTION,
) -> None:
    """Show one stream with its counters."""
    _ensure_api_credential()
    client = _client(base_url)
    _print(_call_api(lambda: client.get_stream(stream_id)).model_dump(), as_json)


# Request fields `stream submit` forwards from a JSONL line to create_response.
# Chat-shaped bodies are accepted too: `messages` becomes `input` and
# `max_tokens` becomes `max_output_tokens`.
_STREAM_LINE_FIELDS = (
    "instructions",
    "max_output_tokens",
    "temperature",
    "top_p",
    "include_reasoning",
    "enable_thinking",
)


def _stream_line_request(
    line_no: int, data: object, *, model: Optional[str], stream_id: str
) -> dict[str, Any]:
    """Map one ``stream submit`` JSONL line to ``create_response`` kwargs.

    Accepts the same shapes as batch JSONL — content-only ``{"content"}``, a
    batch envelope ``{"custom_id"?, "body": {...}}`` — plus a bare request
    body ``{"input", ...}``.
    """
    if not isinstance(data, dict):
        _fail(f"Line {line_no}: expected a JSON object.")
    custom_id = data.get("custom_id")
    if isinstance(data.get("body"), dict):
        body = dict(data["body"])
    elif "content" in data and "input" not in data:
        body = {"input": [{"role": "user", "content": data["content"]}]}
    elif "input" in data or "messages" in data:
        body = {k: v for k, v in data.items() if k != "custom_id"}
    else:
        _fail(
            f"Line {line_no}: expected {{content}}, {{input, ...}} or a batch envelope {{body: {{...}}}}."
        )

    if "input" not in body and "messages" in body:
        body["input"] = body.pop("messages")
    if "max_output_tokens" not in body and "max_tokens" in body:
        body["max_output_tokens"] = body.pop("max_tokens")
    metadata = {
        **(body.get("metadata") or {}),
        "stream_id": stream_id,
        # Stream items are async work; the window only applies to background responses.
        "completion_window": "24h",
    }
    if custom_id is not None:
        metadata["custom_id"] = str(custom_id)

    allowed = {"model", "input", "metadata", *_STREAM_LINE_FIELDS}
    unknown = sorted(set(body) - allowed)
    if unknown:
        _fail(f"Line {line_no}: unsupported field(s) {', '.join(unknown)}.")
    return {
        **{k: body[k] for k in _STREAM_LINE_FIELDS if k in body},
        "model": body.get("model") or resolve_model(model),
        "input": body["input"],
        "metadata": metadata,
        "background": True,
    }


@stream_app.command("submit")
def stream_submit(
    stream_id: str = typer.Option(..., "--stream-id"),
    input_file: Path = typer.Option(
        ...,
        "--input-file",
        exists=True,
        dir_okay=False,
        help="JSONL: {content}, {input, ...}, or a batch envelope {custom_id?, body: {...}} per line.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help=f"Model for lines without their own (default: SFERENCE_MODEL or {DEFAULT_MODEL}).",
    ),
    base_url: Optional[str] = BASE_URL_OPTION,
    as_json: bool = JSON_OPTION,
) -> None:
    """Queue one background response per JSONL line on a stream (POST /v1/responses).

    Each response carries metadata.stream_id; follow completions with
    `sference responses tail --stream-id <id>`. The whole file is validated
    before anything is submitted.
    """
    _ensure_api_credential()
    requests = []
    for line_no, line in enumerate(
        input_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail(f"Line {line_no}: invalid JSON: {exc}")
        requests.append(
            _stream_line_request(line_no, data, model=model, stream_id=stream_id)
        )
    if not requests:
        _fail(f"{input_file} has no requests.")

    client = _client(base_url)
    created = [
        _call_api(lambda req=req: client.create_response(**req)).model_dump()
        for req in requests
    ]
    if as_json:
        _print(
            {"stream_id": stream_id, "total_items": len(created), "items": created},
            True,
        )
        return
    for r in created:
        typer.echo(f"{r.get('id')}  {r.get('status')}")
    typer.echo(f"Queued {len(created)} response(s) on stream {stream_id}.")


@stream_app.command("archive")
def stream_archive(
    stream_id: str = typer.Option(..., "--stream-id"),
    base_url: Optional[str] = BASE_URL_OPTION,
    as_json: bool = JSON_OPTION,
) -> None:
    """Finalize a stream; it accepts no new items."""
    _ensure_api_credential()
    client = _client(base_url)
    _print(_call_api(lambda: client.archive_stream(stream_id)).model_dump(), as_json)


@stream_app.command("cancel")
def stream_cancel(
    stream_id: str = typer.Option(..., "--stream-id"),
    base_url: Optional[str] = BASE_URL_OPTION,
    as_json: bool = JSON_OPTION,
) -> None:
    """Stop accepting new items and stop enqueueing pending work; in-flight items are not auto-cancelled."""
    _ensure_api_credential()
    client = _client(base_url)
    _print(_call_api(lambda: client.cancel_stream(stream_id)).model_dump(), as_json)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
