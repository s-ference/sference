"""Launcher for the Sference↔Claude Code transparent proxy.

Spawns a local mitmproxy forward proxy (installed as a separate ``uv tool`` so
``mitmdump`` is on PATH), then launches ``claude`` with ``HTTPS_PROXY`` pointing
at it and ``ANTHROPIC_BASE_URL`` *unset* — so Claude Code's first-party
detection stays on, Sference models appear in the native ``/model`` picker, and
real Claude models pass through to Anthropic. Sference models are routed to
Sference's native ``/v1/messages`` endpoint with the body passed through
untranslated (see ``_proxy_routing.py``).

The mitmproxy addon (``_proxy_addon.py``) and the pure routing helpers
(``_proxy_routing.py``) are shipped as package resources and extracted to a
temp dir at runtime so the addon can import the helpers via a sibling path
(mitmproxy's venv has no ``sference_cli``).
"""

from __future__ import annotations

import importlib.resources
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import typer

from sference_cli._proxy_routing import build_allow_hosts_regex

MITM_CA_CERT = Path.home() / ".mitmproxy" / "mitmproxy-ca-cert.pem"
DEFAULT_FLOW_DETAIL = "1"
# Stable, discoverable log path (not the ephemeral temp dir, which is hard to find
# and gets cleared by OS housekeeping). Lives with the other sference state files.
DEFAULT_LOG_FILE = Path.home() / ".sference" / "claude_proxy.log"


def _normalize_base_url(raw: str) -> str:
    """Return the API origin (scheme + netloc), accepting either
    https://api.sference.com or https://api.sference.com/v1."""
    raw = raw.rstrip("/")
    if raw.endswith("/v1"):
        raw = raw[: -len("/v1")]
    return raw


def fetch_sference_models(base_url: str, api_key: str) -> set[str]:
    """Pull the live text-generation model list from GET {base_url}/v1/models.

    Keeps the router in sync with what the account can serve, so a
    deprecated/renamed model never slips through and 400s mid-session.
    """
    api_root = _normalize_base_url(base_url) + "/v1"
    req = urllib.request.Request(
        f"{api_root}/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return {
        item["id"]
        for item in payload.get("data", [])
        if isinstance(item, dict)
        and item.get("id")
        and item.get("modality", "text_generation") == "text_generation"
    }


def pick_free_port() -> int:
    """Allocate an ephemeral port the OS guarantees is free at bind time."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def mitmproxy_available() -> bool:
    """mitmproxy is installed as a separate ``uv tool`` → ``mitmdump`` is on PATH."""
    return shutil.which("mitmdump") is not None


def _extract_addon_files() -> tuple[Path, Path]:
    """Extract the addon + routing helpers to a temp dir; return (tmpdir, addon_path).

    Both files live in the same dir so the addon can ``from _proxy_routing import …``
    via a sibling path (mitmproxy's venv has no ``sference_cli``).
    """
    res = importlib.resources.files("sference_cli")
    addon_src = (res / "_proxy_addon.py").read_text(encoding="utf-8")
    routing_src = (res / "_proxy_routing.py").read_text(encoding="utf-8")
    tmpdir = Path(tempfile.mkdtemp(prefix="sference_proxy_"))
    (tmpdir / "_proxy_routing.py").write_text(routing_src, encoding="utf-8")
    addon_path = tmpdir / "_proxy_addon.py"
    addon_path.write_text(addon_src, encoding="utf-8")
    return tmpdir, addon_path


def _wait_for_mitmproxy(proc: subprocess.Popen, port: int, log_file: Path) -> bool:
    """Wait for mitmproxy to bind the port; return False if it died."""
    for _ in range(40):
        if proc.poll() is not None:
            typer.echo(f"ERROR: mitmproxy exited immediately (code {proc.returncode}).", err=True)
            typer.echo(f"See {log_file} for the error.", err=True)
            return False
        if port_in_use(port):
            return True
        time.sleep(0.25)
    typer.echo(f"ERROR: mitmproxy did not start listening on {port}. See {log_file}.", err=True)
    return False


def _wait_for_ca_cert(timeout: float = 15.0) -> bool:
    """mitmproxy generates its CA on first run; wait for it to appear."""
    if MITM_CA_CERT.exists():
        return True
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if MITM_CA_CERT.exists():
            return True
        time.sleep(0.5)
    return False


def launch_claude_via_proxy(
    *,
    api_key: str,
    base_url: str,
    models: set[str],
    claude_args: list[str],
    dry_run: bool,
    proxy_port: Optional[int] = None,
    log_file: Optional[str] = None,
) -> None:
    """Start mitmproxy, then exec ``claude`` pointed at it. Blocks until claude exits."""
    if not mitmproxy_available():
        typer.echo(
            "mitmproxy not found. Install it with:  uv tool install mitmproxy\n"
            "Or run Claude Code without the proxy:  sference launch claude --no-anthropic",
            err=True,
        )
        raise typer.Exit(code=1)

    port = proxy_port or pick_free_port()
    sference_base = _normalize_base_url(base_url)
    log_path = Path(log_file) if log_file else DEFAULT_LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)

    claude_bin = shutil.which("claude")
    if claude_bin is None:
        typer.echo(
            "Claude Code CLI not found on PATH.\n"
            "Install from https://docs.anthropic.com/en/docs/claude-code "
            "and ensure the `claude` command is available.",
            err=True,
        )
        raise typer.Exit(code=1)

    if dry_run:
        typer.echo(f"mitmproxy: mitmdump on 127.0.0.1:{port} (log: {log_path})")
        typer.echo(f"Sference API: {sference_base}")
        typer.echo(f"Models ({len(models)}): {', '.join(sorted(models))}")
        typer.echo("apiKey: <redacted> (x-api-key, injected at launch)")
        typer.echo(f"HTTPS_PROXY=http://127.0.0.1:{port}")
        typer.echo(f"NODE_EXTRA_CA_CERTS={MITM_CA_CERT}")
        typer.echo("ANTHROPIC_BASE_URL=<unset> (first-party detection stays on)")
        typer.echo(f"allow_hosts={build_allow_hosts_regex(sference_base)} (only these hosts are TLS-intercepted)")
        typer.echo(f"command: {claude_bin} {' '.join(claude_args)}".rstrip())
        return

    tmpdir, addon_path = _extract_addon_files()
    mitm_env = {
        **os.environ,
        "SFERENCE_API_KEY": api_key,
        "SFERENCE_BASE_URL": sference_base,
        "SFERENCE_PROXY_MODELS": json.dumps(sorted(models)),
    }
    # Scope TLS interception to just the hosts we route. Without this, every
    # tool Claude Code spawns (Bash → curl, gh, …) also goes through the proxy
    # and fails TLS validation unless it trusts the mitmproxy CA. allow_hosts
    # makes non-matching hosts pass through as raw TCP — no CA involvement.
    allow_hosts = build_allow_hosts_regex(sference_base)
    mitm_cmd = [
        shutil.which("mitmdump"),
        "-p", str(port),
        "--listen-host", "127.0.0.1",
        "--set", f"flow_detail={os.environ.get('FLOW_DETAIL', DEFAULT_FLOW_DETAIL)}",
        "--set", f"allow_hosts={allow_hosts}",
        "-s", str(addon_path),
    ]

    log_fh = open(log_path, "w")  # truncate: never tail stale lines from a prior run
    mitm_proc = subprocess.Popen(
        mitm_cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=mitm_env
    )

    claude_proc: Optional[subprocess.Popen] = None

    def _cleanup(*_args: object) -> None:
        if claude_proc is not None:
            try:
                claude_proc.terminate()
            except Exception:
                pass
        if mitm_proc.poll() is None:
            mitm_proc.terminate()
            try:
                mitm_proc.wait(timeout=5)
            except Exception:
                mitm_proc.kill()
        log_fh.close()
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Install signal handlers so Ctrl-C tears down both children (Unix only).
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_a: (_cleanup(), sys.exit(130))[1])

    try:
        if not _wait_for_mitmproxy(mitm_proc, port, log_path):
            raise typer.Exit(code=1)

        if not _wait_for_ca_cert():
            typer.echo(
                f"ERROR: mitmproxy CA cert not found at {MITM_CA_CERT}. "
                "Run `mitmdump` once standalone to generate it.",
                err=True,
            )
            raise typer.Exit(code=1)

        # Quiet by default: one line. The full config (ports, proxy env, models)
        # is available via --dry-run; the proxy detail is in the log file.
        typer.echo("Launching Claude Code...")

        claude_env = os.environ.copy()
        claude_env["HTTPS_PROXY"] = f"http://127.0.0.1:{port}"
        claude_env["NODE_EXTRA_CA_CERTS"] = str(MITM_CA_CERT)
        claude_env.pop("ANTHROPIC_BASE_URL", None)  # keep first-party detection on

        claude_proc = subprocess.Popen(
            [claude_bin, *claude_args],
            env=claude_env,
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        claude_proc.wait()
        exit_code = claude_proc.returncode
    finally:
        _cleanup()

    raise typer.Exit(code=exit_code)
