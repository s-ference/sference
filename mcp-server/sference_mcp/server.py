"""Sference MCP server — expose Sference Agent Cloud as MCP tools.

Any MCP-compatible harness (Claude Desktop, Goose, Cline, Hermes, Zed, Kimi CLI)
connects to this server. The harness calls ``delegate_task`` to send work to
Sference's GPU agent loop. Sference runs the full agent loop (inference → tools
→ repeat) and returns the result.

Install:

    pip install sference-mcp

Configure (Claude Desktop ``claude_desktop_config.json``):

    {
      "mcpServers": {
        "sference": {
          "command": "sference-mcp",
          "env": {
            "SFERENCE_API_KEY": "sk_...",
            "SFERENCE_BASE_URL": "https://api.sference.com"
          }
        }
      }
    }
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

API_KEY = os.environ.get("SFERENCE_API_KEY", "")
BASE_URL = os.environ.get("SFERENCE_BASE_URL", "https://api.sference.com")
DEFAULT_MODEL = os.environ.get("SFERENCE_DEFAULT_MODEL", "zai-org/GLM-5.2")
TIMEOUT_S = int(os.environ.get("SFERENCE_TIMEOUT_S", "1860"))

if not API_KEY:
    print("SFERENCE_API_KEY not set — get one at https://app.sference.com/settings", file=sys.stderr)
    sys.exit(1)

mcp = FastMCP("sference")


@mcp.tool()
def list_models() -> str:
    """List GPU models available on Sference.

    These are exclusive models running on Sference's bare-metal GPUs
    (B200/B300/AMD BW100) — not available on OpenAI or Anthropic.

    Use this to discover which models you can pass to delegate_task.
    """
    try:
        r = httpx.get(
            f"{BASE_URL}/v1/models",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        models = data.get("data", [])
        if not models:
            return "No models available."
        lines = ["Available Sference models:"]
        for m in models:
            mid = m.get("id", "?")
            lines.append(f"  - {mid}")
        lines.append("")
        lines.append("Pass any model ID to delegate_task(model=...).")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing models: {e}"


@mcp.tool()
def delegate_task(
    task: str,
    model: str | None = None,
    tools: str | None = None,
    flex: bool = False,
    max_steps: int = 25,
    timeout_s: int = 1800,
    instructions: str | None = None,
    mcp_command: str | None = None,
    mcp_env_file: str | None = None,
) -> str:
    """Delegate a task to Sference Agent Cloud — a GPU-powered agent loop.

    The agent runs on Sference's bare-metal GPUs (B200/B300), executes tool
    calls in a loop until done, and returns structured output. Use this for
    tasks that need GPU reasoning, code execution, or multi-step agent loops
    with models like GLM-5.2, MiniMax-M3, or DeepSeek-V4.

    Args:
        task: The task description or question for the agent.
        model: Model to use (default: zai-org/GLM-5.2). Call list_models to see options.
        tools: Comma-separated tool types (e.g. "code_interpreter"). None = no tools.
        flex: Use flex tier for discounted pricing (waits for idle GPU, then runs all steps at realtime).
        max_steps: Maximum agent loop iterations (default: 25).
        timeout_s: Wall-clock timeout for the entire run in seconds (default: 1800 = 30min).
        instructions: Optional system instructions for the agent.
        mcp_command: MCP server command to connect to (e.g. "npx -y @modelcontextprotocol/server-github").
        mcp_env_file: Path to file containing env var for MCP server (e.g. /tmp/gh_token.txt with GITHUB_PERSONAL_ACCESS_TOKEN).

    Returns:
        Agent run result: status, steps, token usage, and output text.
    """
    tool_list: list[dict[str, Any]] = []
    if tools:
        for t in tools.split(","):
            t = t.strip()
            if t:
                tool_list.append({"type": t})

    # Add MCP server if configured
    if mcp_command:
        parts = mcp_command.split()
        mcp_tool: dict[str, Any] = {
            "type": "mcp",
            "name": "mcp_server",
            "transport": "stdio",
            "command": parts[0],
            "args": parts[1:] if len(parts) > 1 else [],
        }
        env: dict[str, str] = {}
        if mcp_env_file:
            try:
                with open(mcp_env_file) as f:
                    env["GITHUB_PERSONAL_ACCESS_TOKEN"] = f.read().strip()
            except Exception:
                pass
        if env:
            mcp_tool["env"] = env
        tool_list.append(mcp_tool)

    body: dict[str, Any] = {
        "model": model or DEFAULT_MODEL,
        "input": task,
        "max_steps": max_steps,
        "tools": tool_list,
        "service_tier": "flex" if flex else "default",
        "stream": False,
        "timeout_s": timeout_s,
    }
    if instructions:
        body["instructions"] = instructions

    try:
        with httpx.Client(timeout=TIMEOUT_S) as client:
            r = client.post(
                f"{BASE_URL}/v1/agents",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            r.raise_for_status()
            result = r.json()

        status = result.get("status", "?")
        steps = result.get("total_steps", "?")
        usage = result.get("total_usage", {})
        total_tokens = usage.get("total_tokens", 0)
        error = result.get("error")

        parts = [f"Status: {status}", f"Steps: {steps}", f"Tokens: {total_tokens}"]

        if error:
            parts.append(f"Error: {json.dumps(error)}")

        output_items = result.get("output") or []
        for item in output_items:
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        parts.append("")
                        parts.append(c["text"])
            elif item.get("type") == "reasoning":
                for s in item.get("summary", []):
                    if s.get("type") == "summary_text":
                        parts.append(f"[reasoning] {s['text']}")

        return "\n".join(parts)

    except httpx.HTTPStatusError as e:
        return f"Sference API error ({e.response.status_code}): {e.response.text[:500]}"
    except httpx.TimeoutException:
        return f"Sference API timeout after {TIMEOUT_S}s"
    except Exception as e:
        return f"Error: {e}"


def main() -> None:
    """Entry point for the sference-mcp console script."""
    mcp.run()


if __name__ == "__main__":
    main()
