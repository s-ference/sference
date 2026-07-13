# Sference MCP Server

Delegate tasks to Sference Agent Cloud from any MCP-compatible harness.

## What this gives you

Your chat assistant (Claude Desktop, Goose, Zed, etc.) gains the ability to
**delegate tasks to GPU models** running on Sference's bare-metal GPUs —
models like GLM-5.2, MiniMax-M3, and DeepSeek-V4 that aren't available on
OpenAI or Anthropic.

The agent runs a full loop (inference → tool calls → repeat) on Sference and
returns the result. You get GPU reasoning + code execution without leaving
your existing workflow.

## Install

```bash
pip install sference-mcp
```

Set your API key:

```bash
export SFERENCE_API_KEY="sk_..."
```

Get a key at [app.sference.com/settings](https://app.sference.com/settings).

## Configure your harness

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)
or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "sference": {
      "command": "sference-mcp",
      "env": {
        "SFERENCE_API_KEY": "sk_..."
      }
    }
  }
}
```

Restart Claude Desktop. You now have access to GPU models. Try asking Claude:
"Use sference to calculate the first 100 Fibonacci numbers with code interpreter"

### Goose (Block)

```yaml
# ~/.config/goose/config.yaml
mcp_servers:
  sference:
    command: sference-mcp
    env:
      SFERENCE_API_KEY: sk_...
```

### Cline / Roo Code (VSCode)

```json
// .cline/mcp_settings.json
{
  "mcpServers": {
    "sference": {
      "command": "sference-mcp",
      "env": {
        "SFERENCE_API_KEY": "sk_..."
      }
    }
  }
}
```

### Zed

Add to `~/.config/zed/settings.json`:

```json
{
  "agent_servers": {
    "sference": {
      "command": "sference-mcp",
      "env": {
        "SFERENCE_API_KEY": "sk_..."
      }
    }
  }
}
```

### Hermes Agent

```bash
hermes mcp add sference --command sference-mcp
# or use the sference-agent-cloud skill
```

### Kimi CLI

```bash
kimi config set mcp.sference.command sference-mcp
kimi config set mcp.sference.env.SFERENCE_API_KEY sk_...
```

## Tools

### `delegate_task`

Delegate a task to Sference Agent Cloud. The agent runs on Sference's
bare-metal GPUs, executes tool calls in a loop, and returns the result.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `task` | (required) | Task description or question |
| `model` | `zai-org/GLM-5.2` | Model to use |
| `tools` | `none` | Comma-separated: `code_interpreter`, `webhook`, `mcp` |
| `flex` | `false` | Flex tier — discount, waits for idle GPU |
| `max_steps` | `25` | Max agent loop iterations |
| `timeout_s` | `1800` | Wall-clock timeout (30 min) |
| `instructions` | `none` | System instructions for the agent |
| `mcp_command` | `none` | MCP server to connect (e.g. `npx -y @modelcontextprotocol/server-github`) |
| `mcp_env_file` | `none` | Path to file with env var for MCP server |

### `list_models`

List available GPU models on Sference.

## Flex tier

Pass `flex=true` to use discounted pricing. The agent waits once for idle GPU
capacity, then runs all steps at realtime priority. The flex penalty is paid
once per run, not per step — making multi-step agent tasks significantly
cheaper than per-request flex.

## Available models

```bash
curl -s "https://api.sference.com/v1/models" -H "Authorization: Bearer sk_..." | jq
```

Common: `zai-org/GLM-5.2`, `tencent/Hy3`, `MiniMaxAI/MiniMax-M3`, `deepseek-ai/DeepSeek-V4-Pro`.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SFERENCE_API_KEY` | (required) | API key from app.sference.com |
| `SFERENCE_BASE_URL` | `https://api.sference.com` | API endpoint |
| `SFERENCE_DEFAULT_MODEL` | `zai-org/GLM-5.2` | Default model for delegate_task |
| `SFERENCE_TIMEOUT_S` | `1860` | HTTP timeout for delegate_task |
