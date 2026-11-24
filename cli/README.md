# sference CLI

Command-line interface for the sference API (`sference`): batches, streams, responses, decisions, and launchers that point coding agents (Claude Code, Codex, opencode, Pi, Cursor) at sference. It uses the Python SDK (`sference-sdk`) and is published on PyPI as `sference-cli`.

**Default model:** `deepseek-ai/DeepSeek-V4.1-Flash-1017` (DeepSeek V4.1 Flash). Every command that takes `--model` falls back to `SFERENCE_MODEL`, then to this default. Run `sference models` for the live catalog.

## Installation

```bash
# One-line install (macOS / Linux)
eval "$(curl -fsSL https://raw.githubusercontent.com/s-ference/sference/main/install.sh)"
```

Or install from PyPI:

```bash
uv tool install sference-cli
```

Fallback:

```bash
pip install sference-cli
# or:
pipx install sference-cli
```

From a clone of this repo:

```bash
uv sync --package sference-cli
uv run sference --help
```

Then:

```bash
sference --help
```

## Authentication

1. **Interactive (browser):** `sference auth login` signs in with the OAuth 2.0 device flow (RFC 8628). The CLI opens the console approval page and shows a short code; confirm it matches, approve the device, and the CLI finishes on its own. No API key is involved. Pass `--no-browser` on a remote or headless machine to print the URL and code instead.
2. **Non-interactive / CI:** `sference auth login --api-key 'sk_...'` saves an API key from **Console → API keys** without opening a browser.
3. **Environment variable:** `SFERENCE_API_KEY` is used when this device is **not** signed in with the device flow. A device login wins over the variable (the CLI says so once); if the device grant is revoked, the CLI falls back to `SFERENCE_API_KEY`.

Credentials are stored in `~/.sference/credentials.json` (owner-only permissions). A device login stores a 24 h access token and a 30 d rotating refresh token; the CLI refreshes the access token automatically when you run a command, and `sference launch claude` refreshes it at launch so a session starts on a fresh 24 h token. Launched tools (Claude Code, Codex, opencode, Pi) run on the token issued at launch. Signed-in devices are listed, and can be revoked, under **Profile → Signed-in devices** in the console.

For local development, `--base-url` (or `SFERENCE_BASE_URL`) points the device flow at another API and `--console-url` (or `SFERENCE_CONSOLE_URL`) overrides the approval page the browser opens.

Verify the current credential:

```bash
sference auth me
sference auth me --json
```

## Quick examples

`--model` is optional everywhere below; it defaults to `deepseek-ai/DeepSeek-V4.1-Flash-1017`.

**Models**

```bash
sference models          # table: id, context, capabilities, alias/retirement notes
sference models --json   # raw GET /v1/models
```

**Responses**

```bash
sference responses create --content "Summarize RFC 8628 in one line"
sference responses create --content "Long task" --background   # returns immediately
sference responses result --id <response_id>
```

**Batches**

```bash
sference batch submit --input-file ./workload.jsonl --window 24h
sference batch status --batch-id <batch_id>
sference batch wait --batch-id <batch_id>
sference batch results --batch-id <batch_id>
sference batch download-results --batch-id <batch_id> --out ./out.jsonl
# Submit, wait, print JSONL results on stdout (stderr: progress; resumable cache)
sference batch stream --input-file ./workload.jsonl --window 24h
```

**Streams**

```bash
sference stream create --name "my-stream" --window 24h
sference stream list
sference stream submit --stream-id <stream_id> --input-file ./lines.jsonl
sference stream status --stream-id <stream_id>
sference responses tail --stream-id <stream_id>
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `SFERENCE_API_KEY` | API key (or access token); used when the device is not signed in via `sference auth login` |
| `SFERENCE_BASE_URL` | API base URL for every command (default `https://api.sference.com`; same as `--base-url`) |
| `SFERENCE_CONSOLE_URL` | Console URL the device-flow approval page opens on (default: server-provided) |
| `SFERENCE_MODEL` | Default model for every command and launcher (default `deepseek-ai/DeepSeek-V4.1-Flash-1017`) |
| `SFERENCE_STREAM_CACHE` | Optional path to the stream resumable-cache file (default `~/.sference/stream_cache.json`) |
| `SFERENCE_STREAM_CHECKPOINTS` | Optional path for **`responses tail`** event checkpoints (default `~/.sference/stream_checkpoints.json`) |

## Commands

### Auth

| Command | Description |
|---------|-------------|
| `sference auth login` | Sign in with the browser device flow (`--no-browser` prints the URL and code; `--api-key` saves an API key for CI) |
| `sference auth me` | Show current user (`--json` for machine-readable output) |

### Launch

#### `sference launch claude`

`sference launch claude` points Claude Code at Sference's native `/v1/messages` endpoint via `ANTHROPIC_BASE_URL` and enables gateway model discovery: the live catalog is primed into Claude Code's picker cache before launch, so `/model` lists every Sference model the account can serve. No proxy, no mitmproxy.

| Command | Description |
| --- | --- |
| `sference launch claude` | Launch Claude Code against Sference; catalog primed into `/model` |
| `sference launch claude --dry-run` | Print the env config + command without launching |
| `sference launch claude --model zai-org/GLM-5.3` | Pin a different model (default `deepseek-ai/DeepSeek-V4.1-Flash-1017`) |
| `sference launch claude --models a,b` | Override the primed picker set (default: live `GET /v1/models`) |
| `sference launch claude --enable-tool-search` | Set `ENABLE_TOOL_SEARCH=true` |
| `sference launch claude -- -p "hi"` | Forward Claude Code flags/args after `--` |

Requires only the [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code). Every Claude Code model alias (opus/sonnet/haiku/fable, sub-agents) is pinned to the chosen model so no request goes out with a `claude-*` id. The default model accepts Claude Code's thinking-disabled request shape; the GLM-5.3 family does not (it always reasons), so pick it only for clients that send thinking. Device-flow credentials are refreshed at launch so long sessions start on a fresh 24 h token. `--no-anthropic` and `--proxy-port` are still accepted (hidden) as no-ops for older scripts.

All launchers fill their model picker from the live `GET /v1/models` catalog (text-generation models only); aliases past their `retires_on` date are left out, since the API asks callers to migrate to the `alias_of` target.

#### `sference launch pi`

| Command | Description |
|---------|-------------|
| `sference launch pi` | Launch Pi with Sference API routing (writes `~/.pi/agent/models.json`) |
| `sference launch pi --dry-run` | Print provider config and command without launching Pi |
| `sference launch pi --model zai-org/GLM-5.3` | Override catalog model |
| `sference launch pi -- /path/to/project` | Forward Pi args after `sference launch pi` |

Requires the [Pi CLI](https://pi.dev/docs) on `PATH`. Writes a `sference` provider block to `~/.pi/agent/models.json` using pi's `openai-responses` API (the Sference `/v1/responses` endpoint) with `baseUrl`, `apiKey`, and the full list of Sference-provided models (fetched live from `GET /v1/models`), then execs `pi --provider sference --model <id> --models sference/**`. The `--models sference/**` flag scopes pi's `/model` picker (and Ctrl+P cycling) to Sference models only, so other configured providers are hidden by default (`**` is required because Sference model ids contain a slash and `*` would match nothing). All catalog models appear in the picker; `--model` selects the active one. Reasoning is driven by pi's native `reasoning.effort` (no per-model thinking-format config needed). If the catalog fetch fails (offline), only the chosen model is registered. Uses `~/.sference/credentials.json` or `SFERENCE_API_KEY`. Default model: `deepseek-ai/DeepSeek-V4.1-Flash-1017` (override with `--model` or `SFERENCE_MODEL`).

#### `sference launch codex`

| Command | Description |
|---------|-------------|
| `sference launch codex` | Launch the [Codex CLI](https://developers.openai.com/codex/cli) against Sference's `/v1/responses` |
| `sference launch codex --dry-run` | Print provider config and command without launching Codex |
| `sference launch codex --model zai-org/GLM-5.3` | Override catalog model (default `deepseek-ai/DeepSeek-V4.1-Flash-1017`) |
| `sference launch codex -- exec "fix the bug"` | Forward Codex args (here: non-interactive `codex exec`) |
| `sference launch codex -- -c model_reasoning_effort=high` | Override the reasoning effort (default `medium`) |

Requires `codex` on `PATH`. The `sference` model provider is passed as `codex -c` overrides, so `~/.codex/config.toml` is never modified and your normal Codex sessions are unaffected; the API key reaches Codex through the `SFERENCE_API_KEY` env var, never on disk. The launcher disables Codex's hosted `web_search` tool (Sference's `/v1/responses` serves function tools, including Codex's `namespace` tool groups, but no hosted tools) and sets `model_context_window` from the live catalog so auto-compaction triggers at the model's real limit.

**Sub-agents and MCP servers** work: the Sference provider applies to the whole session, so agents Codex spawns run on Sference too. The launcher also writes `~/.sference/codex/models.json`, a Codex model catalog listing only the Sference models, and passes it as `model_catalog_json`. Without it Codex offers `spawn_agent` its own OpenAI models, and a sub-agent started on one of them fails. Each entry is built at launch from the catalog bundled with your Codex (`codex debug models --bundled`), so it always matches your Codex version; if that command is unavailable the launcher warns and starts without the catalog. An `agents.default_subagent_model` in your Codex config still pins sub-agents to an OpenAI model, which Sference cannot serve (the launcher warns about it).

#### `sference launch cursor`

| Command | Description |
|---------|-------------|
| `sference launch cursor` | Turn Sference routing on for Cursor |
| `sference launch cursor --dry-run` | Print what would change without writing anything |
| `sference launch cursor --model zai-org/GLM-5.3` | Override the default model for new chats (default `deepseek-ai/DeepSeek-V4.1-Flash-1017`) |
| `sference launch cursor --sference-only` | Disable all Cursor-provided models, leaving only Sference models |
| `sference launch cursor --api-key sk_...` | Store this API key in Cursor (default: your credential, or a newly created one) |
| `sference launch cursor status` | Show whether Cursor chats on Sference right now |
| `sference launch cursor off` | Restore Cursor's settings from before `launch cursor` |
| `sference launch cursor off --revoke-key` | Also revoke the API key that was created for Cursor |

Configures [Cursor](https://cursor.com) to run its chat on Sference through Cursor's own-API-key (BYOK) mode: the launcher writes the Sference base URL (`/v1`), your key, and the live model catalog into Cursor's settings database, and sets the default model for new chats (default `deepseek-ai/DeepSeek-V4.1-Flash-1017`). Cursor's own models stay available side by side unless `--sference-only` is passed. Requires a **signed-in paid Cursor plan** (Pro/Business/Enterprise) — BYOK is not available on the free plan.

Key handling: with an `sk_...` credential (or `--api-key`) that key is stored directly; with a device login the launcher creates a dedicated API key (tagged `sference-cli-cursor`, visible in Console → API Keys) so the 24 h device token never lands in another app. If Cursor already has an OpenAI key saved, the launcher refuses unless `--replace-existing-key` is passed (the old key cannot be restored; on macOS you will be asked to confirm the Keychain change).

Safety: quit Cursor first — it rewrites its settings from memory on exit and would discard the change (the launcher refuses while Cursor is running; `--force` overrides, and you must then fully restart Cursor). Everything is written in one transaction, the prior settings are backed up to `~/.sference/cursor/backup.json`, and `off` restores them. **Scope:** this covers Cursor chat and the model picker — Tab completions, chat titles, Agent Review (Bugbot), and cloud agents keep running on Cursor's own models, same as any BYOK setup.

### Batch

| Command | Description |
|---------|-------------|
| `sference batch list` | List batches (table; `--json` for raw payload) |
| `sference batch submit` | Submit a JSONL file (`--input-file`; `--model` applies to content-only lines and defaults to `SFERENCE_MODEL` / DeepSeek V4.1 Flash; `--window` `24h`) |
| `sference batch stream` | Submit, wait, print **JSONL results on stdout** (see below) |
| `sference batch status` | Get one batch (`--batch-id`, `--json`) |
| `sference batch wait` | Poll until terminal state (`--batch-id`, `--poll-interval`, `--timeout` (default 30 s), `--json`) |
| `sference batch results` | JSON results payload (`--batch-id`, `--json`) |
| `sference batch cancel` | Cancel a batch (`--batch-id`, `--json`) |
| `sference batch download-results` | Download results JSONL to a file (`--batch-id`, `--out`, `--format jsonl`) |

### Responses (`/v1/responses`)

| Command | Description |
|---------|-------------|
| `sference responses create` | Create one response (`--content`; optional `--model`, `--instructions`, `--max-output-tokens`, `--enable-thinking/--disable-thinking`; `--background` submits async and returns immediately). The sync default blocks; if the server-side wait times out, the request is **cancelled** and the command fails — use `--background` + `sference responses result` for long requests |
| `sference responses result` | Poll until terminal state (`--id`, `--poll-interval` seconds) |
| `sference responses tail` | Print completion events as JSONL via `GET /v1/responses/events` (optional `--stream-id` to scope to a stream; omit for non-stream completions). Flags: `--consumer`, `--from-latest`, `--no-checkpoint`, `--poll-interval` |

### Decisions (`/v1/decisions`)

| Command | Description |
|---------|-------------|
| `sference decisions create` | Answer up to 64 questions about one state in a single realtime call (`--model`, `--questions` JSON or `@file`, and `--state` text or `--state-json` JSON / `@file` / `@-`). Prints one line per answer; `--json` prints the full response |

```bash
sference decisions create --model Cloudflare/clef \
  --state "I was charged twice, please refund me" \
  --questions '{"route": {"type": "choice", "criteria": {"billing": "Payments", "support": "Other"}},
                "refund": {"type": "noul"}}'
```

### Stream (first-class streams API)

Long-lived **streams** are separate from **batches**: you create a stream, submit **responses** tied to it over time (`POST /v1/responses` with `metadata.stream_id`), and consume **completion events** with cursor-based pagination on **`GET /v1/responses/events`** (pass **`stream_id`** when scoping to a stream). Authenticate with your **secret API key** like other `/v1` calls.

| Command | Description |
|---------|-------------|
| `sference stream create` | Create a stream (`--name`, `--window` `24h`, `--json`) |
| `sference stream list` | List streams (`--json`) |
| `sference stream status` | Full detail + counters (`--stream-id`, `--json`) |
| `sference stream submit` | Queue one **background** response per JSONL line via `POST /v1/responses` (`metadata.stream_id` and `completion_window: "24h"` set automatically; `--stream-id`, `--input-file`, optional `--model` for lines without one). The whole file is validated before anything is sent |
| `sference stream cancel` | Stop accepting new items and stop enqueueing pending work; does not auto-cancel in-flight requests (`--stream-id`, `--json`) |
| `sference stream archive` | Finalize the stream (optional after cancel); no new items (`--stream-id`, `--json`) |

Accepted JSONL lines for `stream submit`:

```json
{"content":"hi"}
{"input":[{"role":"user","content":"hi"}],"instructions":"Be brief.","max_output_tokens":256}
{"custom_id":"req-1","method":"POST","url":"/v1/chat/completions","body":{"model":"zai-org/GLM-5.3","messages":[{"role":"user","content":"hi"}],"max_tokens":256}}
```

Chat-shaped bodies are mapped onto the Responses API (`messages` → `input`, `max_tokens` → `max_output_tokens`); `custom_id` lands in `metadata.custom_id`. Forwarded fields: `model`, `input`, `instructions`, `max_output_tokens`, `temperature`, `top_p`, `include_reasoning`, `enable_thinking`, `metadata`; any other field is rejected rather than silently dropped.

---

## Streaming batches (`batch stream`)

Use **`sference batch stream`** when you want a **single command** that submits a JSONL file, waits until the batch finishes, and **writes result lines to stdout** so you can pipe or redirect them.

### Pipe-friendly UX

- **Stdout:** only the **results JSONL** (one JSON object per line, same shape as `GET /v1/batches/{id}/results.jsonl`).
- **Stderr:** status lines while waiting, e.g. `Batch batch_abc status=running (42s)`.

Example:

```bash
sference batch stream --input-file workload.jsonl > results.jsonl
```

Content-only JSONL (model supplied globally; omit `--model` for the default):

```bash
sference batch stream --input-file prompts.jsonl --model zai-org/GLM-5.3 > results.jsonl
```

### Resumable cache

Batches can take a long time. If you **interrupt** the command (e.g. Ctrl+C) and run it again with the **same input file contents**, the CLI **reuses the cached batch id** instead of submitting a duplicate job.

- Cache file: **`~/.sference/stream_cache.json`** (override with **`SFERENCE_STREAM_CACHE`**).
- Key: **SHA-256** of the raw input file bytes (same bytes ⇒ same key, regardless of path).
- Stored fields: `batch_id`, `base_url`, `created_at` (an entry is only reused against the same API base URL).
- After results are written to stdout, the entry for that input is **removed** so the cache does not grow forever.
- If the cached batch no longer exists on the server (404), the cache entry is dropped and a **new** batch is submitted.

Force a **fresh** submission (ignore cache):

```bash
sference batch stream --input-file workload.jsonl --no-cache > results.jsonl
```

### Polling

- **`--poll-interval`** (default `2`): seconds between `GET /v1/batches/{id}` polls. There is **no** built-in maximum wait time (suited to 24h-style batches).

### Exit codes

- **0** — batch status is `completed`.
- **1** — batch status is `failed` or `cancelled` (results JSONL is still printed when available).

### End-to-end example

```bash
export SFERENCE_API_KEY=sk_...
sference batch stream --input-file fixtures/example_batch.jsonl --poll-interval 5 > out.jsonl
```

---

## JSONL input formats

The SDK and CLI accept two line shapes (see also [`fixtures/example_batch.jsonl`](fixtures/example_batch.jsonl)):

1. **OpenAI-compatible envelope:** each line has `custom_id`, `method`, `url`, and `body`. The API receives only `custom_id` + inner `body` (`method`/`url` are ignored). Inner `body` may use:
   - **Chat completions:** `messages` (non-empty array), plus optional `temperature`, `max_tokens`, `tools`, …
   - **Responses API:** `input` (string or message array), optional `instructions`, `max_output_tokens`, … — normalized to chat format at batch create.
2. **Content-only:** each line is `{"content": "..."}`. The model comes from `--model`, else `SFERENCE_MODEL`, else `deepseek-ai/DeepSeek-V4.1-Flash-1017`.

Invalid row bodies return **HTTP 400** at create with `requests[i]` and `custom_id` in the error message (nothing is enqueued). Do not set `background: true` inside batch row bodies.

Example Responses-shaped JSONL line:

```json
{"custom_id":"r2","method":"POST","url":"/v1/responses","body":{"model":"deepseek-ai/DeepSeek-V4.1-Flash-1017","input":[{"role":"user","content":"hi"}],"max_output_tokens":256}}
```

---

## Python SDK

The CLI uses the sync **`SferenceClient`** from **`sference-sdk`** (`import sference_sdk`).

For your own code, see **[`../sdk-python/README.md`](../sdk-python/README.md)** for:

- **Batches (sync):** `submit_batch`, `wait_for_completion`, `get_results`
- **`/v1/responses` (sync):** `create_response`, `get_response` (standalone or `metadata.stream_id` for streams)
- **Async:** **`AsyncSferenceClient`** — same surface as sync with `await`, plus `iter_responses_events` / `list_responses_events` for completion tailing (`GET /v1/responses/events`)

That README also documents **`./workload.jsonl`** input and when to prefer **batches** vs **streams**.
