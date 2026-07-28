# sference CLI

Command-line interface for the sference batch API (`sference`). It uses the Python SDK (`sference-sdk`) and is published on PyPI as `sference-cli`.

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

1. **Interactive (browser):** `sference auth login` — opens the console login page, then prompts for an API key from **Console → API keys**.
2. **Non-interactive / CI:** `sference auth login --api-key 'sk_...'`
3. **Environment variable:** `SFERENCE_API_KEY` overrides the saved credential file.

Credentials are stored in `~/.sference/credentials.json` unless `SFERENCE_API_KEY` is set.

Verify the current credential:

```bash
sference auth me
sference auth me --json
```

## Quick examples (batches and streams)

Use a `model` string supported by your sference deployment.

**Batches**

```bash
sference batch submit --input-file ./workload.jsonl --model Qwen/Qwen3.6-35B-A3B --window 24h
sference batch status --batch-id <batch_id>
sference batch wait --batch-id <batch_id>
sference batch results --batch-id <batch_id>
sference batch download-results --batch-id <batch_id> --out ./out.jsonl
# Submit, wait, print JSONL results on stdout (stderr: progress; resumable cache)
sference batch stream --input-file ./workload.jsonl --model Qwen/Qwen3.6-35B-A3B --window 24h
```

**Streams**

```bash
sference stream create --name "my-stream" --window 24h
sference stream list
sference stream submit --stream-id <stream_id> --input-file ./lines.jsonl --model Qwen/Qwen3.6-35B-A3B
sference stream status --stream-id <stream_id>
sference responses tail --stream-id <stream_id>
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `SFERENCE_API_KEY` | API key (or JWT); overrides `~/.sference/credentials.json` |
| `SFERENCE_MODEL` | Default catalog model for `sference launch` (overrides built-in default) |
| `SFERENCE_STREAM_CACHE` | Optional path to the stream resumable-cache file (default `~/.sference/stream_cache.json`) |
| `SFERENCE_STREAM_CHECKPOINTS` | Optional path for **`responses tail`** event checkpoints (default `~/.sference/stream_checkpoints.json`) |

## Commands

### Auth

| Command | Description |
|---------|-------------|
| `sference auth login` | Store an API key (optional `--api-key`, `--no-browser`) |
| `sference auth me` | Show current user (`--json` for machine-readable output) |

### Launch

#### `sference launch claude` (default: transparent proxy)

By default, `sference launch claude` starts a local [mitmproxy](https://mitmproxy.org) forward proxy and launches Claude Code with `HTTPS_PROXY` pointing at it (and `ANTHROPIC_BASE_URL` *unset*, so Claude Code's first-party detection stays on). Sference models are routed to Sference's native `/v1/messages` endpoint (body passed through untranslated) and appear in Claude Code's `/model` picker; real Claude models pass through to Anthropic. Hybrid routing — use both Sference and Claude models in one session.

| Command | Description |
|---------|-------------|
| `sference launch claude` | Start the proxy and launch Claude Code (default). Sference models appear in `/model`; Claude models pass through. |
| `sference launch claude --dry-run` | Print the proxy config + command without launching |
| `sference launch claude --model zai-org/GLM-5.2` | Inject this model into the `/model` picker (default: live `GET /v1/models`) |
| `sference launch claude --models a,b` | Inject a comma-separated set of models into the picker |
| `sference launch claude --proxy-port 8082` | Use a fixed local mitmproxy port (default: auto-pick) |
| `sference launch claude -- --model zai-org/GLM-5.2 -p "hi"` | Forward Claude Code flags/args after `--` |
| `sference launch claude --no-anthropic` | Disable the proxy; route everything directly to Sference via `ANTHROPIC_BASE_URL` (the previous behavior — no hybrid routing, no `/model` picker) |
| `sference launch claude --no-anthropic --enable-tool-search` | Direct mode: set `ENABLE_TOOL_SEARCH=true` on a custom host |

Proxy mode requires `mitmproxy` (`mitmdump` on `PATH`) and the [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code). The installer (`install.sh` / `install.ps1`) installs mitmproxy alongside the CLI; to add it to an existing install: `uv tool install mitmproxy`. If mitmproxy is missing, proxy mode errors with a `--no-anthropic` suggestion.

> **Security note:** the proxy uses mitmproxy's local CA cert (`~/.mitmproxy/mitmproxy-ca-cert.pem`), scoped to the Claude Code process via `NODE_EXTRA_CA_CERTS` — it is **not** added to system trust. The CA can sign certs for any host; only the Claude Code process trusts it for this session.

Uses `~/.sference/credentials.json` or `SFERENCE_API_KEY`. Default model: `moonshotai/Kimi-K2.7-Code` (override with `--model` or `SFERENCE_MODEL`).

#### `sference launch pi`

| Command | Description |
|---------|-------------|
| `sference launch pi` | Launch Pi with Sference API routing (writes `~/.pi/agent/models.json`) |
| `sference launch pi --dry-run` | Print provider config and command without launching Pi |
| `sference launch pi --model moonshotai/Kimi-K2.7-Code` | Override catalog model |
| `sference launch pi -- /path/to/project` | Forward Pi args after `sference launch pi` |

Requires the [Pi CLI](https://pi.dev/docs) on `PATH`. Writes a `sference` provider block to `~/.pi/agent/models.json` with `baseUrl`, `apiKey`, and the chosen `model`, then execs `pi --provider sference --model <id>`. Uses `~/.sference/credentials.json` or `SFERENCE_API_KEY`. Default model: `moonshotai/Kimi-K2.7-Code` (override with `--model` or `SFERENCE_MODEL`).

### Batch

| Command | Description |
|---------|-------------|
| `sference batch list` | List batches (table; `--json` for raw payload) |
| `sference batch submit` | Submit a JSONL file (`--input-file`, optional `--model` for content-only lines, `--window` `24h`) |
| `sference batch stream` | Submit, wait, print **JSONL results on stdout** (see below) |
| `sference batch status` | Get one batch (`--batch-id`, `--json`) |
| `sference batch wait` | Poll until terminal state (`--batch-id`, `--poll-interval`, `--timeout`, `--json`) |
| `sference batch results` | JSON results payload (`--batch-id`, `--json`) |
| `sference batch cancel` | Cancel a batch (`--batch-id`, `--json`) |
| `sference batch download-results` | Download results JSONL to a file (`--batch-id`, `--out`, `--format jsonl`) |

### Responses (`/v1/responses`)

| Command | Description |
|---------|-------------|
| `sference responses create` | Create one response (`--model`, `--content`; `--background` submits async and returns immediately). The sync default blocks; if the server-side wait times out, the request is **cancelled** and the command fails — use `--background` + `sference responses result` for long requests |
| `sference responses result` | Poll until terminal state (`--id`, `--poll-ms`) |
| `sference responses tail` | Print completion events as JSONL via `GET /v1/responses/events` (optional `--stream-id` to scope to a stream; omit for non-stream completions). Flags: `--consumer`, `--from-latest`, `--no-checkpoint`, `--poll-ms` |

### Stream (first-class streams API)

Long-lived **streams** are separate from **batches**: you create a stream, submit **responses** tied to it over time (`POST /v1/responses` with `metadata.stream_id`), and consume **completion events** with cursor-based pagination on **`GET /v1/responses/events`** (pass **`stream_id`** when scoping to a stream). Authenticate with your **secret API key** like other `/v1` calls.

| Command | Description |
|---------|-------------|
| `sference stream create` | Create a stream (`--name`, `--window` `24h`, `--json`) |
| `sference stream list` | List streams (`--json`) |
| `sference stream status` | Full detail + counters (`--stream-id`, `--json`) |
| `sference stream submit` | Create responses from JSONL via `POST /v1/responses` per line (`metadata.stream_id` set automatically; `--stream-id`, `--input-file`, `--model` required for content-only lines) — per line: OpenAI batch-style `{custom_id?, method, url, body}` or content-only `{content}` |
| `sference stream cancel` | Stop accepting new items and stop enqueueing pending work; does not auto-cancel in-flight requests (`--stream-id`, `--json`) |
| `sference stream archive` | Finalize the stream (optional after cancel); no new items (`--stream-id`, `--json`) |

Example JSONL lines for `stream submit` (both accepted):

```json
{"custom_id":"req-1","method":"POST","url":"/v1/chat/completions","body":{"model":"Qwen/Qwen3.6-35B-A3B","messages":[{"role":"user","content":"hi"}]}}
```

```json
{"content":"hi"}
```

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

Content-only JSONL (model supplied globally):

```bash
sference batch stream --input-file prompts.jsonl --model Qwen/Qwen3.6-35B-A3B > results.jsonl
```

### Resumable cache

Batches can take a long time. If you **interrupt** the command (e.g. Ctrl+C) and run it again with the **same input file contents**, the CLI **reuses the cached batch id** instead of submitting a duplicate job.

- Cache file: **`~/.sference/stream_cache.json`** (override with **`SFERENCE_STREAM_CACHE`**).
- Key: **SHA-256** of the raw input file bytes (same bytes ⇒ same key, regardless of path).
- Stored fields: `batch_id`, `created_at`.
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
2. **Content-only:** each line is `{"content": "..."}`. Then **`--model` is required** on submit/stream.

Invalid row bodies return **HTTP 400** at create with `requests[i]` and `custom_id` in the error message (nothing is enqueued). Do not set `background: true` inside batch row bodies.

Example Responses-shaped JSONL line:

```json
{"custom_id":"r2","method":"POST","url":"/v1/responses","body":{"model":"Qwen/Qwen3.6-35B-A3B","input":[{"role":"user","content":"hi"}],"max_output_tokens":256}}
```

---

## Python SDK

The CLI uses the sync **`SferenceClient`** from **`sference-sdk`** (`import sference_sdk`).

For your own code, see **[`../sdk-python/README.md`](../sdk-python/README.md)** for:

- **Batches (sync):** `submit_batch`, `wait_for_completion`, `get_results`
- **`/v1/responses` (sync):** `create_response`, `get_response` (standalone or `metadata.stream_id` for streams)
- **Async:** **`AsyncSferenceClient`** — same surface as sync with `await`, plus `iter_responses_events` / `list_responses_events` for completion tailing (`GET /v1/responses/events`)

That README also documents **`./workload.jsonl`** input and when to prefer **batches** vs **streams**.
