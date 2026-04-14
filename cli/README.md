# sference CLI

Command-line interface for the sference batch API (`sference`). It uses the Python SDK (`sference-sdk`) and is published on PyPI as `sference-cli`.

## Installation

```bash
pip install sference-cli
# or isolated install on PATH:
uv tool install sference-cli
# or:
pipx install sference-cli
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

Use a `model` string supported by your sference deployment (for self-hosted stacks, match the model your workers consume).

**Batches**

```bash
sference batch submit --input-file ./workload.jsonl --model Qwen/Qwen2.5-7B-Instruct --window 24h
sference batch status --batch-id <batch_id>
sference batch wait --batch-id <batch_id>
sference batch results --batch-id <batch_id>
sference batch download-results --batch-id <batch_id> --out ./out.jsonl
# Submit, wait, print JSONL results on stdout (stderr: progress; resumable cache)
sference batch stream --input-file ./workload.jsonl --model Qwen/Qwen2.5-7B-Instruct --window 24h
```

**Streams**

```bash
sference stream create --name "my-stream" --window 24h
sference stream list
sference stream submit --stream-id <stream_id> --input-file ./lines.jsonl --model Qwen/Qwen2.5-7B-Instruct
sference stream status --stream-id <stream_id>
sference responses tail --stream-id <stream_id>
```

### cURL: OpenAI-compatible `/v1/responses`

The CLI uses this API for `stream submit` (via `POST /v1/responses`). You can call it directly with the same API key as `sference auth login` (or `SFERENCE_API_KEY`). For self-hosted APIs, set `SFERENCE_BASE_URL` to your API origin.

```bash
export TOKEN=sk_...   # or: export TOKEN="$SFERENCE_API_KEY"

RID=$(curl -sS -X POST "${SFERENCE_BASE_URL:-https://api.sference.com}/v1/responses" \
  -H "X-API-Key: $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "input": [{"role": "user", "content": "Hello"}],
    "metadata": {"completion_window": "24h"}
  }' | jq -r '.id')

curl -sS "${SFERENCE_BASE_URL:-https://api.sference.com}/v1/responses/${RID}" \
  -H "X-API-Key: $TOKEN"
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `SFERENCE_API_KEY` | API key (or JWT); overrides `~/.sference/credentials.json` |
| `SFERENCE_BASE_URL` | API base URL (default `https://api.sference.com`) |
| `SFERENCE_CONSOLE_URL` | Console URL for `auth login` browser step (default `https://console.sference.com`) |
| `SFERENCE_STREAM_CACHE` | Optional path to the stream resumable-cache file (default `~/.sference/stream_cache.json`) |
| `SFERENCE_STREAM_CHECKPOINTS` | Optional path for **`responses tail`** event checkpoints (default `~/.sference/stream_checkpoints.json`) |

## Commands

### Auth

| Command | Description |
|---------|-------------|
| `sference auth login` | Store an API key (optional `--api-key`, `--console-url`, `--no-browser`) |
| `sference auth me` | Show current user (`--json` for machine-readable output) |

### Batch

| Command | Description |
|---------|-------------|
| `sference batch list` | List batches (table; `--json` for raw payload) |
| `sference batch submit` | Submit a JSONL file (`--input-file`, optional `--model` for content-only lines, `--window` must be `24h`) |
| `sference batch stream` | Submit, wait, print **JSONL results on stdout** (see below) |
| `sference batch status` | Get one batch (`--batch-id`, `--json`) |
| `sference batch wait` | Poll until terminal state (`--batch-id`, `--poll-interval`, `--timeout`, `--json`) |
| `sference batch results` | JSON results payload (`--batch-id`, `--json`) |
| `sference batch cancel` | Cancel a batch (`--batch-id`, `--json`) |
| `sference batch download-results` | Download results JSONL to a file (`--batch-id`, `--out`, `--format jsonl`) |

Global options on most batch commands: `--base-url` (default `https://api.sference.com`).

### Responses (`/v1/responses`)

| Command | Description |
|---------|-------------|
| `sference responses create` | Create one response (`--model`, `--content`, optional `--wait`, `--poll-ms`, `--timeout-s`) |
| `sference responses result` | Poll until terminal state (`--id`, `--poll-ms`) |
| `sference responses tail` | Print completion events as JSONL via `GET /v1/responses/events` (optional `--stream-id` to scope to a stream; omit for non-stream completions). Flags: `--consumer`, `--from-latest`, `--no-checkpoint`, `--poll-ms` |

### Stream (first-class streams API)

Long-lived **streams** are separate from **batches**: you create a stream, submit **responses** tied to it over time (`POST /v1/responses` with `metadata.stream_id`), and consume **completion events** with cursor-based pagination on **`GET /v1/responses/events`** (pass **`stream_id`** when scoping to a stream). Authenticate with your **secret API key** like other `/v1` calls.

| Command | Description |
|---------|-------------|
| `sference stream create` | Create a stream (`--name`, `--window` `1h` or `24h`, `--json`) |
| `sference stream list` | List streams (`--json`) |
| `sference stream status` | Full detail + counters (`--stream-id`, `--json`) |
| `sference stream submit` | Create responses from JSONL via `POST /v1/responses` per line (`metadata.stream_id` set automatically; `--stream-id`, `--input-file`, `--model` required for content-only lines) — per line: OpenAI batch-style `{custom_id?, method, url, body}` or content-only `{content}` |
| `sference stream cancel` | Stop accepting new items and stop enqueueing pending work; does not auto-cancel in-flight requests (`--stream-id`, `--json`) |
| `sference stream archive` | Finalize the stream (optional after cancel); no new items (`--stream-id`, `--json`) |

Example JSONL lines for `stream submit` (both accepted):

```json
{"custom_id":"req-1","method":"POST","url":"/v1/chat/completions","body":{"model":"Qwen/Qwen3.5-4B","messages":[{"role":"user","content":"hi"}]}}
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
sference batch stream --input-file prompts.jsonl --model Qwen/Qwen2.5-7B-Instruct > results.jsonl
```

### Resumable cache

Batches can take a long time. If you **interrupt** the command (e.g. Ctrl+C) and run it again with the **same input file contents**, the CLI **reuses the cached batch id** instead of submitting a duplicate job.

- Cache file: **`~/.sference/stream_cache.json`** (override with **`SFERENCE_STREAM_CACHE`**).
- Key: **SHA-256** of the raw input file bytes (same bytes ⇒ same key, regardless of path).
- Stored fields: `batch_id`, `base_url` (must match current `--base-url`), `created_at`.
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

1. **OpenAI-compatible:** each line has `custom_id`, `method`, `url`, and `body` (e.g. chat completions payload with per-line `model`). The CLI `--model` flag is ignored for these lines (a warning may be emitted by the SDK).
2. **Content-only:** each line is `{"content": "..."}`. Then **`--model` is required** on submit/stream.

---

## Python SDK

The CLI uses the sync **`SferenceClient`** from **`sference-sdk`** (`import sference_sdk`).

For your own code, see **[`../sdk-python/README.md`](../sdk-python/README.md)** for:

- **Batches (sync):** `submit_batch`, `wait_for_completion`, `get_results`
- **`/v1/responses` (sync):** `create_response`, `get_response` (standalone or `metadata.stream_id` for streams)
- **Async:** **`AsyncSferenceClient`** — same surface as sync with `await`, plus `iter_responses_events` / `list_responses_events` for completion tailing (`GET /v1/responses/events`)

That README also documents when to prefer **batches** vs **streams** and includes cURL examples.
