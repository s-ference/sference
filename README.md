# sference

Python SDK and CLI for the [sference](https://sference.com) batch inference API.

## Coding agents (Cursor, Claude Code, …)

1. **Paste [PROMPT.txt](PROMPT.txt)** into your project rules or `CLAUDE.md` — short rules for vibe coding with `sference-sdk`.
2. Optionally install the full **[SKILL.md](SKILL.md)** as a Cursor skill (`.cursor/skills/sference-sdk/SKILL.md`).

Raw URLs (no clone): [PROMPT.txt](https://raw.githubusercontent.com/s-ference/sference/main/PROMPT.txt) · [SKILL.md](https://raw.githubusercontent.com/s-ference/sference/main/SKILL.md)

Details: **[AGENTS.md](AGENTS.md)** · website: [sference.com/docs/sdk](https://sference.com/docs/sdk)

## Install

```bash
# One-line install (macOS / Linux) — updates PATH in this Terminal window
eval "$(curl -fsSL https://raw.githubusercontent.com/s-ference/sference/main/install.sh)"
```

The installer also writes `~/.local/bin` into your shell profile for future sessions. Use `eval "$(curl …)"` (not plain `curl | sh`) so the script can export PATH into your **current** shell — a piped script runs in a subshell and cannot change the parent process otherwise.

Or install from PyPI:

```bash
uv tool install sference-cli     # CLI (includes SDK)
# or library only:
uv add sference-sdk
```

Fallback:

```bash
pip install sference-cli
pip install sference-sdk
# or:
pipx install sference-cli
```

## Quick start

```python
from sference_sdk import SferenceClient

client = SferenceClient(api_key="sk_...")
batch = client.submit_batch(input_file="workload.jsonl", model="Qwen/Qwen3.6-35B-A3B", window="24h")
done = client.wait_for_completion(batch.id)
results = client.get_results(done.id)
```

```bash
sference auth login --api-key 'sk_...'
sference batch submit --input-file workload.jsonl --model Qwen/Qwen3.6-35B-A3B --window 24h
sference batch wait --batch-id <id>
```

## Packages

| Package | PyPI | Description |
|---------|------|-------------|
| [sdk-python](sdk-python/) | `sference-sdk` | Sync and async Python clients |
| [cli](cli/) | `sference-cli` | `sference` command-line interface |

## Examples

Orchestration recipes (Prefect, batch `/v1/responses`, etc.) live under [examples/](examples/).

**Batch API reference:** [docs/batches.md](docs/batches.md) (HTTP, JSONL, chat vs Responses row bodies).

## Development

```bash
uv sync --group dev
uv run pytest -q
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
