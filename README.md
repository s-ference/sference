# sference

[![PyPI - sference-sdk](https://img.shields.io/pypi/v/sference-sdk?label=sference-sdk)](https://pypi.org/project/sference-sdk/)
[![PyPI - sference-cli](https://img.shields.io/pypi/v/sference-cli?label=sference-cli)](https://pypi.org/project/sference-cli/)
[![CI](https://github.com/s-ference/sference/actions/workflows/ci.yml/badge.svg)](https://github.com/s-ference/sference/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Python SDK and CLI for the [sference](https://sference.com) inference API: batches, background and streaming responses, and realtime decisions.

## Install

Requires Python 3.11+.

```bash
uv tool install sference-cli     # CLI (includes the SDK)
uv add sference-sdk              # library only
```

With pip: `pip install sference-cli` or `pip install sference-sdk` (or `pipx install sference-cli`).

No Python toolchain? The install script sets up [uv](https://docs.astral.sh/uv/) and the CLI for you:

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/s-ference/sference/main/install.sh | sh
```

```powershell
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/s-ference/sference/main/install.ps1 | iex"
```

The script adds `~/.local/bin` to your shell profile, so new terminals find `sference`. To use it in the current terminal right away, wrap the command: `eval "$(curl -fsSL https://raw.githubusercontent.com/s-ference/sference/main/install.sh | sh)"`.

## Authenticate

Run `sference auth login` to sign in through the browser. For CI and the SDK, create an API key (`sk_...`) in the [console](https://app.sference.com) under API keys. Set it as `SFERENCE_API_KEY` (both the SDK and CLI read it) or pass it to `sference auth login --api-key`.

## Quick start

```python
from sference_sdk import SferenceClient

client = SferenceClient()  # reads SFERENCE_API_KEY
batch = client.submit_batch(input_file="workload.jsonl", model="deepseek-ai/DeepSeek-V4.1-Flash-1017", window="24h")
done = client.wait_for_completion(batch.id)
results = client.get_results(done.id)
```

```bash
sference auth login   # browser device flow; --api-key 'sk_...' for CI
sference batch submit --input-file workload.jsonl --model deepseek-ai/DeepSeek-V4.1-Flash-1017 --window 24h
sference batch wait --batch-id <id>
```

## Packages

| Package | PyPI | Description |
|---------|------|-------------|
| [sdk-python](sdk-python/) | `sference-sdk` | Sync and async Python clients |
| [cli](cli/) | `sference-cli` | `sference` command-line interface |

## Coding agents (Cursor, Claude Code, …)

1. **Paste [PROMPT.txt](PROMPT.txt)** into your project rules or `CLAUDE.md` — short rules for vibe coding with `sference-sdk`.
2. Optionally install the full **[SKILL.md](SKILL.md)** as a Cursor skill (`.cursor/skills/sference-sdk/SKILL.md`).

Raw URLs (no clone): [PROMPT.txt](https://raw.githubusercontent.com/s-ference/sference/main/PROMPT.txt) · [SKILL.md](https://raw.githubusercontent.com/s-ference/sference/main/SKILL.md)

Details: **[AGENTS.md](AGENTS.md)** · website: [sference.com/docs/sdk](https://sference.com/docs/sdk)

## Examples

[examples/](examples/) has runnable integrations with Airflow, Dagster, Prefect, Spark, Ray Data, Hugging Face Datasets, LangChain, LlamaIndex, Label Studio and Argilla.

**Batch API reference:** [docs/batches.md](docs/batches.md) (HTTP, JSONL, chat vs Responses row bodies).

## Development

```bash
uv sync --group dev --group examples-test  # example tests need the framework deps
uv run pytest -q
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
