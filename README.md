# sference

Python SDK and CLI for the [sference](https://sference.com) batch inference API.

## Install

```bash
# One-line install (macOS / Linux)
curl -fsSL https://raw.githubusercontent.com/s-ference/sference/main/install.sh | sh
```

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

## AI agents (Cursor, etc.)

A Cursor / Claude Code skill for writing **`sference-sdk`** Python without reading this repo lives in **[`.cursor/skills/sference-sdk/SKILL.md`](.cursor/skills/sference-sdk/SKILL.md)**. Copy that folder into your project’s `.cursor/skills/` (or personal agent skills) when coding against sference.

## Examples

Orchestration recipes (Prefect, batch `/v1/responses`, etc.) live under [examples/](examples/).

## Development

```bash
uv sync --group dev
uv run pytest -q
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
