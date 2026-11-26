# AGENTS.md

This repo is the source of the `sference-sdk` and `sference-cli` Python packages. To have a coding agent *use* sference from your own project, see [Coding agents](README.md#coding-agents) in the README.

## Working in this repo

```bash
uv sync --group dev --group examples-test
uv run pytest -q
```

[CONTRIBUTING.md](CONTRIBUTING.md) covers the API contract, version bumps and releases.
