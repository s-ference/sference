# Contributing

## API contract (`contract/openapi.json`)

Contract tests validate mock fixtures against the OpenAPI document at `contract/openapi.json`.

When the public API surface changes, update `contract/openapi.json` in your PR (and any fixtures under `sdk-python/tests/fixtures/` or `cli/tests/fixtures/` as needed). CI runs contract tests on every pull request.

## Development

```bash
uv sync --group dev --group examples-test  # example tests need the framework deps
uvx pre-commit install  # runs ruff on every commit
uv run ruff check --fix . && uv run ruff format .
uv run pytest -q
```

## Releasing

The SDK (`sference-sdk`) and the CLI (`sference-cli`) are versioned and released independently. Each has its own tag and is published to PyPI by its own workflow.

| Package | Version in | Tag | Workflow |
|---------|------------|-----|----------|
| SDK | `sdk-python/pyproject.toml` | `sdk-v<version>` | `publish-sdk.yml` |
| CLI | `cli/pyproject.toml` | `cli-v<version>` | `publish-cli.yml` |

1. Bump the package version in a PR and run `uv lock`. If a PR changes `sdk-python/sference_sdk/` or `cli/sference_cli/` without bumping that package's version, the **Version bump** check fails. For changes that don't need publishing, such as comments or tests, add the `no-release` label to skip it.
2. If the CLI needs new SDK code, raise `sference-sdk>=` in `cli/pyproject.toml` to the new SDK version in the same PR.
3. After merging, tag the merge commit on `main` and push the tag, for example `git tag cli-v0.6.0 && git push origin cli-v0.6.0`. If the tag doesn't match the version in `pyproject.toml`, the publish workflow stops.
4. To release both packages, push both tags. The CLI workflow waits until the SDK version it requires can be installed from PyPI before it uploads, so it never publishes a CLI that can't be installed.

Tags before 0.5.0 used a shared `v<version>` tag that published both packages.

## License

By contributing, you agree your contributions are licensed under the Apache-2.0 license.
