# Contributing

## API contract (`contract/openapi.json`)

Contract tests validate mock fixtures against the OpenAPI document at `contract/openapi.json`.

When the public API surface changes, update `contract/openapi.json` in your PR (and any fixtures under `sdk-python/tests/fixtures/` or `cli/tests/fixtures/` as needed). CI runs contract tests on every pull request.

## Development

```bash
uv sync --group dev
uv run pytest -q
```

## License

By contributing, you agree your contributions are licensed under the Apache-2.0 license.
