# Contributing

## API contract (`contract/openapi.json`)

Contract tests validate mock fixtures against the OpenAPI document at `contract/openapi.json`.

Maintainers update it from the private monorepo **[github.com/s-ference/sference-platform](https://github.com/s-ference/sference-platform)**: run `just openapi` there (which regenerates `apps/api/openapi.json` and copies into `oss/contract/openapi.json` when the `oss` submodule is checked out). Then commit the change in this repo (**[github.com/s-ference/sference](https://github.com/s-ference/sference)**) and bump the submodule pointer in `sference-platform`.

If you only have this repo cloned, open a PR that updates `contract/openapi.json` when the public API surface changes; CI will run contract tests.

## Development

```bash
uv sync --group dev
uv run pytest -q
```

## License

By contributing, you agree your contributions are licensed under the Apache-2.0 license.
