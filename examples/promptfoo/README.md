# promptfoo (LLM evals) + sference

Evaluate prompts against sference-hosted models with [promptfoo](https://promptfoo.dev).

[`provider.py`](provider.py) is a promptfoo custom Python provider: promptfoo renders each
prompt + test case and calls `call_api`, which forwards the rendered prompt to sference's
`POST /v1/responses` and returns the completion. One request per row, so promptfoo's
concurrency parallelizes the calls and results stream in.

[`promptfooconfig.yaml`](promptfooconfig.yaml) is the eval — one prompt template, three
sentiment cases, built-in `icontains` assertions.

## Setup

```bash
uv sync --group dev --group examples
export SFERENCE_API_KEY=sk_...
export PROMPTFOO_PYTHON="$PWD/.venv/bin/python3"   # the venv with sference_sdk; promptfoo spawns its own python
```

`SFERENCE_MODEL` overrides the model; `SFERENCE_BASE_URL` targets a non-prod API.

## Run

```bash
npx promptfoo@latest eval -c examples/promptfoo/promptfooconfig.yaml
npx promptfoo@latest view        # results UI
```

Compare models by adding more entries under `providers:` (each with its own `config.model`);
promptfoo runs every test case against each.

Prefer no Python at all? sference's `/v1/responses` is OpenAI-shaped, so you can swap the
custom provider for promptfoo's built-in `openai:responses:<model>` (with `apiBaseUrl` +
`apiKey`) — verify auth and response shape against your API first.

## Very large / offline sweeps (batch)

One request per row is right for interactive evals. promptfoo providers are synchronous
(every built-in provider works this way), so for a huge sweep — where you'd rather pay one
bulk request than N — don't push batching into the provider. Instead generate the outputs
offline with one `submit_batch`, then have promptfoo score them:

```bash
uv run python examples/promptfoo/batch_generate.py                       # submit one batch, write batch_tests.json
npx promptfoo@latest eval -c examples/promptfoo/promptfooconfig.batch.yaml   # score with echo
```

[`batch_generate.py`](batch_generate.py) submits every row as one sference batch and writes
each precomputed completion into a generated tests file (`batch_tests.json`);
[`promptfooconfig.batch.yaml`](promptfooconfig.batch.yaml) scores those with promptfoo's
built-in `echo` provider — no live calls. You get batch cost/throughput while promptfoo
still does the assertions. (`batch_tests.json` is generated; don't commit it.)

## Self-check (no Node)

`provider.py` runs standalone against the SDK — handy for smoke-testing auth and wiring:

```bash
uv run python examples/promptfoo/provider.py
```
