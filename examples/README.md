# Examples

Runnable workflows that show how to integrate **sference-sdk** with common orchestration and data tools.

| Path | Description |
|------|-------------|
| [prefect/](prefect/) | Prefect flow — background `/v1/responses` (24h window) |
| [dagster/](dagster/) | Dagster assets — `submit_batch` document enrichment |
| [ray_data/](ray_data/) | Ray Data table → batch sentiment labels |
| [hf_datasets/](hf_datasets/) | Hugging Face `Dataset` → batch question generation |
| [spark/](spark/) | PySpark DataFrame join after batch summarization |
| [airflow/](airflow/) | Airflow TaskFlow DAG — submit, wait, write artifact |
| [langchain/](langchain/) | LangChain split + prompt template → batch summaries |
| [llamaindex/](llamaindex/) | LlamaIndex nodes → batch metadata extraction |
| [label_studio/](label_studio/) | Pre-label classification tasks for import |
| [argilla/](argilla/) | Offline reply suggestions for human feedback |
| [promptfoo/](promptfoo/) | promptfoo custom provider — eval prompts on sference models (+ offline batch) |

Install example dependencies from the **oss** repo root:

```bash
uv sync --group dev --group examples
```

Some examples need an extra package (documented in each README):

- **Spark:** `uv pip install pyspark` (+ Java for local mode)
- **Airflow:** `uv pip install "apache-airflow>=3.0"`
- **Argilla (live push):** `uv pip install argilla`
- **promptfoo (eval run):** Node.js + `npx promptfoo` (the Python provider itself needs no extra package; a `uv run` self-check works without Node)

Each example is self-contained: copy a script into your repo and set `SFERENCE_API_KEY` (and optional `SFERENCE_MODEL`, poll env vars). Batch workflow helpers (`InferenceRequest.chat`, `get_results_indexed`, `BatchResultRow.completion_text`) live in **sference-sdk**.

Each subdirectory has its own README with environment variables and run commands.

Run example script tests (mock API, no live inference):

```bash
uv sync --group dev --group examples-test
uv run pytest examples/tests -q
```

Spark is local-only (needs Java + PySpark; skipped in CI):

```bash
uv sync --group dev --group examples-test --group examples-spark
uv run pytest examples/tests -q
```
