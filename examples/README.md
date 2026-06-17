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

Install example dependencies from the **oss** repo root:

```bash
uv sync --group dev --group examples
```

Some examples need an extra package (documented in each README):

- **Spark:** `uv pip install pyspark` (+ Java for local mode)
- **Airflow:** `uv pip install "apache-airflow>=2.9"`
- **Argilla (live push):** `uv pip install argilla`

Shared helpers live in [`_common.py`](_common.py) (`chat_batch_request`, `wait_for_batch_terminal`, result parsing).

Each subdirectory has its own README with environment variables and run commands.
