# Airflow + sference batch

[`sference_batch_dag.py`](sference_batch_dag.py) is a small [Airflow 2.x](https://airflow.apache.org/) TaskFlow DAG:

| Task | Responsibility |
|------|----------------|
| `build_requests` | Pure Python — format prompts |
| `submit_batch_task` | `POST /v1/batches` |
| `wait_and_enrich` | Poll until terminal, join by `custom_id` |
| `write_artifact` | Persist JSON for downstream systems |

Airflow retries and SLA alerts apply to orchestration; sference owns GPU scheduling.

## Setup

```bash
uv sync --group dev --group examples
uv pip install "apache-airflow>=2.9"
export SFERENCE_API_KEY=sk_...
export AIRFLOW_HOME=/tmp/airflow-sference-example
airflow db migrate   # once
```

## Run (no webserver)

```bash
uv run python examples/airflow/sference_batch_dag.py
```

Uses `dag.test()` for a single local run. Deploy with `airflow dags test sference_product_taglines_batch` after copying the module into your `dags/` folder.
