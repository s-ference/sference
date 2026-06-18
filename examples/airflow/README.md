# Airflow + sference batch

[`sference_batch_dag.py`](sference_batch_dag.py) is a small [Airflow 3.x](https://airflow.apache.org/) TaskFlow DAG (imports from `airflow.sdk`):

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
uv pip install "apache-airflow>=3.0"
export SFERENCE_API_KEY=sk_...
export AIRFLOW_HOME=/tmp/airflow-sference-example
airflow db migrate   # once
```

## Run (no webserver)

```bash
uv run python examples/airflow/sference_batch_dag.py
```

Uses shared pipeline functions for a single local run (Airflow 3 `dag.test()` requires a serialized DAG bundle). Deploy by copying the module into your `dags/` folder and running tasks on a scheduler.
