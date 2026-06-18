# ---
# PySpark: build prompts on executors, submit one sference batch from the driver, join results.
#
# Extra install (not in uv examples group by default):
#   uv pip install pyspark
#
# Run:
#   uv run python examples/spark/enrich_dataframe_batch.py
# ---

from __future__ import annotations

import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from sference_sdk import SferenceClient
from sference_sdk.models import InferenceRequest

DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
COMPLETION_WINDOW = "24h"
BATCH_POLL_INTERVAL_S = float(os.environ.get("SFERENCE_BATCH_POLL_INTERVAL_S", "5.0"))
BATCH_WAIT_TIMEOUT_S = float(os.environ.get("SFERENCE_BATCH_WAIT_TIMEOUT_S", "86400.0"))


def require_api_key() -> None:
    if not os.getenv("SFERENCE_API_KEY"):
        print("Error: SFERENCE_API_KEY is not set", file=sys.stderr)
        sys.exit(1)


def model_id() -> str:
    return os.environ.get("SFERENCE_MODEL", DEFAULT_MODEL)


client = SferenceClient()

SYSTEM = "Summarize the ticket in one sentence for a support manager."


def main() -> None:
    require_api_key()

    spark = (
        SparkSession.builder.appName("sference-batch-enrichment")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df = spark.createDataFrame(
        [
            ("t-100", "VPN drops every hour after the firmware update."),
            ("t-101", "Invoice #8842 shows double billing for March."),
            ("t-102", "Feature request: export audit logs to CSV."),
        ],
        ["ticket_id", "body"],
    )

    # Executors format prompts; driver owns the single-model batch contract.
    prompt_rows = (
        df.select(
            "ticket_id",
            F.concat(F.lit("Ticket:\n"), F.col("body")).alias("user_content"),
        )
        .collect()
    )

    requests = [
        InferenceRequest.chat(
            custom_id=row.ticket_id,
            user_content=row.user_content,
            system_content=SYSTEM,
            model=model_id(),
        )
        for row in prompt_rows
    ]

    print(f"Submitting batch ({len(requests)} rows, window={COMPLETION_WINDOW!r})...")
    batch = client.submit_batch(requests=requests, window=COMPLETION_WINDOW)
    terminal = client.wait_for_completion(
        batch.id,
        poll_interval=BATCH_POLL_INTERVAL_S,
        timeout=BATCH_WAIT_TIMEOUT_S,
    )
    if terminal.status != "completed":
        raise RuntimeError(f"Batch ended as {terminal.status}")

    by_id = client.get_results_indexed(batch.id)

    summaries = [
        {
            "ticket_id": row.ticket_id,
            "summary": (by_id[row.ticket_id].completion_text if row.ticket_id in by_id else ""),
        }
        for row in prompt_rows
    ]

    summary_df = spark.createDataFrame(summaries)
    enriched = df.join(summary_df, on="ticket_id", how="left")

    print("Enriched tickets:")
    enriched.show(truncate=False)

    spark.stop()


if __name__ == "__main__":
    print(f"Model: {model_id()}")
    main()
