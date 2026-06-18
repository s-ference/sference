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

import sys
from pathlib import Path

_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from _common import (
    COMPLETION_WINDOW,
    chat_batch_request,
    completion_text_from_row,
    index_results_by_custom_id,
    model_id,
    require_api_key,
    wait_for_batch_terminal,
)
from sference_sdk import SferenceClient

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
        chat_batch_request(
            custom_id=row.ticket_id,
            user_content=row.user_content,
            system_content=SYSTEM,
        )
        for row in prompt_rows
    ]

    print(f"Submitting batch ({len(requests)} rows, window={COMPLETION_WINDOW!r})...")
    batch = client.submit_batch(requests=requests, window=COMPLETION_WINDOW)
    terminal = wait_for_batch_terminal(client, batch.id)
    if terminal.status != "completed":
        raise RuntimeError(f"Batch ended as {terminal.status}")

    by_id = index_results_by_custom_id(client.get_results(batch.id).results)

    summaries = [
        {
            "ticket_id": row.ticket_id,
            "summary": completion_text_from_row(by_id.get(row.ticket_id, {})),
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
