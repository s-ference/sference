"""Run integration example scripts against a local mock sference API."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples"


@dataclass(frozen=True)
class ExampleScriptCase:
    rel_path: str
    stdout_contains: tuple[str, ...] = ()
    expected_rows: int | None = None
    row_id_pattern: str | None = None
    output_json: Path | None = None
    local_only: bool = False
    needs_prefect_home: bool = False


EXAMPLE_CASES: tuple[ExampleScriptCase, ...] = (
    ExampleScriptCase(
        "dagster/document_enrichment_batch.py",
        stdout_contains=("doc-1:", "doc-2:", "doc-3:"),
        expected_rows=3,
        row_id_pattern=r"doc-\d:",
    ),
    ExampleScriptCase(
        "ray_data/classify_reviews_batch.py",
        stdout_contains=("Ray dataset: 4 rows", "Results:"),
        expected_rows=4,
    ),
    ExampleScriptCase(
        "hf_datasets/map_splits_to_batch.py",
        stdout_contains=("[ex-1]", "[ex-2]", "[ex-3]"),
        expected_rows=3,
        row_id_pattern=r"\[ex-\d+\]",
    ),
    ExampleScriptCase(
        "spark/enrich_dataframe_batch.py",
        stdout_contains=("Submitting batch (3 rows", "Enriched tickets:"),
        expected_rows=3,
        local_only=True,
    ),
    ExampleScriptCase(
        "airflow/sference_batch_dag.py",
        stdout_contains=("Wrote /tmp/sference_airflow_taglines.json",),
        expected_rows=3,
        output_json=Path("/tmp/sference_airflow_taglines.json"),
    ),
    ExampleScriptCase(
        "langchain/split_and_summarize_batch.py",
        stdout_contains=("LangChain produced", "Chunk summaries:"),
    ),
    ExampleScriptCase(
        "llamaindex/metadata_extraction_batch.py",
        stdout_contains=("LlamaIndex produced", "Node metadata:"),
    ),
    ExampleScriptCase(
        "label_studio/prelabel_classification_batch.py",
        stdout_contains=("Loaded 3 Label Studio tasks", "Wrote 3 pre-annotations"),
        expected_rows=3,
        output_json=EXAMPLES_ROOT / "label_studio" / "sample_predictions.json",
    ),
    ExampleScriptCase(
        "argilla/suggestion_batch.py",
        stdout_contains=("Loaded 2 Argilla-style records", "Wrote suggestions"),
        expected_rows=2,
        output_json=EXAMPLES_ROOT / "argilla" / "sample_suggestions.json",
    ),
    ExampleScriptCase(
        "promptfoo/provider.py",
        stdout_contains=(
            "Sference promptfoo provider",
            "Self-check complete: 3/3 produced output",
        ),
        expected_rows=3,
        row_id_pattern=r"\[\w+-review\]",
    ),
    ExampleScriptCase(
        "promptfoo/batch_generate.py",
        stdout_contains=("Submitted batch", "Wrote 3 precomputed rows"),
    ),
    ExampleScriptCase(
        "prefect/ai_data_analyst_batch_responses.py",
        stdout_contains=(
            "BATCH RESPONSE ANALYSIS",
            "--- dataset-overview (completed) ---",
            "--- anomaly-hunt (completed) ---",
            "--- recommendations (completed) ---",
        ),
        expected_rows=3,
        needs_prefect_home=True,
    ),
)


def _pyspark_available() -> bool:
    return importlib.util.find_spec("pyspark") is not None


def _java_available() -> bool:
    if shutil.which("java") is None:
        return False
    try:
        subprocess.run(["java", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        return False


def _skip_local_only(case: ExampleScriptCase) -> str | None:
    if not case.local_only:
        return None
    if not _pyspark_available():
        return "PySpark not installed (uv sync --group examples-spark)"
    if not _java_available():
        return "Java runtime not available for PySpark example"
    return None


def _count_pattern(text: str, pattern: str, *, flags: int = 0) -> int:
    return len(re.findall(pattern, text, flags=flags))


def _assert_row_count(case: ExampleScriptCase, stdout: str, output_json: Path | None) -> None:
    if case.expected_rows is None:
        if case.rel_path.startswith("langchain/"):
            produced = re.search(r"LangChain produced (\d+) chunks", stdout)
            assert produced is not None, stdout
            expected = int(produced.group(1))
            assert _count_pattern(stdout, r"^\s+\[\d+\]", flags=re.MULTILINE) == expected
        elif case.rel_path.startswith("llamaindex/"):
            produced = re.search(r"LlamaIndex produced (\d+) nodes", stdout)
            assert produced is not None, stdout
            expected = int(produced.group(1))
            assert _count_pattern(stdout, r"^\s+\S+-node-\d+:", flags=re.MULTILINE) == expected
        return

    if case.row_id_pattern:
        assert _count_pattern(stdout, case.row_id_pattern) == case.expected_rows, stdout

    if output_json is not None and output_json.exists():
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        assert isinstance(payload, list)
        assert len(payload) == case.expected_rows


def _prepare_env(case: ExampleScriptCase, tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["SFERENCE_API_KEY"] = "sk_test_examples"
    env.setdefault("RAY_DISABLE_IMPORT_WARNING", "1")
    if case.needs_prefect_home:
        env["PREFECT_HOME"] = str(tmp_path / "prefect")
    return env


@pytest.mark.parametrize("case", EXAMPLE_CASES, ids=[c.rel_path for c in EXAMPLE_CASES])
def test_example_script(case: ExampleScriptCase, mock_sference_server, tmp_path: Path) -> None:
    if reason := _skip_local_only(case):
        pytest.skip(reason)

    script = EXAMPLES_ROOT / case.rel_path
    assert script.is_file(), script

    env = _prepare_env(case, tmp_path)
    env["SFERENCE_BASE_URL"] = mock_sference_server.base_url
    env["SFERENCE_BATCH_POLL_INTERVAL_S"] = "0.1"
    env["SFERENCE_BATCH_WAIT_TIMEOUT_S"] = "30"
    # Keep the promptfoo offline-batch script's generated tests file out of the repo tree.
    env["BATCH_TESTS_OUT"] = str(tmp_path / "batch_tests.json")
    env["PYTHONUNBUFFERED"] = "1"
    # Intentionally NOT setting RAY_ENABLE_UV_RUN_RUNTIME_ENV here: the Ray example
    # must disable it itself (before `import ray`) for `uv run` to work.

    output_json = case.output_json
    if output_json is not None and output_json.parent.exists():
        output_json.unlink(missing_ok=True)

    combined_path = tmp_path / "combined.txt"
    with combined_path.open("w", encoding="utf-8") as combined_file:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=REPO_ROOT,
            env=env,
            stdout=combined_file,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
        )

    combined = combined_path.read_text(encoding="utf-8")
    assert proc.returncode == 0, combined
    for needle in case.stdout_contains:
        assert needle in combined, combined

    _assert_row_count(case, combined, output_json)
