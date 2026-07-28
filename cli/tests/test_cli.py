import json
from pathlib import Path
from unittest.mock import MagicMock

from typer.testing import CliRunner

from sference_sdk import ApiError
from sference_sdk.models import StreamEventList

import sference_cli.main as cli_main
from sference_cli.stream_cache import cache_key_from_file, get_cached_batch_id, set_cached_batch


runner = CliRunner()
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_cli_version_flag() -> None:
    from importlib.metadata import version

    r = runner.invoke(cli_main.app, ["--version"])
    assert r.exit_code == 0
    assert "sference-cli" in r.stdout
    assert version("sference-cli") in r.stdout
    assert "sference-sdk" in r.stdout
    assert version("sference-sdk") in r.stdout


def _with_fake_credential(monkeypatch) -> None:
    """auth me / batch commands require a stored credential before calling the API."""
    monkeypatch.setattr(cli_main, "_read_token", lambda: "sk_fake_for_tests")


class FakeResult:
    def __init__(self, payload: dict):
        self._payload = payload
        self.access_token = payload.get("access_token")

    def model_dump(self) -> dict:
        return self._payload


class FakeClient:
    def __init__(self, *args, **kwargs):
        pass

    def get_response(self, response_id: str):
        _ = response_id
        # Default fake: always in progress.
        return self.create_response(model="m", input=[{"role": "user", "content": "x"}])

    def get_me(self):
        return json.loads((FIXTURES / "V1AuthMeMe" / "200.json").read_text(encoding="utf-8"))

    def submit_batch(self, **kwargs):
        return FakeResult(json.loads((FIXTURES / "V1BatchesCreateBatch" / "201.json").read_text(encoding="utf-8")))

    def list_batches(self):
        return FakeResult(
            json.loads((FIXTURES / "V1BatchesListBatches" / "200.json").read_text(encoding="utf-8"))
        )

    def get_batch(self, batch_id: str):
        d = json.loads((FIXTURES / "V1BatchesBatchIdGetBatch" / "200.json").read_text(encoding="utf-8"))
        d["id"] = batch_id
        return FakeResult(d)

    def wait_for_completion(self, batch_id: str, poll_interval: float, timeout: float):
        return self.get_batch(batch_id)

    def get_results(self, batch_id: str):
        d = json.loads((FIXTURES / "V1BatchesBatchIdResultsGetResults" / "200.json").read_text(encoding="utf-8"))
        d["batch_id"] = batch_id
        return FakeResult(d)

    def cancel_batch(self, batch_id: str):
        d = json.loads((FIXTURES / "V1BatchesBatchIdCancelCancelBatch" / "200.json").read_text(encoding="utf-8"))
        d["id"] = batch_id
        return FakeResult(d)

    def download_results_jsonl(self, batch_id: str, out):
        out.write(b'{"batch_id": "' + batch_id.encode("utf-8") + b'"}\n')

    def create_stream(self, name: str, window: str = "24h"):
        return FakeResult(
            json.loads((FIXTURES / "V1StreamsCreateStream" / "201.json").read_text(encoding="utf-8"))
        )

    def list_streams(self):
        return FakeResult(
            json.loads((FIXTURES / "V1StreamsListStreams" / "200.json").read_text(encoding="utf-8"))
        )

    def get_stream(self, stream_id: str):
        d = json.loads((FIXTURES / "V1StreamsStreamIdGetStream" / "200.json").read_text(encoding="utf-8"))
        d["id"] = stream_id
        return FakeResult(d)

    def append_stream_items(self, stream_id: str, items: list):
        d = json.loads((FIXTURES / "V1StreamsStreamIdItemsAppendItems" / "200.json").read_text(encoding="utf-8"))
        d["id"] = stream_id
        return FakeResult(d)

    def submit_stream_items_from_file(self, stream_id: str, *, input_file: str, model: str | None = None):
        _ = input_file
        _ = model
        return self.append_stream_items(stream_id, items=[])

    def create_response(self, *, model: str, input: list, metadata: dict | None = None, **kwargs):
        """Mock create_response (keyword-only, matches SDK signature)."""
        _ = kwargs
        return FakeResult({
            "id": "resp_019d58a7e8b472c8a8f10346c9b7f3c5",
            "object": "response",
            "created_at": 1712345678,
            "model": model,
            "status": "in_progress",
            "metadata": metadata or {},
            "input": input,
            "output": [
                {
                    "type": "message",
                    "id": "msg_0",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "Mock response", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        })

    def archive_stream(self, stream_id: str):
        d = json.loads((FIXTURES / "V1StreamsStreamIdPatchStream" / "200.json").read_text(encoding="utf-8"))
        d["id"] = stream_id
        return FakeResult(d)

    def cancel_stream(self, stream_id: str):
        d = json.loads((FIXTURES / "V1StreamsStreamIdPatchStream" / "200.json").read_text(encoding="utf-8"))
        d["id"] = stream_id
        d["status"] = "cancelled"
        d["cancelled_at"] = "2026-04-04T10:18:00+00:00"
        d["archived_at"] = None
        return FakeResult(d)

    def list_responses_events(self, *, stream_id: str | None = None, **kwargs):
        _ = stream_id
        _ = kwargs
        raw = json.loads((FIXTURES / "V1ResponsesEventsListResponseEvents" / "200.json").read_text(encoding="utf-8"))
        return StreamEventList.model_validate(raw)

    def list_models(self):
        return json.loads((FIXTURES / "V1ModelsListModels" / "200.json").read_text(encoding="utf-8"))

    def create_embeddings(self, **kwargs):
        _ = kwargs
        return FakeResult(
            {
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
                "model": kwargs.get("model", "m"),
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }
        )


class FakeTrackingClient(FakeClient):
    def __init__(self) -> None:
        self.submit_calls = 0

    def submit_batch(self, **kwargs):
        self.submit_calls += 1
        return super().submit_batch(**kwargs)


class FakeFailedStreamClient(FakeClient):
    def get_batch(self, batch_id: str):
        d = json.loads((FIXTURES / "V1BatchesBatchIdGetBatch" / "200.json").read_text(encoding="utf-8"))
        d["id"] = batch_id
        d["status"] = "failed"
        return FakeResult(d)


def test_auth_login_api_key_noninteractive(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(cli_main, "CREDENTIALS_PATH", tmp_path / "credentials.json")
    result = runner.invoke(
        cli_main.app,
        ["auth", "login", "--api-key", "sk_test_cli_key_abc", "--no-validate"],
    )
    assert result.exit_code == 0
    assert "Credentials saved" in result.stdout
    data = json.loads((tmp_path / "credentials.json").read_text(encoding="utf-8"))
    assert data["token"] == "sk_test_cli_key_abc"


def test_auth_login_interactive_opens_browser(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(cli_main, "CREDENTIALS_PATH", tmp_path / "credentials.json")
    opened: list[str] = []

    def fake_open(url: str, new: int = 0, autoraise: bool = True) -> bool:
        opened.append(url)
        return True

    monkeypatch.setattr(cli_main.webbrowser, "open", fake_open)
    result = runner.invoke(
        cli_main.app,
        ["auth", "login", "--console-url", "http://localhost:3000", "--no-validate"],
        input="sk_pasted_key\n",
    )
    assert result.exit_code == 0
    assert any("/login" in u for u in opened)
    data = json.loads((tmp_path / "credentials.json").read_text(encoding="utf-8"))
    assert data["token"] == "sk_pasted_key"


def test_auth_login_no_browser(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(cli_main, "CREDENTIALS_PATH", tmp_path / "credentials.json")
    mock_open = MagicMock()
    monkeypatch.setattr(cli_main.webbrowser, "open", mock_open)
    result = runner.invoke(
        cli_main.app,
        ["auth", "login", "--no-browser", "--no-validate"],
        input="sk_no_browser\n",
    )
    assert result.exit_code == 0
    mock_open.assert_not_called()
    data = json.loads((tmp_path / "credentials.json").read_text(encoding="utf-8"))
    assert data["token"] == "sk_no_browser"


def test_models_list_json(monkeypatch):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr(cli_main, "SferenceClient", FakeClient)
    result = runner.invoke(cli_main.app, ["models", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["object"] == "list"
    ids = {m["id"] for m in payload["data"]}
    assert "Qwen/Qwen3.6-35B-A3B" in ids
    assert "moonshotai/Kimi-K2.6" in ids
    assert payload["data"][0]["capabilities"]["image_input"]["supported"] is False


def test_models_list_default(monkeypatch):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr(cli_main, "SferenceClient", FakeClient)
    result = runner.invoke(cli_main.app, ["models", "list"])
    assert result.exit_code == 0
    assert "display_name" in result.stdout
    assert "modality" in result.stdout
    assert "generation" in result.stdout
    assert "context" in result.stdout
    assert "vision" in result.stdout
    assert "Qwen/Qwen3.6-35B-A3B" in result.stdout
    assert "moonshotai/Kimi-K2.6" in result.stdout
    assert "262144" in result.stdout


def test_embeddings_create_json(monkeypatch):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr(cli_main, "SferenceClient", FakeClient)
    result = runner.invoke(
        cli_main.app,
        ["embeddings", "create", "--model", "Ettin/Ettin-Encoder-7B", "--input", "hello"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["object"] == "list"
    assert payload["data"][0]["embedding"] == [0.1]


def test_batch_list_json(monkeypatch):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr(cli_main, "SferenceClient", FakeClient)
    result = runner.invoke(cli_main.app, ["batch", "list", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert len(payload["items"]) == 2
    assert payload["items"][0]["id"] == "batch_1"


def test_batch_list_plain_table_includes_tokens(monkeypatch):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr(cli_main, "SferenceClient", FakeClient)
    result = runner.invoke(cli_main.app, ["batch", "list"])
    assert result.exit_code == 0
    assert "tokens" in result.stdout
    assert "150" in result.stdout
    assert "batch_1" in result.stdout


def test_batch_submit_rejects_unknown_window():
    result = runner.invoke(
        cli_main.app,
        ["batch", "submit", "--input-file", "/tmp/does-not-need-to-exist-for-cli-parse.jsonl", "--window", "2h"],
    )
    assert result.exit_code != 0
    out = f"{result.stdout}\n{result.stderr or ''}"
    assert "24h" in out


def test_batch_submit_accepts_all_completion_windows(monkeypatch, tmp_path: Path):
    input_file = tmp_path / "in.jsonl"
    input_file.write_text(
        '{"custom_id":"r1","method":"POST","url":"/v1/chat/completions","body":{"model":"m","messages":[{"role":"user","content":"hi"}]}}\n',
        encoding="utf-8",
    )
    for window in ("24h",):
        _with_fake_credential(monkeypatch)
        monkeypatch.setattr(cli_main, "SferenceClient", FakeClient)
        result = runner.invoke(
            cli_main.app,
            ["batch", "submit", "--input-file", str(input_file), "--window", window, "--json"],
        )
        assert result.exit_code == 0, result.stdout


def test_batch_stream_submits_and_outputs_jsonl(monkeypatch, tmp_path: Path):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr(cli_main, "SferenceClient", FakeClient)
    cache_path = tmp_path / "stream_cache.json"
    monkeypatch.setenv("SFERENCE_STREAM_CACHE", str(cache_path))
    input_file = tmp_path / "in.jsonl"
    input_file.write_text(
        '{"custom_id":"r1","method":"POST","url":"/v1/chat/completions","body":{"model":"m","messages":[{"role":"user","content":"hi"}]}}\n',
        encoding="utf-8",
    )
    result = runner.invoke(
        cli_main.app,
        [
            "batch",
            "stream",
            "--input-file",
            str(input_file),
            "--poll-interval",
            "0.01",
        ],
    )
    assert result.exit_code == 0
    combined = (result.stdout or "") + (result.stderr or "")
    assert "batch_1" in combined
    assert "status=completed" in combined
    assert '"batch_id"' in result.stdout
    assert get_cached_batch_id(cache_key_from_file(input_file), "https://api.sference.com") is None


def test_batch_stream_failed_batch_exits_1(monkeypatch, tmp_path: Path):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr(cli_main, "SferenceClient", FakeFailedStreamClient)
    cache_path = tmp_path / "stream_cache.json"
    monkeypatch.setenv("SFERENCE_STREAM_CACHE", str(cache_path))
    input_file = tmp_path / "in.jsonl"
    input_file.write_text(
        '{"custom_id":"r1","method":"POST","url":"/v1/chat/completions","body":{"model":"m","messages":[{"role":"user","content":"hi"}]}}\n',
        encoding="utf-8",
    )
    result = runner.invoke(
        cli_main.app,
        ["batch", "stream", "--input-file", str(input_file), "--poll-interval", "0.01"],
    )
    assert result.exit_code == 1
    combined = (result.stdout or "") + (result.stderr or "")
    assert "status=failed" in combined


def test_batch_stream_resumes_from_cache(monkeypatch, tmp_path: Path):
    _with_fake_credential(monkeypatch)
    fake = FakeTrackingClient()
    monkeypatch.setattr(cli_main, "SferenceClient", lambda *a, **k: fake)
    cache_path = tmp_path / "stream_cache.json"
    monkeypatch.setenv("SFERENCE_STREAM_CACHE", str(cache_path))
    input_file = tmp_path / "in.jsonl"
    input_file.write_text(
        '{"custom_id":"r1","method":"POST","url":"/v1/chat/completions","body":{"model":"m","messages":[{"role":"user","content":"hi"}]}}\n',
        encoding="utf-8",
    )
    set_cached_batch(cache_key_from_file(input_file), "batch_1", "https://api.sference.com")
    result = runner.invoke(
        cli_main.app,
        ["batch", "stream", "--input-file", str(input_file), "--poll-interval", "0.01"],
    )
    assert result.exit_code == 0
    assert fake.submit_calls == 0
    combined = (result.stdout or "") + (result.stderr or "")
    assert "Resuming" not in combined


def test_batch_stream_no_cache_flag_resubmits(monkeypatch, tmp_path: Path):
    _with_fake_credential(monkeypatch)
    fake = FakeTrackingClient()
    monkeypatch.setattr(cli_main, "SferenceClient", lambda *a, **k: fake)
    cache_path = tmp_path / "stream_cache.json"
    monkeypatch.setenv("SFERENCE_STREAM_CACHE", str(cache_path))
    input_file = tmp_path / "in.jsonl"
    input_file.write_text(
        '{"custom_id":"r1","method":"POST","url":"/v1/chat/completions","body":{"model":"m","messages":[{"role":"user","content":"hi"}]}}\n',
        encoding="utf-8",
    )
    set_cached_batch(cache_key_from_file(input_file), "batch_1", "https://api.sference.com")
    result = runner.invoke(
        cli_main.app,
        ["batch", "stream", "--input-file", str(input_file), "--poll-interval", "0.01", "--no-cache"],
    )
    assert result.exit_code == 0
    assert fake.submit_calls == 1


def test_batch_status_json(monkeypatch):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr(cli_main, "SferenceClient", FakeClient)
    result = runner.invoke(cli_main.app, ["batch", "status", "--batch-id", "batch_1", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "completed"


def test_auth_me_json(monkeypatch):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr(cli_main, "SferenceClient", FakeClient)
    result = runner.invoke(cli_main.app, ["auth", "me", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["username"] == "admin"


def test_batch_submit_wait_and_results_json(monkeypatch, tmp_path: Path):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr(cli_main, "SferenceClient", FakeClient)
    input_file = tmp_path / "content.jsonl"
    input_file.write_text('{"content":"hello"}\n', encoding="utf-8")

    submit = runner.invoke(
        cli_main.app,
        ["batch", "submit", "--input-file", str(input_file), "--model", "Qwen/Qwen3.5-4B", "--json"],
    )
    assert submit.exit_code == 0
    assert json.loads(submit.stdout)["id"] == "batch_1"

    wait = runner.invoke(cli_main.app, ["batch", "wait", "--batch-id", "batch_1", "--json"])
    assert wait.exit_code == 0
    assert json.loads(wait.stdout)["status"] == "completed"

    results = runner.invoke(cli_main.app, ["batch", "results", "--batch-id", "batch_1", "--json"])
    assert results.exit_code == 0
    assert json.loads(results.stdout)["output_url"] == "https://mock-s3.local/x"


def test_batch_cancel_json(monkeypatch):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr(cli_main, "SferenceClient", FakeClient)
    result = runner.invoke(cli_main.app, ["batch", "cancel", "--batch-id", "batch_1", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "cancelled"


def test_batch_download_results_jsonl(monkeypatch, tmp_path: Path):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr(cli_main, "SferenceClient", FakeClient)
    out = tmp_path / "results.jsonl"
    result = runner.invoke(
        cli_main.app,
        ["batch", "download-results", "--batch-id", "batch_1", "--out", str(out)],
    )
    assert result.exit_code == 0
    assert out.exists()
    assert out.read_text(encoding="utf-8").strip() != ""


def test_auth_me_no_credential_exits(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(cli_main, "CREDENTIALS_PATH", tmp_path / "no_credentials.json")
    monkeypatch.delenv("SFERENCE_API_KEY", raising=False)
    result = runner.invoke(cli_main.app, ["auth", "me"])
    assert result.exit_code == 1
    out = (result.stdout or "") + (result.stderr or "")
    assert "No API credential" in out
    assert "auth login" in out


def test_auth_me_401_shows_friendly_message(monkeypatch):
    _with_fake_credential(monkeypatch)

    class UnauthorizedClient:
        def __init__(self, *args, **kwargs):
            pass

        def get_me(self):
            raise ApiError("401: {'detail': 'Unauthorized'}")

    monkeypatch.setattr(cli_main, "SferenceClient", UnauthorizedClient)
    result = runner.invoke(cli_main.app, ["auth", "me"])
    assert result.exit_code == 1
    out = (result.stdout or "") + (result.stderr or "")
    assert "401" in out or "Unauthorized" in out
    assert "auth login" in out


class FakeResponsesTailClient(FakeClient):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._tail_i = 0

    def list_responses_events(self, *, stream_id: str | None = None, **kwargs):
        self._tail_i += 1
        if self._tail_i == 1:
            return super().list_responses_events(stream_id=stream_id, **kwargs)
        return StreamEventList.model_validate({"object": "list", "data": [], "has_more": False})


def test_stream_create_list_status_archive(monkeypatch, tmp_path: Path):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr(cli_main, "SferenceClient", FakeClient)
    r1 = runner.invoke(
        cli_main.app,
        ["stream", "create", "--name", "n1", "--window", "24h", "--json"],
    )
    assert r1.exit_code == 0
    assert json.loads(r1.stdout)["name"] == "daily-extraction"

    r2 = runner.invoke(cli_main.app, ["stream", "list", "--json"])
    assert r2.exit_code == 0
    assert len(json.loads(r2.stdout)["items"]) == 1

    r3 = runner.invoke(
        cli_main.app,
        ["stream", "status", "--stream-id", "123e4567-e89b-12d3-a456-426614174000", "--json"],
    )
    assert r3.exit_code == 0
    assert json.loads(r3.stdout)["total_items"] == 1

    r4 = runner.invoke(
        cli_main.app,
        ["stream", "cancel", "--stream-id", "123e4567-e89b-12d3-a456-426614174000", "--json"],
    )
    assert r4.exit_code == 0
    out = json.loads(r4.stdout)
    assert out["status"] == "cancelled"
    assert out["cancelled_at"] is not None


def test_stream_submit_jsonl(monkeypatch, tmp_path: Path):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr(cli_main, "SferenceClient", FakeClient)
    p = tmp_path / "items.jsonl"
    p.write_text(
        '{"content":"x"}\n',
        encoding="utf-8",
    )
    r = runner.invoke(
        cli_main.app,
        [
            "stream",
            "submit",
            "--stream-id",
            "123e4567-e89b-12d3-a456-426614174000",
            "--input-file",
            str(p),
            "--model",
            "m",
            "--json",
        ],
    )
    assert r.exit_code == 0
    assert json.loads(r.stdout)["total_items"] == 1


def test_responses_create_returns_json_with_id(monkeypatch):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr(cli_main, "SferenceClient", FakeClient)
    r = runner.invoke(
        cli_main.app,
        ["responses", "create", "--model", "m", "--content", "hello"],
    )
    assert r.exit_code == 0
    payload = json.loads(r.stdout)
    assert payload["id"].startswith("resp_")


def test_responses_create_passes_include_reasoning_flag(monkeypatch):
    _with_fake_credential(monkeypatch)

    captured: dict = {}

    class CapturingClient(FakeClient):
        def create_response(self, *, model: str, input: list, metadata: dict | None = None, **kwargs):
            captured.update(kwargs)
            return super().create_response(model=model, input=input, metadata=metadata, **kwargs)

    monkeypatch.setattr(cli_main, "SferenceClient", CapturingClient)
    r = runner.invoke(
        cli_main.app,
        ["responses", "create", "--model", "m", "--content", "hello", "--no-include-reasoning"],
    )
    assert r.exit_code == 0
    assert captured.get("include_reasoning") is False


def test_responses_create_passes_enable_thinking_flag(monkeypatch):
    _with_fake_credential(monkeypatch)

    captured: dict = {}

    class CapturingClient(FakeClient):
        def create_response(self, *, model: str, input: list, metadata: dict | None = None, **kwargs):
            captured.update(kwargs)
            return super().create_response(model=model, input=input, metadata=metadata, **kwargs)

    monkeypatch.setattr(cli_main, "SferenceClient", CapturingClient)
    r = runner.invoke(
        cli_main.app,
        ["responses", "create", "--model", "m", "--content", "hello", "--enable-thinking"],
    )
    assert r.exit_code == 0
    assert captured.get("enable_thinking") is True


def test_responses_create_defaults_to_foreground(monkeypatch):
    _with_fake_credential(monkeypatch)

    captured: dict = {}

    class CapturingClient(FakeClient):
        def create_response(self, *, model: str, input: list, metadata: dict | None = None, **kwargs):
            captured.update(kwargs)
            return super().create_response(model=model, input=input, metadata=metadata, **kwargs)

    monkeypatch.setattr(cli_main, "SferenceClient", CapturingClient)
    r = runner.invoke(
        cli_main.app,
        ["responses", "create", "--model", "m", "--content", "hello"],
    )
    assert r.exit_code == 0
    assert captured.get("background") is False


def test_responses_create_background_flag_returns_immediately(monkeypatch):
    _with_fake_credential(monkeypatch)

    captured: dict = {}

    class CapturingClient(FakeClient):
        def create_response(self, *, model: str, input: list, metadata: dict | None = None, **kwargs):
            captured.update(kwargs)
            return super().create_response(model=model, input=input, metadata=metadata, **kwargs)

    monkeypatch.setattr(cli_main, "SferenceClient", CapturingClient)
    r = runner.invoke(
        cli_main.app,
        ["responses", "create", "--model", "m", "--content", "hello", "--background"],
    )
    assert r.exit_code == 0
    assert captured.get("background") is True


def test_responses_create_passes_timeout_to_client(monkeypatch):
    _with_fake_credential(monkeypatch)

    captured: dict = {}

    class CapturingClient(FakeClient):
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(cli_main, "SferenceClient", CapturingClient)
    r = runner.invoke(
        cli_main.app,
        ["responses", "create", "--model", "m", "--content", "hello", "--timeout", "120"],
    )
    assert r.exit_code == 0
    assert captured.get("timeout") == 120.0


def test_responses_result_returns_json(monkeypatch):
    _with_fake_credential(monkeypatch)
    class CompletedClient(FakeClient):
        def get_response(self, response_id: str):
            r = super().get_response(response_id)
            d = r.model_dump()
            d["id"] = response_id
            d["status"] = "completed"
            return FakeResult(d)

    monkeypatch.setattr(cli_main, "SferenceClient", CompletedClient)
    r = runner.invoke(cli_main.app, ["responses", "result", "--id", "resp_any", "--poll-interval", "0.01"])
    assert r.exit_code == 0
    payload = json.loads(r.stdout)
    assert payload["object"] == "response"


def test_responses_tail_prints_event_then_interrupt(monkeypatch, tmp_path: Path):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr(cli_main, "SferenceClient", FakeResponsesTailClient)
    ck = tmp_path / "ck.json"
    monkeypatch.setenv("SFERENCE_STREAM_CHECKPOINTS", str(ck))

    sleeps = 0

    def fake_sleep(_sec: float) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps >= 2:
            raise KeyboardInterrupt()

    monkeypatch.setattr(cli_main.time, "sleep", fake_sleep)
    result = runner.invoke(
        cli_main.app,
        [
            "responses",
            "tail",
            "--stream-id",
            "123e4567-e89b-12d3-a456-426614174000",
            "--poll-interval",
            "0.001",
            "--no-checkpoint",
        ],
    )
    assert result.exit_code == 130
    assert "019d58a7" in result.stdout


def test_responses_tail_without_stream_id_prints_event_then_interrupt(monkeypatch, tmp_path: Path):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr(cli_main, "SferenceClient", FakeResponsesTailClient)
    ck = tmp_path / "ck.json"
    monkeypatch.setenv("SFERENCE_STREAM_CHECKPOINTS", str(ck))

    sleeps = 0

    def fake_sleep(_sec: float) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps >= 2:
            raise KeyboardInterrupt()

    monkeypatch.setattr(cli_main.time, "sleep", fake_sleep)
    result = runner.invoke(
        cli_main.app,
        ["responses", "tail", "--poll-interval", "0.001", "--no-checkpoint"],
    )
    assert result.exit_code == 130
    assert "019d58a7" in result.stdout


def test_responses_tail_passes_saved_checkpoint_as_starting_after_then_cursor(
    monkeypatch, tmp_path: Path
) -> None:
    """Tail loads checkpoint, lists with that ``starting_after``, then uses last event id on the next poll."""
    from sference_sdk.checkpoint import save_checkpoint

    _with_fake_credential(monkeypatch)
    monkeypatch.setenv("SFERENCE_STREAM_CHECKPOINTS", str(tmp_path / "ck.json"))
    save_checkpoint(
        "https://api.sference.com",
        "__responses_events__",
        "tail-cursor",
        "019d58a7-2ece-7742-bc3e-69ba44168210",
    )

    class Track(FakeClient):
        def __init__(self) -> None:
            self.starting_after_calls: list[str | None] = []
            self._n = 0

        def list_responses_events(self, *, stream_id: str | None = None, **kwargs):
            self.starting_after_calls.append(kwargs.get("starting_after"))
            self._n += 1
            if self._n == 1:
                raw = json.loads(
                    (FIXTURES / "V1ResponsesEventsListResponseEvents" / "200.json").read_text(encoding="utf-8")
                )
                return StreamEventList.model_validate(raw)
            return StreamEventList.model_validate({"object": "list", "data": [], "has_more": False})

    tracker = Track()
    monkeypatch.setattr(cli_main, "SferenceClient", lambda *a, **k: tracker)

    sleeps = 0

    def fake_sleep(_sec: float) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps >= 2:
            raise KeyboardInterrupt()

    monkeypatch.setattr(cli_main.time, "sleep", fake_sleep)
    result = runner.invoke(
        cli_main.app,
        ["responses", "tail", "--poll-interval", "0.001", "--consumer", "tail-cursor"],
    )
    assert result.exit_code == 130
    assert "019d58a7" in result.stdout
    assert tracker.starting_after_calls[0] == "019d58a7-2ece-7742-bc3e-69ba44168210"
    assert tracker.starting_after_calls[1] == "019d58a7-2ece-7742-bc3e-69ba44168279"


def test_cli_exits_nonzero_on_client_error(monkeypatch):
    _with_fake_credential(monkeypatch)

    class FailingClient(FakeClient):
        def get_batch(self, batch_id: str):
            raise RuntimeError("boom")

    monkeypatch.setattr(cli_main, "SferenceClient", FailingClient)
    result = runner.invoke(cli_main.app, ["batch", "status", "--batch-id", "batch_1", "--json"])
    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)


def test_launch_visible_in_root_help() -> None:
    result = runner.invoke(cli_main.app, ["--help"])
    assert result.exit_code == 0
    assert "launch" in result.stdout


def test_launch_still_invocable() -> None:
    result = runner.invoke(cli_main.app, ["launch", "--help"])
    assert result.exit_code == 0
    assert "claude" in result.stdout
    assert "pi" in result.stdout
    assert "opencode" in result.stdout


def test_launch_uses_sference_model_env(monkeypatch):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr("sference_cli.launch.find_claude_executable", lambda: "/usr/local/bin/claude")
    monkeypatch.setenv("SFERENCE_MODEL", "Qwen/Qwen3.6-35B-A3B")
    result = runner.invoke(cli_main.app, ["launch", "claude", "--no-anthropic", "--dry-run"])
    assert result.exit_code == 0
    assert "ANTHROPIC_MODEL=Qwen/Qwen3.6-35B-A3B" in result.stdout


def test_claude_dry_run_prints_env(monkeypatch):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr("sference_cli.launch.find_claude_executable", lambda: "/usr/local/bin/claude")
    result = runner.invoke(
        cli_main.app,
        ["launch", "claude", "--no-anthropic", "--dry-run", "--model", "moonshotai/Kimi-K2.6", "--enable-tool-search"],
    )
    assert result.exit_code == 0
    assert "ANTHROPIC_BASE_URL=https://api.sference.com" in result.stdout
    assert "ANTHROPIC_MODEL=moonshotai/Kimi-K2.6" in result.stdout
    assert "ANTHROPIC_DEFAULT_OPUS_MODEL=moonshotai/Kimi-K2.6" in result.stdout
    assert "ANTHROPIC_DEFAULT_SONNET_MODEL=moonshotai/Kimi-K2.6" in result.stdout
    assert "ANTHROPIC_DEFAULT_HAIKU_MODEL=moonshotai/Kimi-K2.6" in result.stdout
    assert "ANTHROPIC_DEFAULT_FABLE_MODEL=moonshotai/Kimi-K2.6" in result.stdout
    assert "CLAUDE_CODE_SUBAGENT_MODEL=moonshotai/Kimi-K2.6" in result.stdout
    assert "ANTHROPIC_AUTH_TOKEN=<redacted>" in result.stdout
    assert "ENABLE_TOOL_SEARCH=true" in result.stdout
    assert "command: /usr/local/bin/claude" in result.stdout


def test_claude_forwards_extra_args(monkeypatch):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr("sference_cli.launch.find_claude_executable", lambda: "/usr/local/bin/claude")
    captured: dict = {}

    def fake_execvpe(path: str, args: list[str], env: dict[str, str]) -> None:
        captured["path"] = path
        captured["args"] = args
        captured["env"] = env
        raise SystemExit(0)

    monkeypatch.setattr("sference_cli.launch.os.execvpe", fake_execvpe)
    result = runner.invoke(cli_main.app, ["launch", "claude", "--no-anthropic", "--resume", "abc"])
    assert result.exit_code == 0
    assert captured["path"] == "/usr/local/bin/claude"
    assert captured["args"] == ["/usr/local/bin/claude", "--resume", "abc"]
    assert captured["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk_fake_for_tests"
    assert captured["env"]["ANTHROPIC_MODEL"] == "moonshotai/Kimi-K2.7-Code"
    assert captured["env"]["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "moonshotai/Kimi-K2.7-Code"
    assert captured["env"]["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "moonshotai/Kimi-K2.7-Code"
    assert captured["env"]["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "moonshotai/Kimi-K2.7-Code"
    assert captured["env"]["ANTHROPIC_DEFAULT_FABLE_MODEL"] == "moonshotai/Kimi-K2.7-Code"
    assert captured["env"]["CLAUDE_CODE_SUBAGENT_MODEL"] == "moonshotai/Kimi-K2.7-Code"
    assert captured["env"]["ANTHROPIC_SMALL_FAST_MODEL"] == "moonshotai/Kimi-K2.7-Code"


def test_claude_missing_binary_exits(monkeypatch):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr("sference_cli.launch.find_claude_executable", lambda: None)
    result = runner.invoke(cli_main.app, ["launch", "claude", "--no-anthropic", "--dry-run"])
    assert result.exit_code == 1
    out = (result.stdout or "") + (result.stderr or "")
    assert "not found on PATH" in out


def test_claude_requires_credential(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(cli_main, "CREDENTIALS_PATH", tmp_path / "no_credentials.json")
    monkeypatch.delenv("SFERENCE_API_KEY", raising=False)
    result = runner.invoke(cli_main.app, ["launch", "claude", "--dry-run"])
    assert result.exit_code == 1
    out = (result.stdout or "") + (result.stderr or "")
    assert "No API credential" in out


def test_pi_dry_run_prints_config(monkeypatch, tmp_path: Path):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr("sference_cli.launch.find_pi_executable", lambda: "/usr/local/bin/pi")
    monkeypatch.setattr("sference_cli.launch._pi_models_json_path", lambda: tmp_path / "models.json")
    result = runner.invoke(
        cli_main.app,
        ["launch", "pi", "--dry-run", "--model", "moonshotai/Kimi-K2.6"],
    )
    assert result.exit_code == 0
    assert "Wrote provider 'sference' to" in result.stdout
    assert "baseUrl: https://api.sference.com/v1" in result.stdout
    assert "model: moonshotai/Kimi-K2.6" in result.stdout
    assert "apiKey: <redacted>" in result.stdout
    assert "command: /usr/local/bin/pi --provider sference --model moonshotai/Kimi-K2.6" in result.stdout


def test_pi_writes_models_json(monkeypatch, tmp_path: Path):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr("sference_cli.launch.find_pi_executable", lambda: "/usr/local/bin/pi")
    models_json = tmp_path / "models.json"
    monkeypatch.setattr("sference_cli.launch._pi_models_json_path", lambda: models_json)
    runner.invoke(cli_main.app, ["launch", "pi", "--dry-run"])
    assert models_json.exists()
    data = json.loads(models_json.read_text())
    assert data["providers"]["sference"]["api"] == "openai-completions"
    assert data["providers"]["sference"]["baseUrl"] == "https://api.sference.com/v1"
    assert data["providers"]["sference"]["apiKey"] == "sk_fake_for_tests"
    assert data["providers"]["sference"]["models"][0]["id"] == "moonshotai/Kimi-K2.7-Code"


def test_pi_forwards_extra_args(monkeypatch, tmp_path: Path):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr("sference_cli.launch.find_pi_executable", lambda: "/usr/local/bin/pi")
    monkeypatch.setattr("sference_cli.launch._pi_models_json_path", lambda: tmp_path / "models.json")
    captured: dict = {}

    def fake_execvpe(path: str, args: list[str], env: dict[str, str]) -> None:
        captured["path"] = path
        captured["args"] = args
        raise SystemExit(0)

    monkeypatch.setattr("sference_cli.launch.os.execvpe", fake_execvpe)
    result = runner.invoke(cli_main.app, ["launch", "pi", "--", "/path/to/project"])
    assert result.exit_code == 0
    assert captured["path"] == "/usr/local/bin/pi"
    assert captured["args"] == [
        "/usr/local/bin/pi",
        "--provider",
        "sference",
        "--model",
        "moonshotai/Kimi-K2.7-Code",
        "/path/to/project",
    ]


def test_pi_missing_binary_exits(monkeypatch):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr("sference_cli.launch.find_pi_executable", lambda: None)
    result = runner.invoke(cli_main.app, ["launch", "pi", "--dry-run"])
    assert result.exit_code == 1
    out = (result.stdout or "") + (result.stderr or "")
    assert "not found on PATH" in out


def test_pi_requires_credential(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(cli_main, "CREDENTIALS_PATH", tmp_path / "no_credentials.json")
    monkeypatch.delenv("SFERENCE_API_KEY", raising=False)
    result = runner.invoke(cli_main.app, ["launch", "pi", "--dry-run"])
    assert result.exit_code == 1
    out = (result.stdout or "") + (result.stderr or "")
    assert "No API credential" in out


def test_opencode_dry_run_prints_config(monkeypatch, tmp_path: Path):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr("sference_cli.launch.find_opencode_executable", lambda: "/usr/local/bin/opencode")
    monkeypatch.setattr("sference_cli.launch._opencode_config_path", lambda: tmp_path / "opencode.json")
    result = runner.invoke(
        cli_main.app,
        ["launch", "opencode", "--dry-run", "--model", "moonshotai/Kimi-K2.6"],
    )
    assert result.exit_code == 0
    assert "Wrote provider 'sference' to" in result.stdout
    assert "baseURL: https://api.sference.com/v1" in result.stdout
    assert "model: moonshotai/Kimi-K2.6" in result.stdout
    assert "apiKey: {env:SFERENCE_API_KEY}" in result.stdout
    assert "command: /usr/local/bin/opencode --model sference/moonshotai/Kimi-K2.6" in result.stdout


def test_opencode_writes_config(monkeypatch, tmp_path: Path):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr("sference_cli.launch.find_opencode_executable", lambda: "/usr/local/bin/opencode")
    config_path = tmp_path / "opencode.json"
    monkeypatch.setattr("sference_cli.launch._opencode_config_path", lambda: config_path)
    runner.invoke(cli_main.app, ["launch", "opencode", "--dry-run"])
    assert config_path.exists()
    data = json.loads(config_path.read_text())
    provider = data["provider"]["sference"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["name"] == "Sference"
    assert provider["options"]["baseURL"] == "https://api.sference.com/v1"
    assert provider["options"]["apiKey"] == "{env:SFERENCE_API_KEY}"
    assert provider["models"]["moonshotai/Kimi-K2.7-Code"]["name"] == "moonshotai/Kimi-K2.7-Code"


def test_opencode_merges_existing_config(monkeypatch, tmp_path: Path):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr("sference_cli.launch.find_opencode_executable", lambda: "/usr/local/bin/opencode")
    config_path = tmp_path / "opencode.json"
    # Pre-existing user config: another provider + a default model + keybinds.
    config_path.write_text(
        json.dumps(
            {
                "model": "anthropic/claude-sonnet-4-5",
                "keybinds": {"ctrl_k": "open"},
                "provider": {"openai": {"options": {"apiKey": "sk_user_openai"}}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("sference_cli.launch._opencode_config_path", lambda: config_path)
    result = runner.invoke(cli_main.app, ["launch", "opencode", "--dry-run", "--model", "Qwen/Qwen3.6-35B-A3B"])
    assert result.exit_code == 0
    data = json.loads(config_path.read_text())
    # User config preserved...
    assert data["model"] == "anthropic/claude-sonnet-4-5"
    assert data["keybinds"] == {"ctrl_k": "open"}
    assert data["provider"]["openai"]["options"]["apiKey"] == "sk_user_openai"
    # ...and sference merged in.
    sference = data["provider"]["sference"]
    assert sference["options"]["baseURL"] == "https://api.sference.com/v1"
    assert sference["models"]["Qwen/Qwen3.6-35B-A3B"]["name"] == "Qwen/Qwen3.6-35B-A3B"


def test_opencode_refuses_unparseable_existing_config(monkeypatch, tmp_path: Path):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr("sference_cli.launch.find_opencode_executable", lambda: "/usr/local/bin/opencode")
    config_path = tmp_path / "opencode.json"
    config_path.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr("sference_cli.launch._opencode_config_path", lambda: config_path)
    result = runner.invoke(cli_main.app, ["launch", "opencode", "--dry-run"])
    assert result.exit_code == 1
    out = (result.stdout or "") + (result.stderr or "")
    assert "Could not parse existing opencode config" in out
    # File left untouched.
    assert config_path.read_text() == "{not valid json"


def test_opencode_forwards_extra_args(monkeypatch, tmp_path: Path):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr("sference_cli.launch.find_opencode_executable", lambda: "/usr/local/bin/opencode")
    monkeypatch.setattr("sference_cli.launch._opencode_config_path", lambda: tmp_path / "opencode.json")
    captured: dict = {}

    def fake_execvpe(path: str, args: list[str], env: dict[str, str]) -> None:
        captured["path"] = path
        captured["args"] = args
        captured["env"] = env
        raise SystemExit(0)

    monkeypatch.setattr("sference_cli.launch.os.execvpe", fake_execvpe)
    result = runner.invoke(cli_main.app, ["launch", "opencode", "--", "/path/to/project"])
    assert result.exit_code == 0
    assert captured["path"] == "/usr/local/bin/opencode"
    assert captured["args"] == [
        "/usr/local/bin/opencode",
        "--model",
        "sference/moonshotai/Kimi-K2.7-Code",
        "/path/to/project",
    ]
    assert captured["env"]["SFERENCE_API_KEY"] == "sk_fake_for_tests"


def test_opencode_missing_binary_exits(monkeypatch):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr("sference_cli.launch.find_opencode_executable", lambda: None)
    result = runner.invoke(cli_main.app, ["launch", "opencode", "--dry-run"])
    assert result.exit_code == 1
    out = (result.stdout or "") + (result.stderr or "")
    assert "not found on PATH" in out


def test_opencode_requires_credential(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(cli_main, "CREDENTIALS_PATH", tmp_path / "no_credentials.json")
    monkeypatch.delenv("SFERENCE_API_KEY", raising=False)
    result = runner.invoke(cli_main.app, ["launch", "opencode", "--dry-run"])
    assert result.exit_code == 1
    out = (result.stdout or "") + (result.stderr or "")
    assert "No API credential" in out


# ── launch claude proxy mode (default) ────────────────────────────────────────

from sference_cli._proxy_routing import (  # noqa: E402
    BOOTSTRAP,
    PASSTHROUGH,
    SFERENCE,
    decide_routing,
    inject_models_into_bootstrap,
    parse_models_env,
    rewrite_request_for_sference,
    strip_unsigned_thinking_blocks,
)


def test_claude_proxy_dry_run_prints_config(monkeypatch):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr(
        "sference_cli.proxy.shutil.which",
        lambda b: "/usr/local/bin/mitmdump" if b == "mitmdump" else "/usr/local/bin/claude",
    )
    result = runner.invoke(
        cli_main.app, ["launch", "claude", "--model", "zai-org/GLM-5.2", "--dry-run"]
    )
    assert result.exit_code == 0
    assert "mitmdump on 127.0.0.1:" in result.stdout
    assert "Models (1): zai-org/GLM-5.2" in result.stdout
    assert "HTTPS_PROXY=http://127.0.0.1:" in result.stdout
    assert "NODE_EXTRA_CA_CERTS=" in result.stdout
    assert "ANTHROPIC_BASE_URL=<unset>" in result.stdout
    assert "command: /usr/local/bin/claude" in result.stdout
    # log lives at a stable, discoverable path (not the ephemeral temp dir)
    assert "claude_proxy.log" in result.stdout


def test_claude_proxy_missing_mitmproxy_exits(monkeypatch):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr("sference_cli.proxy.shutil.which", lambda b: None)
    result = runner.invoke(
        cli_main.app, ["launch", "claude", "--model", "zai-org/GLM-5.2", "--dry-run"]
    )
    assert result.exit_code == 1
    out = (result.stdout or "") + (result.stderr or "")
    assert "mitmproxy not found" in out
    assert "--no-anthropic" in out


def test_claude_proxy_launches_through_mitmproxy(monkeypatch, tmp_path: Path):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr(
        "sference_cli.proxy.shutil.which",
        lambda b: "/usr/local/bin/mitmdump" if b == "mitmdump" else "/usr/local/bin/claude",
    )
    monkeypatch.setattr("sference_cli.proxy._extract_addon_files", lambda: (tmp_path, tmp_path / "_proxy_addon.py"))
    monkeypatch.setattr("sference_cli.proxy.MITM_CA_CERT", tmp_path / "ca.pem")
    (tmp_path / "ca.pem").write_text("FAKE CA", encoding="utf-8")
    monkeypatch.setattr("sference_cli.proxy.port_in_use", lambda port: True)
    monkeypatch.setattr("sference_cli.proxy.time.sleep", lambda *_a: None)
    monkeypatch.setattr("sference_cli.proxy.signal.signal", lambda *_a: None)

    launches: list = []

    class FakeProc:
        def __init__(self, cmd, env):
            self.cmd = cmd
            self.env = env
            launches.append(self)

        def poll(self):  # mitmproxy "running"
            return None

        def wait(self, timeout=None):
            return 0

        @property
        def returncode(self):
            return 0

        def terminate(self):
            pass

        def kill(self):
            pass

    def fake_popen(cmd, **kwargs):
        return FakeProc(cmd, kwargs.get("env"))

    monkeypatch.setattr("sference_cli.proxy.subprocess.Popen", fake_popen)

    result = runner.invoke(
        cli_main.app,
        ["launch", "claude", "--model", "zai-org/GLM-5.2", "--", "-p", "hi"],
    )
    assert result.exit_code == 0
    assert len(launches) == 2  # mitmproxy, then claude
    assert launches[0].cmd[0] == "/usr/local/bin/mitmdump"
    assert "-s" in launches[0].cmd
    assert launches[1].cmd[0] == "/usr/local/bin/claude"
    assert launches[1].cmd[1:] == ["-p", "hi"]  # forwarded args
    claude_env = launches[1].env
    assert claude_env["HTTPS_PROXY"].startswith("http://127.0.0.1:")
    assert "NODE_EXTRA_CA_CERTS" in claude_env
    assert "ANTHROPIC_BASE_URL" not in claude_env
    mitm_env = launches[0].env
    assert mitm_env["SFERENCE_API_KEY"] == "sk_fake_for_tests"
    assert "zai-org/GLM-5.2" in mitm_env["SFERENCE_PROXY_MODELS"]
    assert mitm_env["SFERENCE_BASE_URL"] == "https://api.sference.com"


def test_claude_proxy_enable_tool_search_is_noop_notice(monkeypatch):
    _with_fake_credential(monkeypatch)
    monkeypatch.setattr(
        "sference_cli.proxy.shutil.which",
        lambda b: "/usr/local/bin/mitmdump" if b == "mitmdump" else "/usr/local/bin/claude",
    )
    result = runner.invoke(
        cli_main.app,
        ["launch", "claude", "--model", "zai-org/GLM-5.2", "--enable-tool-search", "--dry-run"],
    )
    assert result.exit_code == 0
    err = result.stderr or ""
    assert "no-op in proxy mode" in err


# ── pure routing helpers (no mitmproxy, no subprocess) ───────────────────────


def test_decide_routing() -> None:
    models = {"zai-org/GLM-5.2", "Qwen/Qwen3.6-35B-A3B"}
    assert decide_routing("downloads.claude.ai", "POST", "/v1/messages", {"model": "zai-org/GLM-5.2"}, models) == PASSTHROUGH
    assert decide_routing("api.anthropic.com", "GET", "/api/claude_cli/bootstrap?entrypoint=sdk-cli", None, models) == BOOTSTRAP
    assert decide_routing("api.anthropic.com", "POST", "/v1/messages?beta=true", {"model": "zai-org/GLM-5.2"}, models) == SFERENCE
    # real Claude model -> passthrough
    assert decide_routing("api.anthropic.com", "POST", "/v1/messages", {"model": "claude-sonnet-4-5"}, models) == PASSTHROUGH
    # GET on /v1/messages -> passthrough
    assert decide_routing("api.anthropic.com", "GET", "/v1/messages", None, models) == PASSTHROUGH
    # no model -> passthrough
    assert decide_routing("api.anthropic.com", "POST", "/v1/messages", {}, models) == PASSTHROUGH
    # non-dict body -> passthrough
    assert decide_routing("api.anthropic.com", "POST", "/v1/messages", None, models) == PASSTHROUGH


def test_rewrite_request_for_sference() -> None:
    url, set_h, remove_h = rewrite_request_for_sference(
        "https://api.anthropic.com/v1/messages?beta=true", "https://api.sference.com", "sk_test"
    )
    assert url == "https://api.sference.com/v1/messages?beta=true"
    assert set_h == {"x-api-key": "sk_test"}
    assert remove_h == ["authorization"]
    # accepts a base url that already has /v1 (strips it; path is preserved)
    url2, _, _ = rewrite_request_for_sference(
        "https://api.anthropic.com/v1/messages", "https://api.sference.com/v1", "sk_test"
    )
    assert url2 == "https://api.sference.com/v1/messages"


def test_inject_models_into_bootstrap() -> None:
    body = {"additional_model_options": [{"model": "claude-sonnet-4-5", "name": "Sonnet"}]}
    out = inject_models_into_bootstrap(body, {"zai-org/GLM-5.2", "Qwen/Qwen3.6-35B-A3B"})
    opts = out["additional_model_options"]
    rendered = {o["model"] for o in opts}
    assert "claude-sonnet-4-5" in rendered  # existing preserved
    assert {"zai-org/GLM-5.2", "Qwen/Qwen3.6-35B-A3B"} <= rendered  # added
    sference_opt = next(o for o in opts if o["model"] == "zai-org/GLM-5.2")
    assert sference_opt["name"] == "[Sference] zai-org/GLM-5.2"
    # idempotent (dedup)
    out2 = inject_models_into_bootstrap(out, {"zai-org/GLM-5.2"})
    assert len(out2["additional_model_options"]) == len(opts)
    # non-dict body returned unchanged
    assert inject_models_into_bootstrap("not a dict", {"x"}) == "not a dict"
    # missing options key -> created
    out3 = inject_models_into_bootstrap({}, {"zai-org/GLM-5.2"})
    assert out3["additional_model_options"][0]["model"] == "zai-org/GLM-5.2"


def test_strip_unsigned_thinking_blocks() -> None:
    # Sference-produced block (empty signature) is dropped; Anthropic-minted one stays.
    body = {
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "unsigned", "signature": ""},
                    {"type": "thinking", "thinking": "signed", "signature": "CAIS9Ag"},
                    {"type": "text", "text": "answer"},
                ],
            },
        ]
    }
    out, dropped = strip_unsigned_thinking_blocks(body)
    assert dropped == 1
    kept = out["messages"][1]["content"]
    assert [b["type"] for b in kept] == ["thinking", "text"]
    assert kept[0]["signature"] == "CAIS9Ag"  # signed reasoning preserved
    # string content (no blocks) untouched
    assert out["messages"][0]["content"] == "hi"

    # a missing signature key counts as unsigned (sibling block keeps the
    # message non-empty, so the strip is allowed to happen)
    body2 = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "x"},
                    {"type": "text", "text": "kept"},
                ],
            }
        ]
    }
    _, dropped2 = strip_unsigned_thinking_blocks(body2)
    assert dropped2 == 1

    # nothing to strip -> reported as 0 so the caller skips re-serializing
    body3 = {"messages": [{"role": "assistant", "content": [{"type": "text", "text": "a"}]}]}
    _, dropped3 = strip_unsigned_thinking_blocks(body3)
    assert dropped3 == 0

    # counts across multiple messages
    body4 = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "a", "signature": ""},
                    {"type": "text", "text": "one"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "b", "signature": ""},
                    {"type": "text", "text": "two"},
                ],
            },
        ]
    }
    _, dropped4 = strip_unsigned_thinking_blocks(body4)
    assert dropped4 == 2

    # a message that is ONLY unsigned thinking is left intact -- emitting an
    # empty content array would be a 400 we caused, worse than the one we avoid
    body5 = {
        "messages": [
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "a", "signature": ""}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "b", "signature": ""},
                    {"type": "text", "text": "kept"},
                ],
            },
        ]
    }
    out5, dropped5 = strip_unsigned_thinking_blocks(body5)
    assert dropped5 == 1  # only the strippable one counted
    assert len(out5["messages"][0]["content"]) == 1  # thinking-only left alone
    assert [b["type"] for b in out5["messages"][1]["content"]] == ["text"]

    # malformed shapes are returned unchanged rather than raising
    assert strip_unsigned_thinking_blocks("not a dict") == ("not a dict", 0)
    assert strip_unsigned_thinking_blocks({}) == ({}, 0)
    assert strip_unsigned_thinking_blocks({"messages": "nope"}) == ({"messages": "nope"}, 0)


def test_parse_models_env() -> None:
    assert parse_models_env('["a", "b"]') == {"a", "b"}
    assert parse_models_env("") == set()
    assert parse_models_env("not json") == set()
    assert parse_models_env('["", "c"]') == {"c"}  # empty strings dropped


