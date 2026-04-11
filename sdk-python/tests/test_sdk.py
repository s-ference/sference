import json
import warnings
from pathlib import Path

import httpx
import pytest

from sference_sdk.client import ApiError, SferenceClient

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_parse_openai_jsonl_format(tmp_path: Path) -> None:
    p = tmp_path / "openai.jsonl"
    p.write_text(
        '{"custom_id":"request-1","method":"POST","url":"/v1/chat/completions","body":{"model":"Qwen/Qwen3.5-4B","messages":[{"role":"user","content":"hi"}]}}\n',
        encoding="utf-8",
    )
    with pytest.warns(UserWarning, match="model argument is ignored"):
        parsed = SferenceClient._parse_jsonl(p, model="ignored")
    assert len(parsed) == 1
    assert parsed[0]["custom_id"] == "request-1"


def test_parse_openai_jsonl_no_model_no_warning(tmp_path: Path) -> None:
    p = tmp_path / "openai.jsonl"
    p.write_text(
        '{"custom_id":"request-1","method":"POST","url":"/v1/chat/completions","body":{"model":"Qwen/Qwen3.5-4B","messages":[{"role":"user","content":"hi"}]}}\n',
        encoding="utf-8",
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        parsed = SferenceClient._parse_jsonl(p, model=None)
    assert len(w) == 0
    assert len(parsed) == 1


def test_parse_openai_jsonl_model_warns_once_per_file(tmp_path: Path) -> None:
    line = '{"custom_id":"r","method":"POST","url":"/v1/chat/completions","body":{"model":"m","messages":[{"role":"user","content":"x"}]}}\n'
    p = tmp_path / "openai.jsonl"
    p.write_text(line * 3, encoding="utf-8")
    with pytest.warns(UserWarning, match="model argument is ignored") as record:
        parsed = SferenceClient._parse_jsonl(p, model="extra")
    assert len(record) == 1
    assert len(parsed) == 3


def test_parse_content_only_jsonl_format(tmp_path: Path) -> None:
    p = tmp_path / "content.jsonl"
    p.write_text('{"content":"What is the capital of France?"}\n', encoding="utf-8")
    parsed = SferenceClient._parse_jsonl(p, model="Qwen/Qwen3.5-4B")
    assert len(parsed) == 1
    assert parsed[0]["body"]["model"] == "Qwen/Qwen3.5-4B"
    assert parsed[0]["body"]["messages"][0]["content"] == "What is the capital of France?"


def test_parse_content_only_requires_model(tmp_path: Path) -> None:
    p = tmp_path / "content.jsonl"
    p.write_text('{"content":"x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="model is required"):
        SferenceClient._parse_jsonl(p, model=None)


def test_login_and_get_me_via_httpx_mock_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/auth/login":
            payload = json.loads((FIXTURES / "V1AuthLoginLogin" / "200.json").read_text(encoding="utf-8"))
            return httpx.Response(
                status_code=200,
                json=payload,
            )
        if request.method == "GET" and request.url.path == "/v1/auth/me":
            assert request.headers.get("authorization") == "Bearer mock-admin-token"
            payload = json.loads((FIXTURES / "V1AuthMeMe" / "200.json").read_text(encoding="utf-8"))
            return httpx.Response(status_code=200, json=payload)
        return httpx.Response(status_code=404, json={"detail": "not found"})

    with SferenceClient(transport=httpx.MockTransport(handler)) as client:
        login = client.login("admin", "admin")
        assert login.access_token == "mock-admin-token"
        me = client.get_me()
        assert me["username"] == "admin"


def test_list_batches_via_httpx_mock_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/batches":
            assert request.headers.get("authorization") == "Bearer tok"
            payload = json.loads((FIXTURES / "V1BatchesListBatches" / "200.json").read_text(encoding="utf-8"))
            return httpx.Response(
                status_code=200,
                json=payload,
            )
        return httpx.Response(status_code=404, json={"detail": "not found"})

    with SferenceClient(transport=httpx.MockTransport(handler), api_key="tok") as client:
        out = client.list_batches()
        assert len(out.items) == 1
        assert out.items[0].id == "batch_abc"
        assert out.items[0].request_count == 2


def test_submit_batch_raises_api_error_on_http_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/batches":
            return httpx.Response(status_code=401, json={"detail": "Unauthorized"})
        return httpx.Response(status_code=404, json={"detail": "not found"})

    with SferenceClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ApiError, match="401"):
            client.submit_batch(
                requests=[
                    {
                        "custom_id": "request-1",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {"model": "Qwen/Qwen3.5-4B", "messages": [{"role": "user", "content": "hi"}]},
                    }
                ]
            )


def test_cancel_batch_via_httpx_mock_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/batches/batch_abc/cancel":
            assert request.headers.get("authorization") == "Bearer tok"
            payload = json.loads(
                (FIXTURES / "V1BatchesBatchIdCancelCancelBatch" / "200.json").read_text(encoding="utf-8")
            )
            return httpx.Response(
                status_code=200,
                json=payload,
            )
        return httpx.Response(status_code=404, json={"detail": "not found"})

    with SferenceClient(transport=httpx.MockTransport(handler), api_key="tok") as client:
        out = client.cancel_batch("batch_abc")
        assert out.status == "cancelled"


def test_download_results_jsonl_via_httpx_mock_transport(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/batches/batch_abc/results.jsonl":
            assert request.headers.get("authorization") == "Bearer tok"
            return httpx.Response(status_code=200, content=b'{"x": 1}\n{"x": 2}\n')
        return httpx.Response(status_code=404, json={"detail": "not found"})

    out_path = tmp_path / "out.jsonl"
    with SferenceClient(transport=httpx.MockTransport(handler), api_key="tok") as client:
        client.download_results_jsonl("batch_abc", out_path)
    assert out_path.read_text(encoding="utf-8").splitlines() == ['{"x": 1}', '{"x": 2}']


def test_create_and_list_streams_via_httpx_mock_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/streams":
            payload = json.loads((FIXTURES / "V1StreamsCreateStream" / "201.json").read_text(encoding="utf-8"))
            return httpx.Response(status_code=201, json=payload)
        if request.method == "GET" and request.url.path == "/v1/streams":
            payload = json.loads((FIXTURES / "V1StreamsListStreams" / "200.json").read_text(encoding="utf-8"))
            return httpx.Response(status_code=200, json=payload)
        return httpx.Response(status_code=404, json={"detail": "not found"})

    with SferenceClient(transport=httpx.MockTransport(handler), api_key="tok") as client:
        st = client.create_stream("x", window="24h")
        assert st.name == "daily-extraction"
        lst = client.list_streams()
        assert len(lst.items) == 1


def test_list_stream_events_passes_query_params() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events"):
            captured.update(dict(request.url.params))
            payload = json.loads(
                (FIXTURES / "V1StreamsStreamIdEventsListEvents" / "200.json").read_text(encoding="utf-8")
            )
            return httpx.Response(status_code=200, json=payload)
        return httpx.Response(status_code=404, json={"detail": "not found"})

    with SferenceClient(transport=httpx.MockTransport(handler), api_key="tok") as client:
        page = client.list_stream_events(
            "123e4567-e89b-12d3-a456-426614174000",
            limit=10,
            starting_after="019d58a7-2ece-7742-bc3e-69ba44168279",
            wait_ms=1000,
        )
    assert captured.get("limit") == "10"
    assert captured.get("starting_after") == "019d58a7-2ece-7742-bc3e-69ba44168279"
    assert captured.get("wait_ms") == "1000"
    assert page.data[0].completion_id == "019d58a7-2ece-7742-bc3e-69ba44168279"


def test_checkpoint_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from sference_sdk.checkpoint import clear_checkpoint, load_checkpoint, save_checkpoint

    monkeypatch.setenv("SFERENCE_STREAM_CHECKPOINTS", str(tmp_path / "cp.json"))
    assert load_checkpoint("http://127.0.0.1:8000", "sid", "c1") is None
    save_checkpoint("http://127.0.0.1:8000", "sid", "c1", "e1")
    assert load_checkpoint("http://127.0.0.1:8000", "sid", "c1") == "e1"
    clear_checkpoint("http://127.0.0.1:8000", "sid", "c1")
    assert load_checkpoint("http://127.0.0.1:8000", "sid", "c1") is None

