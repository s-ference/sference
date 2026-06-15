import json
import warnings
from pathlib import Path

import httpx
import pytest

import sference_sdk
from sference_sdk.client import ApiError, SferenceClient

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_sdk_version_matches_metadata() -> None:
    from importlib.metadata import version

    assert sference_sdk.__version__ == version("sference-sdk")
    assert len(sference_sdk.__version__.split(".")) >= 2


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


def test_client_timeout_defaults_to_30s_and_can_be_overridden() -> None:
    with SferenceClient(api_key="tok") as default_client:
        assert default_client._client.timeout.read == 30.0

    with SferenceClient(api_key="tok", timeout=600.0) as long_client:
        assert long_client._client.timeout.read == 600.0


def test_get_me_via_httpx_mock_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/auth/me":
            assert request.headers.get("authorization") == "Bearer mock-admin-token"
            payload = json.loads((FIXTURES / "V1AuthMeMe" / "200.json").read_text(encoding="utf-8"))
            return httpx.Response(status_code=200, json=payload)
        return httpx.Response(status_code=404, json={"detail": "not found"})

    with SferenceClient(transport=httpx.MockTransport(handler), api_key="mock-admin-token") as client:
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


def test_submit_batch_rejects_unknown_window() -> None:
    with SferenceClient(transport=httpx.MockTransport(lambda r: httpx.Response(500)), api_key="tok") as client:
        with pytest.raises(ValueError, match='"15m", "1h", "24h"'):
            client.submit_batch(
                requests=[{"custom_id": "r1", "body": {"model": "m", "messages": []}}],
                window="2h",
            )


def test_submit_batch_sends_selected_window() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/batches":
            body = json.loads(request.content.decode("utf-8"))
            seen["window"] = body["window"]
            payload = json.loads(
                (FIXTURES / "V1BatchesCreateBatch" / "201.json").read_text(encoding="utf-8")
            )
            payload["window"] = body["window"]
            return httpx.Response(status_code=201, json=payload)
        return httpx.Response(status_code=404, json={"detail": "not found"})

    with SferenceClient(transport=httpx.MockTransport(handler), api_key="tok") as client:
        batch = client.submit_batch(
            requests=[{"custom_id": "r1", "body": {"model": "m", "messages": []}}],
            window="1h",
        )
        assert seen["window"] == "1h"
        assert batch.window == "1h"


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


def test_list_responses_events_passes_query_params() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/responses/events":
            captured.update(dict(request.url.params))
            payload = json.loads(
                (FIXTURES / "V1ResponsesEventsListResponseEvents" / "200.json").read_text(encoding="utf-8")
            )
            return httpx.Response(status_code=200, json=payload)
        return httpx.Response(status_code=404, json={"detail": "not found"})

    with SferenceClient(transport=httpx.MockTransport(handler), api_key="tok") as client:
        page = client.list_responses_events(
            stream_id="123e4567-e89b-12d3-a456-426614174000",
            limit=10,
            starting_after="019d58a7-2ece-7742-bc3e-69ba44168279",
            wait_ms=1000,
        )
    assert captured.get("limit") == "10"
    assert captured.get("stream_id") == "123e4567-e89b-12d3-a456-426614174000"
    assert captured.get("starting_after") == "019d58a7-2ece-7742-bc3e-69ba44168279"
    assert captured.get("wait_ms") == "1000"
    assert page.data[0].completion_id == "019d58a7-2ece-7742-bc3e-69ba44168279"


def test_get_response_parses_reasoning_output_item() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/responses/resp_1":
            return httpx.Response(
                status_code=200,
                json={
                    "id": "resp_1",
                    "object": "response",
                    "created_at": 1712345678,
                    "model": "Qwen/Qwen3.6-35B-A3B",
                    "status": "completed",
                    "output": [
                        {"type": "reasoning", "text": "internal reasoning", "format": "think_tag"},
                        {
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {"type": "output_text", "text": "The answer is 42.", "annotations": []}
                            ],
                        },
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
                },
            )
        return httpx.Response(status_code=404, json={"detail": "not found"})

    with SferenceClient(transport=httpx.MockTransport(handler), api_key="tok") as client:
        resp = client.get_response("resp_1")
        assert resp.status == "completed"
        assert resp.output is not None
        assert resp.output[0].type == "reasoning"
        assert resp.output[0].text == "internal reasoning"
        assert resp.output[1].type == "message"
        assert resp.output[1].content[0].text == "The answer is 42."


def test_get_response_parses_qwen_preamble_cot_reasoning_format() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/responses/resp_qwen":
            return httpx.Response(
                status_code=200,
                json={
                    "id": "resp_qwen",
                    "object": "response",
                    "created_at": 1712345678,
                    "model": "Qwen/Qwen3.6-35B-A3B",
                    "status": "completed",
                    "output": [
                        {
                            "type": "reasoning",
                            "text": "chain",
                            "format": "qwen_preamble_cot",
                        },
                        {
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                        },
                    ],
                },
            )
        return httpx.Response(status_code=404, json={"detail": "not found"})

    with SferenceClient(transport=httpx.MockTransport(handler), api_key="tok") as client:
        resp = client.get_response("resp_qwen")
        assert resp.output is not None
        assert resp.output[0].format == "qwen_preamble_cot"


def test_create_response_sends_include_reasoning_and_parses_reasoning() -> None:
    captured_json: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_json
        if request.method == "POST" and request.url.path == "/v1/responses":
            captured_json = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                status_code=201,
                json={
                    "id": "resp_2",
                    "object": "response",
                    "created_at": 1712345678,
                    "model": captured_json.get("model", "m"),
                    "status": "completed",
                    "output": [
                        {"type": "reasoning", "text": "r", "format": "provider_field"},
                        {
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "t", "annotations": []}],
                        },
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
                },
            )
        return httpx.Response(status_code=404, json={"detail": "not found"})

    with SferenceClient(transport=httpx.MockTransport(handler), api_key="tok") as client:
        resp = client.create_response(
            model="m",
            input=[{"role": "user", "content": "hi"}],
            include_reasoning=False,
        )

    assert captured_json.get("include_reasoning") is False
    assert resp.output is not None
    assert resp.output[0].type == "reasoning"


def test_create_response_sends_enable_thinking_when_set() -> None:
    captured_json: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_json
        if request.method == "POST" and request.url.path == "/v1/responses":
            captured_json = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                status_code=201,
                json={
                    "id": "resp_et",
                    "object": "response",
                    "created_at": 1712345678,
                    "model": "q",
                    "status": "in_progress",
                    "output": None,
                    "error": None,
                    "usage": None,
                },
            )
        return httpx.Response(status_code=404, json={"detail": "not found"})

    with SferenceClient(transport=httpx.MockTransport(handler), api_key="tok") as client:
        client.create_response(
            model="q",
            input=[{"role": "user", "content": "hi"}],
            enable_thinking=True,
        )

    assert captured_json.get("enable_thinking") is True


def test_create_response_sends_string_input_verbatim() -> None:
    """A bare string input is forwarded as-is; the server expands it to a user message."""
    captured_json: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_json
        if request.method == "POST" and request.url.path == "/v1/responses":
            captured_json = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                status_code=201,
                json={
                    "id": "resp_str",
                    "object": "response",
                    "created_at": 1712345678,
                    "model": "m",
                    "status": "in_progress",
                    "output": None,
                },
            )
        return httpx.Response(status_code=404, json={"detail": "not found"})

    with SferenceClient(transport=httpx.MockTransport(handler), api_key="tok") as client:
        client.create_response(model="m", input="grade this answer")

    assert captured_json["input"] == "grade this answer"


def test_checkpoint_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from sference_sdk.checkpoint import clear_checkpoint, load_checkpoint, save_checkpoint

    monkeypatch.setenv("SFERENCE_STREAM_CHECKPOINTS", str(tmp_path / "cp.json"))
    assert load_checkpoint("http://127.0.0.1:8000", "sid", "c1") is None
    save_checkpoint("http://127.0.0.1:8000", "sid", "c1", "e1")
    assert load_checkpoint("http://127.0.0.1:8000", "sid", "c1") == "e1"
    clear_checkpoint("http://127.0.0.1:8000", "sid", "c1")
    assert load_checkpoint("http://127.0.0.1:8000", "sid", "c1") is None


def _completion_event_row(completion_id: str) -> dict:
    return {
        "completion_id": completion_id,
        "response_id": None,
        "custom_id": None,
        "status": "completed",
        "result": {},
        "error": None,
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
        "completed_at": "2026-04-04T10:12:09+00:00",
    }


def test_iter_responses_events_uses_starting_after_from_saved_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from sference_sdk.checkpoint import save_checkpoint

    monkeypatch.setenv("SFERENCE_STREAM_CHECKPOINTS", str(tmp_path / "cp.json"))
    base = "http://iter-checkpoint.local"
    saved_cursor = "019d58a7-2ece-7742-bc3e-69ba44168279"
    next_id = "019d58a7-2ece-7742-bc3e-69ba44168280"
    save_checkpoint(base, "__responses_events__", "consumer-a", saved_cursor)

    starting_after_log: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/v1/responses/events":
            return httpx.Response(status_code=404, json={"detail": "not found"})
        params = dict(request.url.params)
        starting_after_log.append(params.get("starting_after"))
        assert params.get("starting_after") == saved_cursor
        body = {
            "object": "list",
            "data": [_completion_event_row(next_id)],
            "has_more": False,
        }
        return httpx.Response(status_code=200, json=body)

    with SferenceClient(transport=httpx.MockTransport(handler), api_key="tok", base_url=base) as client:
        events = list(client.iter_responses_events(consumer_name="consumer-a", checkpoint=True))
    assert len(events) == 1
    assert events[0].completion_id == next_id
    assert starting_after_log == [saved_cursor]


def test_iter_responses_events_multipage_updates_checkpoint_to_last_completion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from sference_sdk.checkpoint import load_checkpoint, save_checkpoint

    monkeypatch.setenv("SFERENCE_STREAM_CHECKPOINTS", str(tmp_path / "cp.json"))
    base = "http://iter-pages.local"
    c0 = "019d58a7-2ece-7742-bc3e-69ba44168277"
    c1 = "019d58a7-2ece-7742-bc3e-69ba44168278"
    c2 = "019d58a7-2ece-7742-bc3e-69ba44168279"
    save_checkpoint(base, "__responses_events__", "tail", c0)

    starting_after_log: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/v1/responses/events":
            return httpx.Response(status_code=404, json={"detail": "not found"})
        params = dict(request.url.params)
        starting_after_log.append(params.get("starting_after"))
        sa = params.get("starting_after")
        if sa == c0:
            body = {"object": "list", "data": [_completion_event_row(c1)], "has_more": True}
        elif sa == c1:
            body = {"object": "list", "data": [_completion_event_row(c2)], "has_more": False}
        else:
            return httpx.Response(status_code=500, json={"detail": f"unexpected starting_after={sa!r}"})
        return httpx.Response(status_code=200, json=body)

    with SferenceClient(transport=httpx.MockTransport(handler), api_key="tok", base_url=base) as client:
        events = list(client.iter_responses_events(consumer_name="tail", checkpoint=True))
    assert [e.completion_id for e in events] == [c1, c2]
    assert starting_after_log == [c0, c1]
    assert load_checkpoint(base, "__responses_events__", "tail") == c2


async def test_iter_responses_events_async_same_multipage_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from sference_sdk.async_client import AsyncSferenceClient
    from sference_sdk.checkpoint import load_checkpoint, save_checkpoint

    monkeypatch.setenv("SFERENCE_STREAM_CHECKPOINTS", str(tmp_path / "cp-async.json"))
    base = "http://iter-async.local"
    c0 = "019d58a7-2ece-7742-bc3e-69ba44168281"
    c1 = "019d58a7-2ece-7742-bc3e-69ba44168282"
    c2 = "019d58a7-2ece-7742-bc3e-69ba44168283"
    save_checkpoint(base, "__responses_events__", "async-c", c0)

    starting_after_log: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/v1/responses/events":
            return httpx.Response(status_code=404, json={"detail": "not found"})
        params = dict(request.url.params)
        starting_after_log.append(params.get("starting_after"))
        sa = params.get("starting_after")
        if sa == c0:
            body = {"object": "list", "data": [_completion_event_row(c1)], "has_more": True}
        elif sa == c1:
            body = {"object": "list", "data": [_completion_event_row(c2)], "has_more": False}
        else:
            return httpx.Response(status_code=500, json={"detail": f"unexpected starting_after={sa!r}"})
        return httpx.Response(status_code=200, json=body)

    async with AsyncSferenceClient(transport=httpx.MockTransport(handler), api_key="tok", base_url=base) as client:
        out: list[str] = []
        async for ev in client.iter_responses_events(consumer_name="async-c", checkpoint=True):
            out.append(ev.completion_id)
    assert out == [c1, c2]
    assert starting_after_log == [c0, c1]
    assert load_checkpoint(base, "__responses_events__", "async-c") == c2

