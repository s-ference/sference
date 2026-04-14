from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from sference_sdk.async_client import AsyncSferenceClient
from sference_sdk.client import ApiError

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.mark.asyncio
async def test_async_get_me_via_mock_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/auth/me":
            assert request.headers.get("authorization") == "Bearer mock-admin-token"
            payload = json.loads((FIXTURES / "V1AuthMeMe" / "200.json").read_text(encoding="utf-8"))
            return httpx.Response(status_code=200, json=payload)
        return httpx.Response(status_code=404, json={"detail": "not found"})

    async with AsyncSferenceClient(transport=httpx.MockTransport(handler), api_key="mock-admin-token") as client:
        me = await client.get_me()
        assert me["username"] == "admin"


@pytest.mark.asyncio
async def test_async_list_batches_via_mock_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/batches":
            assert request.headers.get("authorization") == "Bearer tok"
            payload = json.loads((FIXTURES / "V1BatchesListBatches" / "200.json").read_text(encoding="utf-8"))
            return httpx.Response(status_code=200, json=payload)
        return httpx.Response(status_code=404, json={"detail": "not found"})

    async with AsyncSferenceClient(transport=httpx.MockTransport(handler), api_key="tok") as client:
        out = await client.list_batches()
        assert len(out.items) == 1
        assert out.items[0].id == "batch_abc"


@pytest.mark.asyncio
async def test_async_submit_batch_raises_api_error_on_http_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/batches":
            return httpx.Response(status_code=401, json={"detail": "Unauthorized"})
        return httpx.Response(status_code=404, json={"detail": "not found"})

    async with AsyncSferenceClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ApiError, match="401"):
            await client.submit_batch(
                requests=[
                    {
                        "custom_id": "request-1",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {"model": "Qwen/Qwen3.5-4B", "messages": [{"role": "user", "content": "hi"}]},
                    }
                ]
            )


@pytest.mark.asyncio
async def test_async_cancel_batch_via_mock_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/batches/batch_abc/cancel":
            assert request.headers.get("authorization") == "Bearer tok"
            payload = json.loads(
                (FIXTURES / "V1BatchesBatchIdCancelCancelBatch" / "200.json").read_text(encoding="utf-8")
            )
            return httpx.Response(status_code=200, json=payload)
        return httpx.Response(status_code=404, json={"detail": "not found"})

    async with AsyncSferenceClient(transport=httpx.MockTransport(handler), api_key="tok") as client:
        out = await client.cancel_batch("batch_abc")
        assert out.status == "cancelled"


@pytest.mark.asyncio
async def test_async_download_results_jsonl_via_mock_transport(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/batches/batch_abc/results.jsonl":
            assert request.headers.get("authorization") == "Bearer tok"
            return httpx.Response(status_code=200, content=b'{"x": 1}\n{"x": 2}\n')
        return httpx.Response(status_code=404, json={"detail": "not found"})

    out_path = tmp_path / "out.jsonl"
    async with AsyncSferenceClient(transport=httpx.MockTransport(handler), api_key="tok") as client:
        await client.download_results_jsonl("batch_abc", out_path)
    assert out_path.read_text(encoding="utf-8").splitlines() == ['{"x": 1}', '{"x": 2}']


def _minimal_batch_payload(*, batch_id: str, status: str) -> dict:
    return {
        "id": batch_id,
        "status": status,
        "window": "24h",
        "request_count": 1,
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-01T00:00:00+00:00",
        "completed_at": None,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
    }


@pytest.mark.asyncio
async def test_async_wait_for_completion_via_mock_transport() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/batches/batch_wait":
            calls.append("get")
            if len(calls) < 3:
                return httpx.Response(
                    status_code=200,
                    json=_minimal_batch_payload(batch_id="batch_wait", status="pending"),
                )
            return httpx.Response(
                status_code=200,
                json=_minimal_batch_payload(batch_id="batch_wait", status="completed"),
            )
        return httpx.Response(status_code=404, json={"detail": "not found"})

    async with AsyncSferenceClient(transport=httpx.MockTransport(handler), api_key="tok") as client:
        batch = await client.wait_for_completion("batch_wait", poll_interval=0.01, timeout=5.0)
    assert batch.status == "completed"
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_async_get_results_via_mock_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/batches/batch_1/results":
            payload = {
                "batch_id": "batch_1",
                "status": "completed",
                "output_url": None,
                "completed_at": "2025-01-01T00:00:01+00:00",
                "results": None,
            }
            return httpx.Response(status_code=200, json=payload)
        return httpx.Response(status_code=404, json={"detail": "not found"})

    async with AsyncSferenceClient(transport=httpx.MockTransport(handler), api_key="tok") as client:
        res = await client.get_results("batch_1")
        assert res.batch_id == "batch_1"
