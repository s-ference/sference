from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO, Self, TypeVar

import httpx
from pydantic import BaseModel

from ._base import (
    DEFAULT_TIMEOUT,
    EVENTS_CHECKPOINT_KEY,
    TERMINAL_BATCH_STATUSES,
    TERMINAL_RESPONSE_STATUSES,
    BaseClient,
    raise_for_status,
)
from .checkpoint import clear_checkpoint, load_checkpoint, save_checkpoint
from .models import (
    Batch,
    BatchList,
    BatchResultRow,
    BatchResults,
    BillingBalanceResponse,
    BillingSummaryResponse,
    DecisionImage,
    DecisionQuestion,
    DecisionResponse,
    EmbeddingEncodingFormat,
    EmbeddingInput,
    EmbeddingResponse,
    InferenceRequest,
    LedgerListResponse,
    ModelList,
    Response,
    ResponseList,
    ServiceTier,
    Stream,
    StreamEventList,
    StreamInferenceCompletionEvent,
    StreamList,
    StreamWindow,
    UsageByKeyResponse,
    UsageDailyResponse,
    UsageSummaryResponse,
    User,
)

__all__ = ["AsyncSferenceClient"]

_M = TypeVar("_M", bound=BaseModel)


class AsyncSferenceClient(BaseClient):
    """Async HTTP client for the sference API (``httpx.AsyncClient``)."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float | None = None,
    ) -> None:
        super().__init__(base_url, api_key)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=DEFAULT_TIMEOUT if timeout is None else timeout,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def _send(
        self,
        method: str,
        path: str,
        json_body: Any = None,
        *,
        params: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        response = await self._client.request(
            method, path, headers=self._headers(extra_headers), json=json_body, params=params
        )
        raise_for_status(response)
        return response

    async def _call(
        self,
        model: type[_M],
        method: str,
        path: str,
        json_body: Any = None,
        *,
        params: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> _M:
        response = await self._send(method, path, json_body, params=params, extra_headers=extra_headers)
        return model.model_validate(response.json())

    # Account ----------------------------------------------------------------

    async def health(self) -> dict[str, str]:
        """Liveness check (GET /v1/health); needs no credential."""
        return (await self._send("GET", "/v1/health")).json()

    async def get_me(self) -> User:
        """Owner of the current credential (GET /v1/auth/me)."""
        return await self._call(User, "GET", "/v1/auth/me")

    async def list_models(self) -> ModelList:
        """Live model catalog (GET /v1/models)."""
        return await self._call(ModelList, "GET", "/v1/models")

    async def get_billing_balance(self) -> BillingBalanceResponse:
        """Credit balance in micro-USD (GET /v1/billing/balance)."""
        return await self._call(BillingBalanceResponse, "GET", "/v1/billing/balance")

    async def get_billing_summary(self, *, days: int | None = None) -> BillingSummaryResponse:
        """Spend over the last ``days`` (1-365, server default 30) (GET /v1/billing/summary)."""
        return await self._call(BillingSummaryResponse, "GET", "/v1/billing/summary", params=self._params(days=days))

    async def list_billing_transactions(
        self, *, limit: int | None = None, starting_after: str | None = None
    ) -> LedgerListResponse:
        """Ledger entries, newest first (GET /v1/billing/transactions).

        Page with ``starting_after=page.next_starting_after``.
        """
        params = self._params(limit=limit, starting_after=starting_after)
        return await self._call(LedgerListResponse, "GET", "/v1/billing/transactions", params=params)

    async def get_usage_summary(self, *, days: int | None = None, user_id: str | None = None) -> UsageSummaryResponse:
        """Token and cost totals (GET /v1/usage/summary); ``user_id`` is for team admins."""
        params = self._params(days=days, user_id=user_id)
        return await self._call(UsageSummaryResponse, "GET", "/v1/usage/summary", params=params)

    async def get_usage_daily(self, *, days: int | None = None, user_id: str | None = None) -> UsageDailyResponse:
        """Per-day, per-model usage (GET /v1/usage/daily)."""
        params = self._params(days=days, user_id=user_id)
        return await self._call(UsageDailyResponse, "GET", "/v1/usage/daily", params=params)

    async def get_usage_by_key(
        self, *, days: int | None = None, month: str | None = None, tag: str | None = None
    ) -> UsageByKeyResponse:
        """Usage per API key (GET /v1/usage/by-key); ``month`` (YYYY-MM) overrides ``days``."""
        params = self._params(days=days, month=month, tag=tag)
        return await self._call(UsageByKeyResponse, "GET", "/v1/usage/by-key", params=params)

    # Embeddings and decisions --------------------------------------------------

    async def create_embeddings(
        self,
        *,
        model: str,
        input: EmbeddingInput,
        encoding_format: EmbeddingEncodingFormat = "float",
        dimensions: int | None = None,
        user: str | None = None,
    ) -> EmbeddingResponse:
        """Create embeddings (POST /v1/embeddings)."""
        payload = self._embedding_payload(
            model=model, input=input, encoding_format=encoding_format, dimensions=dimensions, user=user
        )
        return await self._call(EmbeddingResponse, "POST", "/v1/embeddings", payload)

    async def create_decision(
        self,
        *,
        model: str,
        state: Any,
        questions: Mapping[str, DecisionQuestion | Mapping[str, Any]],
        images: Sequence[str | DecisionImage | Mapping[str, Any]] | None = None,
    ) -> DecisionResponse:
        """Answer up to 64 questions about ``state`` in one realtime call (POST /v1/decisions).

        ``state`` is any JSON value (text or an object). Questions may be model
        instances or plain dicts with a ``type`` of ``choice``, ``score`` or ``noul``.
        ``images`` accepts up to four PNG/JPEG/WebP data URLs or objects with
        ``content_type`` and ``base64``. Limits: 4 MiB/16 MP each, 8 MiB total.
        """
        payload = self._decision_payload(model=model, state=state, questions=questions, images=images)
        return await self._call(DecisionResponse, "POST", "/v1/decisions", payload)

    # Batches ------------------------------------------------------------------

    async def submit_batch(
        self,
        *,
        input_file: str | None = None,
        requests: Sequence[InferenceRequest | Mapping[str, Any]] | None = None,
        model: str | None = None,
        window: str = "24h",
    ) -> Batch:
        """Submit a batch from a JSONL file or a list of requests (POST /v1/batches)."""
        payload = self._batch_payload(input_file=input_file, requests=requests, model=model, window=window)
        return await self._call(Batch, "POST", "/v1/batches", payload)

    async def get_batch(self, batch_id: str) -> Batch:
        return await self._call(Batch, "GET", f"/v1/batches/{batch_id}")

    async def list_batches(self) -> BatchList:
        return await self._call(BatchList, "GET", "/v1/batches")

    async def get_results(self, batch_id: str) -> BatchResults:
        return await self._call(BatchResults, "GET", f"/v1/batches/{batch_id}/results")

    async def get_results_indexed(self, batch_id: str) -> dict[str, BatchResultRow]:
        """Fetch batch results keyed by ``custom_id``."""
        return (await self.get_results(batch_id)).index_by_custom_id()

    async def cancel_batch(self, batch_id: str) -> Batch:
        return await self._call(Batch, "POST", f"/v1/batches/{batch_id}/cancel")

    async def download_results_jsonl(self, batch_id: str, out: str | Path | BinaryIO) -> None:
        content = (await self._send("GET", f"/v1/batches/{batch_id}/results.jsonl")).content
        if isinstance(out, (str, Path)):
            Path(out).write_bytes(content)
        else:
            out.write(content)

    async def wait_for_completion(
        self, batch_id: str, poll_interval: float = 5.0, timeout: float | None = None
    ) -> Batch:
        """Poll until the batch is terminal; ``timeout=None`` waits as long as it takes."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            batch = await self.get_batch(batch_id)
            if batch.status in TERMINAL_BATCH_STATUSES:
                return batch
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for batch {batch_id}")
            await asyncio.sleep(poll_interval)

    # Streams ------------------------------------------------------------------

    async def create_stream(self, name: str, window: StreamWindow = "24h") -> Stream:
        return await self._call(Stream, "POST", "/v1/streams", {"name": name, "window": window})

    async def list_streams(self) -> StreamList:
        return await self._call(StreamList, "GET", "/v1/streams")

    async def get_stream(self, stream_id: str) -> Stream:
        return await self._call(Stream, "GET", f"/v1/streams/{stream_id}")

    async def archive_stream(self, stream_id: str) -> Stream:
        return await self._call(Stream, "PATCH", f"/v1/streams/{stream_id}", {"status": "archived"})

    async def cancel_stream(self, stream_id: str) -> Stream:
        return await self._call(Stream, "PATCH", f"/v1/streams/{stream_id}", {"status": "cancelled"})

    # Responses ----------------------------------------------------------------

    async def create_response(
        self,
        *,
        model: str,
        input: str | Sequence[Mapping[str, Any]],
        instructions: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        include_reasoning: bool | None = None,
        enable_thinking: bool | None = None,
        reasoning: Mapping[str, Any] | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        include: Sequence[str] | None = None,
        text: Mapping[str, Any] | None = None,
        service_tier: ServiceTier | None = None,
        prompt_cache_key: str | None = None,
        max_queue_wait_s: int | None = None,
        background: bool = False,
        metadata: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> Response:
        """Create a response (POST /v1/responses).

        ``input`` is a string or a list of OpenAI Responses input items. Pass
        ``background=True`` and poll with ``wait_for_response`` for long jobs.
        Attach to a stream with ``metadata={"stream_id": ..., "completion_window": "24h"}``.
        ``service_tier="flex"`` trades latency for price; ``"priority"`` costs more
        and is scheduled first. ``idempotency_key`` makes retries safe.
        """
        # TODO: support `stream=True` (SSE) once there is a user for it; the SDK
        # targets background work, so it only sends non-streaming requests today.
        payload, headers = self._response_request(
            model=model,
            input=input,
            instructions=instructions,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            include_reasoning=include_reasoning,
            enable_thinking=enable_thinking,
            reasoning=reasoning,
            tools=tools,
            tool_choice=tool_choice,
            include=include,
            text=text,
            service_tier=service_tier,
            prompt_cache_key=prompt_cache_key,
            max_queue_wait_s=max_queue_wait_s,
            background=background,
            metadata=metadata,
            idempotency_key=idempotency_key,
        )
        return await self._call(Response, "POST", "/v1/responses", payload, extra_headers=headers)

    async def wait_for_response(self, response_id: str, poll_interval: float = 2.0, timeout: float | None = 3600.0) -> Response:
        """Poll GET /v1/responses/{id} until terminal; ``timeout=None`` waits indefinitely."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            item = await self.get_response(response_id)
            if item.status in TERMINAL_RESPONSE_STATUSES:
                return item
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for response {response_id}")
            await asyncio.sleep(poll_interval)

    async def get_response(self, response_id: str) -> Response:
        return await self._call(Response, "GET", f"/v1/responses/{response_id}")

    async def cancel_response(self, response_id: str) -> Response:
        """Cancel a pending or running response."""
        return await self._call(Response, "DELETE", f"/v1/responses/{response_id}")

    async def list_responses(self, *, limit: int = 100, stream_id: str | None = None) -> ResponseList:
        """List responses, optionally only those in one stream."""
        params = self._params(limit=limit, stream_id=stream_id)
        return await self._call(ResponseList, "GET", "/v1/responses", params=params)

    async def list_responses_events(
        self,
        *,
        stream_id: str | None = None,
        limit: int = 20,
        starting_after: str | None = None,
        ending_before: str | None = None,
        wait_ms: int = 0,
    ) -> StreamEventList:
        """GET /v1/responses/events — completion events (omit stream_id for non-stream feed)."""
        params = self._params(
            limit=limit,
            wait_ms=wait_ms,
            stream_id=stream_id,
            starting_after=starting_after,
            ending_before=ending_before,
        )
        return await self._call(StreamEventList, "GET", "/v1/responses/events", params=params)

    async def iter_responses_events(
        self,
        *,
        stream_id: str | None = None,
        consumer_name: str = "default",
        starting_after: str | None = None,
        checkpoint: bool = True,
        from_latest: bool = False,
    ) -> AsyncIterator[StreamInferenceCompletionEvent]:
        """Yield completion events oldest first, resuming from a saved checkpoint."""
        ck_stream = stream_id if stream_id is not None else EVENTS_CHECKPOINT_KEY
        if from_latest:
            clear_checkpoint(self.base_url, ck_stream, consumer_name)
        cur: str | None = starting_after
        if checkpoint and cur is None:
            cur = load_checkpoint(self.base_url, ck_stream, consumer_name)

        if cur is None:
            # No cursor yet: the API returns newest first, so replay one page in order.
            page = await self.list_responses_events(stream_id=stream_id, limit=100)
            for ev in reversed(page.data):
                yield ev
                if checkpoint:
                    save_checkpoint(self.base_url, ck_stream, consumer_name, ev.completion_id)
            return

        while True:
            page = await self.list_responses_events(stream_id=stream_id, limit=100, starting_after=cur)
            if not page.data:
                return
            for ev in page.data:
                yield ev
                if checkpoint:
                    save_checkpoint(self.base_url, ck_stream, consumer_name, ev.completion_id)
                cur = ev.completion_id
            if not page.has_more:
                return
