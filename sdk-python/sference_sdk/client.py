from __future__ import annotations

import time
from collections.abc import Iterator, Mapping, Sequence
from itertools import chain
from pathlib import Path
from typing import Any, BinaryIO, Self, TypeVar

import httpx
from pydantic import BaseModel

from ._base import (
    BLOCKING_RESPONSE_READ_TIMEOUT,
    DECISION_READ_TIMEOUT,
    DEFAULT_TIMEOUT,
    EVENTS_CHECKPOINT_KEY,
    STREAM_READ_TIMEOUT,
    TERMINAL_BATCH_STATUSES,
    TERMINAL_RESPONSE_STATUSES,
    ApiError,
    BaseClient,
    SSEDecoder,
    parse_stream_event,
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
    ResponseStreamEvent,
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

__all__ = ["ApiError", "SferenceClient"]

_M = TypeVar("_M", bound=BaseModel)


class SferenceClient(BaseClient):
    """Sync HTTP client for the sference API (``httpx.Client``)."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float | None = None,
    ) -> None:
        super().__init__(base_url, api_key, timeout)
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=DEFAULT_TIMEOUT if timeout is None else timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _send(
        self,
        method: str,
        path: str,
        json_body: Any = None,
        *,
        params: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: Any = httpx.USE_CLIENT_DEFAULT,
    ) -> httpx.Response:
        response = self._client.request(
            method, path, headers=self._headers(extra_headers), json=json_body, params=params, timeout=timeout
        )
        raise_for_status(response)
        return response

    def _call(
        self,
        model: type[_M],
        method: str,
        path: str,
        json_body: Any = None,
        *,
        params: Mapping[str, Any] | None = None,
        extra_headers: Mapping[str, str] | None = None,
        timeout: Any = httpx.USE_CLIENT_DEFAULT,
    ) -> _M:
        response = self._send(method, path, json_body, params=params, extra_headers=extra_headers, timeout=timeout)
        return model.model_validate(response.json())

    # Account ----------------------------------------------------------------

    def health(self) -> dict[str, str]:
        """Liveness check (GET /v1/health); needs no credential."""
        return self._send("GET", "/v1/health").json()

    def get_me(self) -> User:
        """Owner of the current credential (GET /v1/auth/me)."""
        return self._call(User, "GET", "/v1/auth/me")

    def list_models(self) -> ModelList:
        """Live model catalog (GET /v1/models)."""
        return self._call(ModelList, "GET", "/v1/models")

    def get_billing_balance(self) -> BillingBalanceResponse:
        """Credit balance in micro-USD (GET /v1/billing/balance)."""
        return self._call(BillingBalanceResponse, "GET", "/v1/billing/balance")

    def get_billing_summary(self, *, days: int | None = None) -> BillingSummaryResponse:
        """Spend over the last ``days`` (1-365, server default 30) (GET /v1/billing/summary)."""
        return self._call(BillingSummaryResponse, "GET", "/v1/billing/summary", params=self._params(days=days))

    def list_billing_transactions(
        self, *, limit: int | None = None, starting_after: str | None = None
    ) -> LedgerListResponse:
        """Ledger entries, newest first (GET /v1/billing/transactions).

        Page with ``starting_after=page.next_starting_after``.
        """
        params = self._params(limit=limit, starting_after=starting_after)
        return self._call(LedgerListResponse, "GET", "/v1/billing/transactions", params=params)

    def get_usage_summary(self, *, days: int | None = None, user_id: str | None = None) -> UsageSummaryResponse:
        """Token and cost totals (GET /v1/usage/summary); ``user_id`` is for team admins."""
        params = self._params(days=days, user_id=user_id)
        return self._call(UsageSummaryResponse, "GET", "/v1/usage/summary", params=params)

    def get_usage_daily(self, *, days: int | None = None, user_id: str | None = None) -> UsageDailyResponse:
        """Per-day, per-model usage (GET /v1/usage/daily)."""
        params = self._params(days=days, user_id=user_id)
        return self._call(UsageDailyResponse, "GET", "/v1/usage/daily", params=params)

    def get_usage_by_key(
        self, *, days: int | None = None, month: str | None = None, tag: str | None = None
    ) -> UsageByKeyResponse:
        """Usage per API key (GET /v1/usage/by-key); ``month`` (YYYY-MM) overrides ``days``."""
        params = self._params(days=days, month=month, tag=tag)
        return self._call(UsageByKeyResponse, "GET", "/v1/usage/by-key", params=params)

    # Embeddings and decisions --------------------------------------------------

    def create_embeddings(
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
        return self._call(EmbeddingResponse, "POST", "/v1/embeddings", payload)

    def create_decision(
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
        return self._call(
            DecisionResponse, "POST", "/v1/decisions", payload, timeout=self._quiet_timeout(DECISION_READ_TIMEOUT)
        )

    # Batches ------------------------------------------------------------------

    def submit_batch(
        self,
        *,
        input_file: str | None = None,
        requests: Sequence[InferenceRequest | Mapping[str, Any]] | None = None,
        model: str | None = None,
        window: str = "24h",
    ) -> Batch:
        """Submit a batch from a JSONL file or a list of requests (POST /v1/batches)."""
        payload = self._batch_payload(input_file=input_file, requests=requests, model=model, window=window)
        return self._call(Batch, "POST", "/v1/batches", payload)

    def get_batch(self, batch_id: str) -> Batch:
        return self._call(Batch, "GET", f"/v1/batches/{batch_id}")

    def list_batches(self) -> BatchList:
        return self._call(BatchList, "GET", "/v1/batches")

    def get_results(self, batch_id: str) -> BatchResults:
        return self._call(BatchResults, "GET", f"/v1/batches/{batch_id}/results")

    def get_results_indexed(self, batch_id: str) -> dict[str, BatchResultRow]:
        """Fetch batch results keyed by ``custom_id``."""
        return self.get_results(batch_id).index_by_custom_id()

    def cancel_batch(self, batch_id: str) -> Batch:
        return self._call(Batch, "POST", f"/v1/batches/{batch_id}/cancel")

    def download_results_jsonl(self, batch_id: str, out: str | Path | BinaryIO) -> None:
        content = self._send("GET", f"/v1/batches/{batch_id}/results.jsonl").content
        if isinstance(out, (str, Path)):
            Path(out).write_bytes(content)
        else:
            out.write(content)

    def wait_for_completion(
        self, batch_id: str, poll_interval: float = 5.0, timeout: float | None = None
    ) -> Batch:
        """Poll until the batch is terminal; ``timeout=None`` waits as long as it takes."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            batch = self.get_batch(batch_id)
            if batch.status in TERMINAL_BATCH_STATUSES:
                return batch
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for batch {batch_id}")
            time.sleep(poll_interval)

    # Streams ------------------------------------------------------------------

    def create_stream(self, name: str, window: StreamWindow = "24h") -> Stream:
        return self._call(Stream, "POST", "/v1/streams", {"name": name, "window": window})

    def list_streams(self) -> StreamList:
        return self._call(StreamList, "GET", "/v1/streams")

    def get_stream(self, stream_id: str) -> Stream:
        return self._call(Stream, "GET", f"/v1/streams/{stream_id}")

    def archive_stream(self, stream_id: str) -> Stream:
        return self._call(Stream, "PATCH", f"/v1/streams/{stream_id}", {"status": "archived"})

    def cancel_stream(self, stream_id: str) -> Stream:
        return self._call(Stream, "PATCH", f"/v1/streams/{stream_id}", {"status": "cancelled"})

    # Responses ----------------------------------------------------------------

    def create_response(
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
        Use ``stream_response`` to receive output as it is generated.
        """
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
        return self._call(
            Response,
            "POST",
            "/v1/responses",
            payload,
            extra_headers=headers,
            # Background creates return at once; blocking ones wait for the whole generation.
            timeout=httpx.USE_CLIENT_DEFAULT if background else self._quiet_timeout(BLOCKING_RESPONSE_READ_TIMEOUT),
        )

    def stream_response(
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
        metadata: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> Iterator[ResponseStreamEvent]:
        """Create a response and yield its server-sent events (POST /v1/responses, ``stream=true``).

        Text arrives as ``response.output_text.delta`` events; the last event is
        ``response.completed``, ``response.incomplete`` or ``response.failed`` and
        carries the full ``Response``. A mid-stream server error raises ``ApiError``.
        Closing the iterator early disconnects, which cancels the generation.
        """
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
            background=False,
            metadata=metadata,
            idempotency_key=idempotency_key,
        )
        payload["stream"] = True
        # Build the request now, not on first iteration, so ``extra_headers()`` applies
        # where the call is made and bad input fails before any network I/O.
        request = self._client.build_request(
            "POST",
            "/v1/responses",
            headers=self._headers(headers),
            json=payload,
            timeout=self._quiet_timeout(STREAM_READ_TIMEOUT),
        )
        return self._iter_stream(request)

    def _iter_stream(self, request: httpx.Request) -> Iterator[ResponseStreamEvent]:
        response = self._client.send(request, stream=True)
        try:
            if response.is_error:
                response.read()
                raise_for_status(response)
            decoder = SSEDecoder()
            # The trailing "" flushes a final event the server didn't end with a blank line.
            for line in chain(response.iter_lines(), [""]):
                data = decoder.feed(line)
                if data is None:
                    continue
                event = parse_stream_event(data)
                if event is None:
                    return
                yield event
        finally:
            response.close()

    def wait_for_response(self, response_id: str, poll_interval: float = 2.0, timeout: float | None = 3600.0) -> Response:
        """Poll GET /v1/responses/{id} until terminal; ``timeout=None`` waits indefinitely."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            item = self.get_response(response_id)
            if item.status in TERMINAL_RESPONSE_STATUSES:
                return item
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for response {response_id}")
            time.sleep(poll_interval)

    def get_response(self, response_id: str) -> Response:
        return self._call(Response, "GET", f"/v1/responses/{response_id}")

    def cancel_response(self, response_id: str) -> Response:
        """Cancel a pending or running response."""
        return self._call(Response, "DELETE", f"/v1/responses/{response_id}")

    def list_responses(self, *, limit: int = 100, stream_id: str | None = None) -> ResponseList:
        """List responses, optionally only those in one stream."""
        params = self._params(limit=limit, stream_id=stream_id)
        return self._call(ResponseList, "GET", "/v1/responses", params=params)

    def list_responses_events(
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
        return self._call(StreamEventList, "GET", "/v1/responses/events", params=params)

    def iter_responses_events(
        self,
        *,
        stream_id: str | None = None,
        consumer_name: str = "default",
        starting_after: str | None = None,
        checkpoint: bool = True,
        from_latest: bool = False,
    ) -> Iterator[StreamInferenceCompletionEvent]:
        """Yield completion events oldest first, resuming from a saved checkpoint."""
        ck_stream = stream_id if stream_id is not None else EVENTS_CHECKPOINT_KEY
        if from_latest:
            clear_checkpoint(self.base_url, ck_stream, consumer_name)
        cur: str | None = starting_after
        if checkpoint and cur is None:
            cur = load_checkpoint(self.base_url, ck_stream, consumer_name)

        if cur is None:
            # No cursor yet: the API returns newest first, so replay one page in order.
            page = self.list_responses_events(stream_id=stream_id, limit=100)
            for ev in reversed(page.data):
                yield ev
                if checkpoint:
                    save_checkpoint(self.base_url, ck_stream, consumer_name, ev.completion_id)
            return

        while True:
            page = self.list_responses_events(stream_id=stream_id, limit=100, starting_after=cur)
            if not page.data:
                return
            for ev in page.data:
                yield ev
                if checkpoint:
                    save_checkpoint(self.base_url, ck_stream, consumer_name, ev.completion_id)
                cur = ev.completion_id
            if not page.has_more:
                return
