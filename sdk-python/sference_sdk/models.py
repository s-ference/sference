from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


BatchStatus = Literal["pending", "running", "completed", "failed", "cancelled"]

# Completion window accepted by batches, streams, and background responses.
CompletionWindow = Literal["24h"]
COMPLETION_WINDOWS: tuple[str, ...] = ("24h",)


class ApiModel(BaseModel):
    """Base for parsed API responses.

    Unknown fields are kept (``model_extra``) so fields the API adds later are
    readable before the SDK models catch up.
    """

    model_config = ConfigDict(extra="allow")


class InferenceRequest(BaseModel):
    """Unified inference request schema for batches and streams.

    custom_id is user-provided and can be used for correlation.
    """

    model_config = ConfigDict(extra="forbid")

    custom_id: str | None = Field(default=None, description="User-provided identifier for correlation.")
    body: dict[str, Any] = Field(description="OpenAI-style request body. Must include `model`.")

    @classmethod
    def chat(
        cls,
        *,
        custom_id: str,
        user_content: str,
        model: str,
        system_content: str | None = None,
        temperature: float | None = None,
    ) -> InferenceRequest:
        """Build a chat-completions batch row (``body.messages``)."""
        messages: list[dict[str, str]] = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_content})
        body: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            body["temperature"] = temperature
        return cls(custom_id=custom_id, body=body)


class User(ApiModel):
    """Owner of the current credential (GET /v1/auth/me)."""

    username: str
    role: str


class Batch(ApiModel):
    id: str
    status: BatchStatus
    window: CompletionWindow
    request_count: int
    created_at: str
    updated_at: str
    completed_at: str | None = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_micros: int = 0


class BatchList(ApiModel):
    items: list[Batch]


class BatchResultRow(ApiModel):
    """One row from ``GET /v1/batches/{id}/results`` (sference ``result_json`` / ``error_json`` shape)."""

    custom_id: str | None = None
    status: str | None = None
    result_json: dict[str, Any] | None = None
    error_json: dict[str, Any] | None = None

    @property
    def completion_text(self) -> str:
        """Assistant text for a completed row, or a bracketed status/error summary."""
        if self.status != "completed":
            if self.error_json:
                return f"[{self.status}] {self.error_json}"
            return f"[{self.status}]"
        result = self.result_json or {}
        choices = result.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        return content if isinstance(content, str) else str(content or "")


class BatchResults(ApiModel):
    batch_id: str
    status: BatchStatus
    output_url: str | None = None
    completed_at: str | None = None
    results: list[dict[str, Any]] | None = None

    def rows(self) -> list[BatchResultRow]:
        return [BatchResultRow.model_validate(row) for row in self.results or []]

    def index_by_custom_id(self) -> dict[str, BatchResultRow]:
        indexed: dict[str, BatchResultRow] = {}
        for row in self.rows():
            if row.custom_id is not None:
                indexed[str(row.custom_id)] = row
        return indexed


class BatchCreatePayload(BaseModel):
    window: CompletionWindow = "24h"
    requests: list[InferenceRequest] = Field(min_length=1)


StreamStatus = Literal["open", "cancelled", "archived"]
StreamWindow = CompletionWindow


class Stream(ApiModel):
    """Stream detail (GET /v1/streams/{id}); list responses omit counter fields (defaults apply)."""

    id: str
    name: str
    window: StreamWindow
    status: StreamStatus
    created_at: str
    updated_at: str
    cancelled_at: str | None = None
    archived_at: str | None = None
    total_items: int = 0
    pending_items: int = 0
    running_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    cancelled_items: int = 0
    completion_ratio: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_micros: int = 0
    last_completion_at: str | None = None
    latest_completion_id: str | None = None


class StreamInferenceCompletionEvent(ApiModel):
    completion_id: str
    response_id: str | None = None
    custom_id: str | None = None
    status: str
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_micros: int | None = None
    completed_at: str | None = None


class StreamEventList(ApiModel):
    object: str = "list"
    data: list[StreamInferenceCompletionEvent]
    has_more: bool


class StreamList(ApiModel):
    items: list[Stream]


# Response (OpenAI-compatible) models
ResponseStatus = Literal["in_progress", "completed", "failed", "cancelled", "incomplete"]
# Accepted request tiers; the response echoes the tier actually served.
ServiceTier = Literal["auto", "default", "flex", "priority", "scale"]


class ResponseReasoningSummaryPart(ApiModel):
    type: Literal["summary_text"] = "summary_text"
    text: str


class ResponseOutputText(ApiModel):
    """An ``output_text`` content part nested inside a message item."""

    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[Any] = Field(default_factory=list)


class ResponseOutputMessage(ApiModel):
    """Assistant message output item wrapping ``output_text`` content parts.

    Mirrors the OpenAI Responses API, where final text lives in a ``message`` item
    with a nested ``content[]`` of ``output_text`` parts rather than a top-level
    ``output_text`` item.
    """

    type: Literal["message"] = "message"
    id: str
    role: Literal["assistant"] = "assistant"
    status: Literal["completed", "in_progress", "incomplete"] = "completed"
    content: list[ResponseOutputText]


class ResponseOutputReasoning(ApiModel):
    """OpenAI Responses API reasoning output item (``summary``, not top-level ``text``)."""

    type: Literal["reasoning"] = "reasoning"
    id: str
    summary: list[ResponseReasoningSummaryPart] = Field(default_factory=list)


def reasoning_summary_plaintext(item: ResponseOutputReasoning) -> str:
    return "".join(part.text for part in item.summary)


class ResponseFunctionToolCall(ApiModel):
    type: Literal["function_call"] = "function_call"
    id: str
    call_id: str
    name: str
    arguments: str = "{}"
    namespace: str | None = None
    status: Literal["completed", "in_progress", "incomplete"] = "completed"


ResponseOutputContent = Annotated[
    ResponseOutputMessage | ResponseOutputReasoning | ResponseFunctionToolCall,
    Field(discriminator="type"),
]


class ResponseInputTokensDetails(ApiModel):
    cached_tokens: int = 0


class ResponseOutputTokensDetails(ApiModel):
    reasoning_tokens: int = 0


class ResponseUsage(ApiModel):
    """Token usage for a response."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: ResponseInputTokensDetails = Field(default_factory=ResponseInputTokensDetails)
    output_tokens_details: ResponseOutputTokensDetails = Field(default_factory=ResponseOutputTokensDetails)


class ResponseError(ApiModel):
    """Error information for a failed response."""

    code: str
    message: str


class ResponseIncompleteDetails(ApiModel):
    """Why a response is ``incomplete`` (e.g. truncated at ``max_output_tokens``)."""

    reason: Literal["max_output_tokens", "content_filter"]


class Response(ApiModel):
    """OpenAI-compatible response object."""

    id: str
    object: Literal["response"] = "response"
    created_at: int
    model: str
    status: ResponseStatus
    incomplete_details: ResponseIncompleteDetails | None = None
    output: list[ResponseOutputContent] | None = None
    error: ResponseError | None = None
    usage: ResponseUsage | None = None
    service_tier: str = "default"


ResponseStreamPart = Annotated[
    ResponseOutputText | ResponseReasoningSummaryPart,
    Field(discriminator="type"),
]


class ResponseStreamEvent(ApiModel):
    """One server-sent event from ``stream_response`` (OpenAI Responses streaming shape).

    ``type`` names the event, e.g. ``response.output_text.delta`` or
    ``response.completed``. Fields an event type doesn't carry stay ``None``.
    """

    type: str
    sequence_number: int | None = None
    # Lifecycle events (created, in_progress, completed, failed, incomplete).
    response: Response | None = None
    output_index: int | None = None
    content_index: int | None = None
    summary_index: int | None = None
    item_id: str | None = None
    # ``output_item.added`` / ``.done``; ``added`` items arrive with empty content.
    item: ResponseOutputContent | None = None
    part: ResponseStreamPart | None = None
    delta: str | None = None
    text: str | None = None
    arguments: str | None = None


class ResponseListItem(ApiModel):
    """Item in a response list."""

    id: str
    object: Literal["response"] = "response"
    created_at: int
    model: str
    status: ResponseStatus


class ResponseList(ApiModel):
    """Paginated list of responses."""

    object: Literal["list"] = "list"
    data: list[ResponseListItem]
    has_more: bool = False


# Embeddings (OpenAI-compatible)
EmbeddingInput = str | list[str] | list[int] | list[list[int]]
EmbeddingEncodingFormat = Literal["float", "base64"]


class EmbeddingData(ApiModel):
    index: int
    object: Literal["embedding"] = "embedding"
    embedding: list[float] | str


class EmbeddingUsage(ApiModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(ApiModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage | None = None


class CreateEmbeddingPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    input: EmbeddingInput
    encoding_format: EmbeddingEncodingFormat = "float"
    dimensions: int | None = None
    user: str | None = None


# Decisions (POST /v1/decisions): classify one ``state`` against up to 64 questions.
class ChoiceQuestion(BaseModel):
    """Pick one of ``criteria`` (label -> description)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"] = "choice"
    instructions: Any = ""
    criteria: dict[str, Any]


class ScoreQuestion(BaseModel):
    """Rate on the ordinal scale ``criteria`` (index 0 = lowest)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["score"] = "score"
    instructions: Any = ""
    criteria: list[Any]


class NoulQuestion(BaseModel):
    """Yes/no probability; ``criteria`` may describe the ``"true"``/``"false"`` cases."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"] = "noul"
    instructions: Any = ""
    criteria: dict[Literal["true", "false"], Any] | None = None


DecisionQuestion = Annotated[ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")]


class DecisionImage(BaseModel):
    """Embedded PNG, JPEG, or WebP. Image bytes are base64-encoded, never uploaded separately."""

    model_config = ConfigDict(extra="forbid")

    content_type: Literal["image/png", "image/jpeg", "image/webp"]
    base64: str


class CreateDecisionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    state: Any
    questions: dict[str, DecisionQuestion]
    images: list[str | DecisionImage] = Field(default_factory=list, max_length=4)


class ChoiceAnswer(ApiModel):
    type: Literal["choice"] = "choice"
    choice: str
    confidence: float
    probabilities: dict[str, float]


class ScoreAnswer(ApiModel):
    type: Literal["score"] = "score"
    score: float
    confidence: float
    legend: dict[str, Any]
    probabilities: dict[str, float]


class NoulAnswer(ApiModel):
    type: Literal["noul"] = "noul"
    noul: float


DecisionAnswer = Annotated[ChoiceAnswer | ScoreAnswer | NoulAnswer, Field(discriminator="type")]


class DecisionUsage(ApiModel):
    input_tokens: int
    output_tokens: int = 0


class DecisionResponse(ApiModel):
    model: str
    answers: dict[str, DecisionAnswer]
    usage: DecisionUsage


# Model catalog (GET /v1/models)
class CapabilityFlag(ApiModel):
    supported: bool = False


class ThinkingCapabilityTypes(ApiModel):
    enabled: CapabilityFlag = Field(default_factory=CapabilityFlag)
    adaptive: CapabilityFlag = Field(default_factory=CapabilityFlag)


class ThinkingCapability(ApiModel):
    supported: bool = False
    types: ThinkingCapabilityTypes = Field(default_factory=ThinkingCapabilityTypes)


class ModelCapabilities(ApiModel):
    image_input: CapabilityFlag = Field(default_factory=CapabilityFlag)
    pdf_input: CapabilityFlag = Field(default_factory=CapabilityFlag)
    thinking: ThinkingCapability = Field(default_factory=ThinkingCapability)
    tools: CapabilityFlag = Field(default_factory=CapabilityFlag)
    dimensions: int | None = None


class OffPeakPricing(ApiModel):
    start: str
    end: str
    timezone: str
    price_factor: float


class ModelPricing(ApiModel):
    input_per_million_usd: float | None = None
    output_per_million_usd: float | None = None
    cached_input_per_million_usd: float | None = None
    off_peak: OffPeakPricing | None = None


class ModelInfo(ApiModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "sference"
    display_name: str
    # str, not Literal: a new modality must not break parsing the whole catalog.
    modality: str = "text_generation"
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    context_tokens: int | None = None
    provider: str | None = None
    pricing: ModelPricing | None = None
    released: str | None = None
    alias_of: str | None = None
    retires_on: str | None = None


class ModelList(ApiModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


# Billing (GET /v1/billing/*). Amounts are in micro-USD.
class BillingBalanceResponse(ApiModel):
    credit_balance_micros: int
    billing_mode: str = "prepaid"


class BillingSummary(ApiModel):
    days: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_cached_prompt_tokens: int = 0
    total_requests: int
    estimated_cost_micros: int
    credit_balance_micros: int
    billing_mode: str = "prepaid"


class PricingInfo(ApiModel):
    model: str
    prompt_token_price_micros: int
    completion_token_price_micros: int
    effective_from: str


class BillingModelSummary(ApiModel):
    model: str
    service_tier: str = "default"
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_prompt_tokens: int = 0
    total_requests: int
    cost_micros: int
    pricing: PricingInfo | None = None


class BillingSummaryResponse(ApiModel):
    summary: BillingSummary
    by_model: list[BillingModelSummary]


class LedgerEntryItem(ApiModel):
    id: str
    created_at: str
    type: str
    amount_micros: int
    balance_after_micros: int | None = None
    description: str
    external_provider: str | None = None
    external_payment_id: str | None = None


class LedgerListResponse(ApiModel):
    items: list[LedgerEntryItem]
    next_starting_after: str | None = None


# Usage (GET /v1/usage/*)
class UsageSummary(ApiModel):
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_cached_prompt_tokens: int = 0
    total_requests: int
    total_cost_micros: int = 0


class ModelUsageSummary(UsageSummary):
    model: str
    service_tier: str = "default"


class UserUsageSummary(UsageSummary):
    user_id: str | None = None


class UsageSummaryResponse(ApiModel):
    summary: UsageSummary
    by_model: list[ModelUsageSummary]
    by_user: list[UserUsageSummary] = Field(default_factory=list)


class DailyUsage(ApiModel):
    date: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_prompt_tokens: int = 0
    request_count: int
    cost_micros: int = 0
    input_cost_micros: int = 0
    cached_cost_micros: int = 0
    output_cost_micros: int = 0


class UsageDailyResponse(ApiModel):
    items: list[DailyUsage]


class UsageByKeyItem(ApiModel):
    api_key_id: str
    name: str
    prefix: str
    tag: str | None = None
    owner_email: str
    department_id: str | None = None
    revoked: bool = False
    last_used_at: str | None = None
    requests: int = 0
    total_tokens: int = 0
    cost_micros: int = 0


class UsageByTagItem(ApiModel):
    tag: str
    keys: int = 0
    requests: int = 0
    total_tokens: int = 0
    cost_micros: int = 0


class UsageByKeyResponse(ApiModel):
    period_start: str
    period_end: str
    items: list[UsageByKeyItem]
    by_tag: list[UsageByTagItem] = Field(default_factory=list)
