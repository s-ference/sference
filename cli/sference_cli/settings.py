"""Defaults shared by every command: API base URL and catalog model."""

from __future__ import annotations

import os

DEFAULT_API_BASE_URL = "https://api.sference.com"

# Pinned snapshot rather than the `deepseek-ai/DeepSeek-V4.1-Flash` alias: the
# alias has a retires_on date and GET /v1/models says to migrate to its target.
# It also accepts thinking-disabled requests (Claude Code's default shape), which
# the GLM-5.3 family rejects, so one default serves every launcher.
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"


def resolve_api_base_url(explicit: str | None = None) -> str:
    """``--base-url`` > ``SFERENCE_BASE_URL`` > the public API."""
    if explicit and explicit.strip():
        return explicit.strip().rstrip("/")
    env = os.environ.get("SFERENCE_BASE_URL")
    if env and env.strip():
        return env.strip().rstrip("/")
    return DEFAULT_API_BASE_URL


def resolve_openai_base_url(explicit: str | None = None) -> str:
    """OpenAI-compatible base URL (``.../v1``) for tool config files."""
    base = resolve_api_base_url(explicit)
    return base if base.endswith("/v1") else f"{base}/v1"


def resolve_model(explicit: str | None = None) -> str:
    """``--model`` > ``SFERENCE_MODEL`` > :data:`DEFAULT_MODEL`."""
    if explicit and explicit.strip():
        return explicit.strip()
    env = os.environ.get("SFERENCE_MODEL")
    if env and env.strip():
        return env.strip()
    return DEFAULT_MODEL
