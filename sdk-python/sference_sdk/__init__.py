from importlib.metadata import PackageNotFoundError, version as _distribution_version


def _package_version(distribution_name: str) -> str:
    """Installed distribution version (matches ``pyproject.toml`` / release tag)."""
    try:
        return _distribution_version(distribution_name)
    except PackageNotFoundError:
        return "0.0.0-dev"


__version__ = _package_version("sference-sdk")

from .async_client import AsyncSferenceClient
from .checkpoint import clear_checkpoint, load_checkpoint, save_checkpoint
from .client import ApiError, SferenceClient
from .models import (
    COMPLETION_WINDOWS,
    Batch,
    BatchList,
    BatchResults,
    CompletionWindow,
    InferenceRequest,
    LoginResponse,
    Stream,
    StreamInferenceCompletionEvent,
    StreamEventList,
    StreamList,
)

__all__ = [
    "__version__",
    "ApiError",
    "AsyncSferenceClient",
    "SferenceClient",
    "Batch",
    "BatchList",
    "BatchResults",
    "COMPLETION_WINDOWS",
    "CompletionWindow",
    "LoginResponse",
    "InferenceRequest",
    "Stream",
    "StreamInferenceCompletionEvent",
    "StreamEventList",
    "StreamList",
    "clear_checkpoint",
    "load_checkpoint",
    "save_checkpoint",
]
