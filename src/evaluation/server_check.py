"""Preflight validator for the llama.cpp OpenAI-compatible server.

Phase 1 and Phase 2 of the ablation study require different GGUFs loaded in the
local llama.cpp server. This module fails hard before any sample runs if the
loaded model does not match the expected identifier.
"""

from __future__ import annotations

import httpx

from src.logging_config import get_logger

logger = get_logger(__name__)


def validate_llamacpp_model(endpoint: str, expected_model: str, timeout: float = 5.0) -> None:
    """Check the llama.cpp server has the expected GGUF loaded.

    Calls ``GET {endpoint}/models`` (OpenAI-compatible) and inspects ``data[0].id``.

    Args:
        endpoint: Base URL ending in ``/v1`` (e.g. ``http://localhost:8001/v1``).
        expected_model: Exact string expected as the loaded model id.
        timeout: HTTP timeout in seconds.

    Raises:
        RuntimeError: If the server is unreachable, returns an unexpected
            response shape, or has a different model loaded.
    """
    url = endpoint.rstrip("/") + "/models"
    try:
        response = httpx.get(url, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"llama.cpp server unreachable at {url}: {exc}") from exc

    payload = response.json()
    data = payload.get("data") or []
    if not data:
        raise RuntimeError(f"llama.cpp /models returned no entries: {payload!r}")

    loaded = data[0].get("id")
    logger.info("llama.cpp loaded model: %s (expected: %s)", loaded, expected_model)
    if loaded != expected_model:
        raise RuntimeError(
            f"llama.cpp model mismatch — loaded={loaded!r}, expected={expected_model!r}. "
            f"Restart the server with the correct GGUF before re-running."
        )
