"""Evaluation-side prompt helpers.

The training-time prompt template lives in :mod:`src.prompts` (single source of
truth, also used by serving). This module re-exports ``build_no_rag_user_message``
for symmetry with the evaluation plan §2.2 and adds ``extract_transcription``.
"""

from __future__ import annotations

from src.prompts import build_no_rag_user_message

__all__ = ["build_no_rag_user_message", "extract_transcription"]

_PATIENT_NOTE_HEADER = "PATIENT NOTE:\n"
_QUESTION_DELIMITER = "\n\nQUESTION:"


def extract_transcription(example: dict) -> str:
    """Pulls the PATIENT NOTE section out of a dataset example's user message.

    Args:
        example: A dataset row with a ``messages`` list (system/user/assistant).

    Returns:
        The transcription string as stored in the example.

    Raises:
        ValueError: If the user message does not match the expected template.
    """
    user_msg = next(m["content"] for m in example["messages"] if m["role"] == "user")
    if not user_msg.startswith(_PATIENT_NOTE_HEADER):
        raise ValueError("User message missing 'PATIENT NOTE:' header")
    body = user_msg[len(_PATIENT_NOTE_HEADER) :]
    idx = body.find(_QUESTION_DELIMITER)
    if idx == -1:
        raise ValueError("User message missing 'QUESTION:' delimiter")
    return body[:idx]
