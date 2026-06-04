"""Shared prompt templates and helpers used by dataset building, RAG serving,
and evaluation.

Single source of truth for the clinical-evidence prompt format. The fine-tuned
model is trained on outputs of ``_user_message`` with ``_SYSTEM_PROMPT`` as the
system message; serving and evaluation must use the exact same template to
avoid out-of-distribution prompt drift.
"""

from __future__ import annotations

_SYSTEM_PROMPT = """\
You are a clinical evidence synthesis assistant.

Given a patient's clinical transcription and retrieved PubMed evidence, generate a structured \
evidence-based response. Use ONLY the provided PubMed evidence. Do not use outside medical knowledge.

Rules:
- Cite ONLY PMIDs from the provided evidence. Do not invent PMIDs.
- Do not cite a PMID unless that document directly supports the claim.
- Format every citation as (PMID: XXXXXXXX) using the exact PMID number shown after "PMID:" in \
each document header. The leading [1], [2], etc. are rank indicators only — never use them as \
citation handles.
- Structure your response exactly as:
  1. Recommendation or finding (with PMID citations)
  2. Evidence basis (study type if inferable from the abstract)
  3. Uncertainty or gaps in the retrieved evidence
- If the retrieved evidence does not address the clinical question, respond ONLY with:
  "The retrieved context is off-topic or insufficient for this clinical case. \
The provided documents address [one concise sentence describing the topics covered \
by the retrieved documents, derived solely from the documents themselves]. \
No evidence-based recommendation can be made from the provided documents."
  Do NOT describe, summarize, diagnose, or add any details about the patient's \
clinical presentation or condition.\
"""

_USER_QUESTION = (
    "Based on this patient's clinical presentation, what evidence-based treatments are recommended?"
)

_NO_EVIDENCE_PREFIX = "the retrieved context is off-topic or insufficient for this clinical case."


def _format_evidence(docs: list[dict]) -> str:
    """Render documents into the per-doc block used in the training prompt."""
    return "\n\n---\n\n".join(
        f"[{d['rank']}] PMID: {d['pmid']} | Year: {d['year']}\n"
        f"Title: {d['title']}\nAbstract: {d['abstract']}"
        for d in docs
    )


def _user_message(transcription: str, docs: list[dict]) -> str:
    """Compose the user message with patient note, question, and evidence block."""
    return (
        f"PATIENT NOTE:\n{transcription}\n\n"
        f"QUESTION: {_USER_QUESTION}\n\n"
        f"RETRIEVED PUBMED EVIDENCE:\n{_format_evidence(docs)}"
    )


def build_no_rag_user_message(transcription: str) -> str:
    """User message for configs A/B (no retrieval); evidence section is ``(none)``."""
    return (
        f"PATIENT NOTE:\n{transcription}\n\n"
        f"QUESTION: {_USER_QUESTION}\n\n"
        f"RETRIEVED PUBMED EVIDENCE:\n(none)"
    )


def _is_no_evidence_response(example: dict) -> bool:
    """True if the assistant message is the no-evidence refusal template."""
    content = next(m["content"] for m in example["messages"] if m["role"] == "assistant")
    return content.lower().strip().startswith(_NO_EVIDENCE_PREFIX)
