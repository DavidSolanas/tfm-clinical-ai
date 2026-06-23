"""Pure render helpers for the Gradio UI — the "Evidence Ledger" signature.

All functions here are side-effect-free and HTML-string-producing so the Gradio
layer (``app.py``) stays declarative and these can be unit-tested without a
running model (the citation linkify and Cited/Available/Hallucinated state
mapping protect the page's signature feature — plan §5).
"""

from __future__ import annotations

import html
import re

# Matches inline (PMID: 12345678) citations in the model's answer. Mirrors the
# verifier regex in src/rag/scorer.py but captures the full span to replace it.
_INLINE_PMID_RE = re.compile(r"\(?\s*PMID[:\s\-]*?(\d{4,9})\s*\)?", re.IGNORECASE)

_PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

# Evidence-ledger state chips.
STATE_CITED = "cited"
STATE_AVAILABLE = "available"
STATE_HALLUCINATED = "hallucinated"


def pubmed_url(pmid: str | int) -> str:
    """Return the canonical PubMed URL for a PMID."""
    return _PUBMED_URL.format(pmid=str(pmid).strip())


def linkify_pmids(text: str) -> str:
    """HTML-escape *text* and turn inline ``(PMID: NNN)`` into PubMed links.

    Args:
        text: Raw model answer (plain text / light markdown).

    Returns:
        HTML-safe string where every PMID reference is an ``<a>`` to PubMed,
        tagged with ``data-pmid`` so the ledger can cross-highlight.
    """
    out: list[str] = []
    last = 0
    for m in _INLINE_PMID_RE.finditer(text):
        out.append(html.escape(text[last : m.start()]))
        pmid = m.group(1)
        out.append(
            f'<a class="evl-cite" data-pmid="{pmid}" href="{pubmed_url(pmid)}" '
            f'target="_blank" rel="noopener">(PMID: {pmid})</a>'
        )
        last = m.end()
    out.append(html.escape(text[last:]))
    return "".join(out)


def citation_states(docs: list[dict], citation_check: dict | None) -> dict[str, str]:
    """Map each retrieved PMID to its ledger state.

    Args:
        docs: Retrieved documents (each with a ``pmid`` key).
        citation_check: The pipeline's ``verify_citations`` output, or ``None``
            for no-RAG / abstained results.

    Returns:
        ``{pmid: state}`` where state is ``cited``/``available``. (Hallucinated
        PMIDs are not in *docs*; they are surfaced separately by the badge.)
    """
    cited = set()
    if citation_check:
        cited = {str(p) for p in citation_check.get("cited_pmids", [])}
    states: dict[str, str] = {}
    for doc in docs:
        pmid = str(doc.get("pmid", ""))
        states[pmid] = STATE_CITED if pmid in cited else STATE_AVAILABLE
    return states


def _score_meter(score: float | None) -> str:
    """Render a relevance score as a quiet horizontal meter."""
    if score is None:
        return ""
    pct = max(0.0, min(1.0, float(score))) * 100
    return (
        f'<div class="evl-meter" role="meter" aria-valuenow="{score:.2f}" '
        f'aria-valuemin="0" aria-valuemax="1" title="Relevance {score:.2f}">'
        f'<span style="width:{pct:.0f}%"></span></div>'
        f'<span class="evl-score">{score:.2f}</span>'
    )


_STATE_CHIP_LABEL = {
    STATE_CITED: "Cited",
    STATE_AVAILABLE: "Available",
    STATE_HALLUCINATED: "Hallucinated",
}


def _state_chip(state: str) -> str:
    label = _STATE_CHIP_LABEL.get(state, state.title())
    return f'<span class="evl-chip evl-chip--{state}">{label}</span>'


def render_evidence_card(doc: dict, state: str) -> str:
    """Render one Evidence-Ledger card (rank, PMID, year, score, state chip)."""
    pmid = str(doc.get("pmid", "—"))
    rank = doc.get("rank")
    year = doc.get("year") or "—"
    title = html.escape(str(doc.get("title", "")).strip() or "Untitled")
    score = doc.get("final_score")
    rank_badge = f'<span class="evl-rank">{rank}</span>' if rank is not None else ""
    pmid_link = (
        f'<a class="evl-pmid" href="{pubmed_url(pmid)}" target="_blank" '
        f'rel="noopener">PMID: {pmid}</a>'
    )
    return (
        f'<article class="evl-card evl-card--{state}" data-pmid="{pmid}">'
        f'<header class="evl-card__head">{rank_badge}{pmid_link}{_state_chip(state)}</header>'
        f'<h4 class="evl-card__title">{title}</h4>'
        f'<footer class="evl-card__foot">'
        f'<span class="evl-year">{year}</span>{_score_meter(score)}'
        f"</footer></article>"
    )


def render_evidence_ledger(docs: list[dict], citation_check: dict | None) -> str:
    """Render the full Evidence Ledger panel (the page's signature element)."""
    if not docs:
        return '<div class="evl-empty">No sources retrieved for this configuration.</div>'
    states = citation_states(docs, citation_check)
    cards = [
        render_evidence_card(d, states.get(str(d.get("pmid", "")), STATE_AVAILABLE)) for d in docs
    ]
    return f'<div class="evl-ledger">{"".join(cards)}</div>'


def render_citation_badge(citation_check: dict | None) -> str:
    """Render the citation-verification badge above the ledger.

    Returns a trust badge when all citations are grounded, or an alarm badge
    listing any hallucinated PMIDs (the citation-verifier made visible).
    """
    if not citation_check:
        return ""
    hallucinated = [str(p) for p in citation_check.get("hallucinated_pmids", [])]
    if not hallucinated:
        n = len(citation_check.get("cited_pmids", []))
        return (
            '<div class="evl-badge evl-badge--ok" role="status">'
            f'<span class="evl-badge__icon" aria-hidden="true">✓</span>'
            f"All {n} citation{'s' if n != 1 else ''} verified against retrieved sources"
            "</div>"
        )
    links = ", ".join(
        f'<a class="evl-pmid" href="{pubmed_url(p)}" target="_blank" rel="noopener">{p}</a>'
        for p in hallucinated
    )
    return (
        '<div class="evl-badge evl-badge--alert" role="alert">'
        '<span class="evl-badge__icon" aria-hidden="true">⚠</span>'
        f"Hallucinated PMID{'s' if len(hallucinated) != 1 else ''} (not retrieved): {links}"
        "</div>"
    )


def render_answer(response: str | None) -> str:
    """Render the model answer with inline PMID citations linkified."""
    if not response:
        return ""
    return f'<div class="evl-answer">{linkify_pmids(response)}</div>'


def render_abstention(reason: str | None) -> str:
    """Render the deliberate, non-error 'No sufficient evidence' state (plan §5)."""
    detail = html.escape(reason or "The retrieved evidence did not meet the relevance threshold.")
    return (
        '<div class="evl-abstain" role="status">'
        '<div class="evl-abstain__title">No sufficient evidence</div>'
        f'<p class="evl-abstain__body">{detail}</p>'
        '<p class="evl-abstain__note">This is a deliberate safety response: the system '
        "declines to recommend rather than answer without grounded sources.</p>"
        "</div>"
    )
