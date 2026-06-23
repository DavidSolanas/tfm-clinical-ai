"""Unit tests for src.api.formatting — protects the Evidence Ledger signature.

These cover the two pieces the plan singles out (6.2): PMID linkify and the
Cited/Available/Hallucinated state mapping. No model/services required.
"""

from __future__ import annotations

from src.api.formatting import (
    STATE_AVAILABLE,
    STATE_CITED,
    citation_states,
    linkify_pmids,
    pubmed_url,
    render_citation_badge,
    render_evidence_ledger,
)


def test_pubmed_url():
    assert pubmed_url("12345678") == "https://pubmed.ncbi.nlm.nih.gov/12345678/"
    assert pubmed_url(12345678) == "https://pubmed.ncbi.nlm.nih.gov/12345678/"


def test_linkify_pmids_creates_pubmed_anchor():
    html = linkify_pmids("Metformin is first-line (PMID: 12345678).")
    assert 'href="https://pubmed.ncbi.nlm.nih.gov/12345678/"' in html
    assert 'data-pmid="12345678"' in html
    assert "(PMID: 12345678)" in html


def test_linkify_escapes_html():
    html = linkify_pmids("Risk <script>alert(1)</script> noted.")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_linkify_handles_no_pmid():
    text = "No citations here."
    assert linkify_pmids(text) == "No citations here."


def test_linkify_multiple_pmids():
    html = linkify_pmids("See (PMID: 111111) and (PMID: 222222).")
    assert html.count("pubmed.ncbi.nlm.nih.gov") == 2
    assert 'data-pmid="111111"' in html
    assert 'data-pmid="222222"' in html


def test_citation_states_cited_vs_available():
    docs = [{"pmid": "111"}, {"pmid": "222"}, {"pmid": "333"}]
    check = {"cited_pmids": ["111", "333"], "hallucinated_pmids": [], "citation_ok": True}
    states = citation_states(docs, check)
    assert states["111"] == STATE_CITED
    assert states["222"] == STATE_AVAILABLE
    assert states["333"] == STATE_CITED


def test_citation_states_no_check_all_available():
    docs = [{"pmid": "111"}, {"pmid": "222"}]
    states = citation_states(docs, None)
    assert all(s == STATE_AVAILABLE for s in states.values())


def test_citation_badge_ok():
    check = {"cited_pmids": ["111", "222"], "hallucinated_pmids": [], "citation_ok": True}
    html = render_citation_badge(check)
    assert "evl-badge--ok" in html
    assert "2 citations verified" in html


def test_citation_badge_hallucinated_flagged():
    check = {"cited_pmids": ["999"], "hallucinated_pmids": ["999"], "citation_ok": False}
    html = render_citation_badge(check)
    assert "evl-badge--alert" in html
    assert "999" in html
    assert 'href="https://pubmed.ncbi.nlm.nih.gov/999/"' in html


def test_citation_badge_none_returns_empty():
    assert render_citation_badge(None) == ""


def test_render_ledger_marks_cited_card():
    docs = [
        {"rank": 1, "pmid": "111", "year": 2021, "title": "A", "final_score": 0.82},
        {"rank": 2, "pmid": "222", "year": 2019, "title": "B", "final_score": 0.41},
    ]
    check = {"cited_pmids": ["111"], "hallucinated_pmids": [], "citation_ok": True}
    html = render_evidence_ledger(docs, check)
    assert "evl-card--cited" in html  # 111 was cited
    assert 'data-pmid="111"' in html
    assert 'data-pmid="222"' in html
    assert "PMID: 111" in html


def test_render_ledger_empty():
    assert "No sources retrieved" in render_evidence_ledger([], None)
