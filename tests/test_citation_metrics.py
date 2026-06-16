"""Unit tests for the deterministic citation metrics."""

import math

from src.evaluation.aggregation import compute_citation_metrics
from src.evaluation.metrics import (
    extract_pmid_citations,
    has_bracket_reference,
    has_placeholder_citation,
)


def test_extract_pmid_citations():
    assert extract_pmid_citations("see (PMID: 12345678) and (PMID: 999111)") == [
        "12345678",
        "999111",
    ]
    assert extract_pmid_citations("(PMID: XXXXXXXX)") == []  # placeholder is not a real cite
    assert extract_pmid_citations(None) == []
    assert extract_pmid_citations("") == []


def test_bracket_and_placeholder_detectors():
    assert has_bracket_reference("supported by [1] and [12]")
    assert not has_bracket_reference("(PMID: 12345678)")
    assert has_placeholder_citation("evidence (PMID: XXXXXXXX)")
    assert has_placeholder_citation("PMID: ####")
    assert not has_placeholder_citation("(PMID: 12345678)")
    assert not has_placeholder_citation(None)


def _rec(response, retrieved=None, abstained=False, error=None, hallucinated=None):
    return {
        "response": response,
        "retrieved_pmids": retrieved or [],
        "abstained": abstained,
        "error": error,
        "hallucinated_pmids": hallucinated or [],
    }


def test_compute_citation_metrics_rag_grounding():
    records = [
        # cites 2 real PMIDs, one grounded one not
        _rec("a (PMID: 111111) b (PMID: 222222)", retrieved=["111111"], hallucinated=["222222"]),
        # cites 1 grounded PMID
        _rec("x (PMID: 333333)", retrieved=["333333", "444444"]),
    ]
    m = compute_citation_metrics(records)
    assert m["pmid_citation_rate"] == 1.0  # both answered cite >=1 real PMID
    assert m["placeholder_citation_rate"] == 0.0
    assert m["bracket_citation_rate"] == 0.0
    # 3 cited, 2 grounded (111111, 333333)
    assert math.isclose(m["citation_grounding_precision"], 2 / 3)
    assert math.isclose(m["pmid_citations_per_answered_mean"], 1.5)
    assert m["hallucinated_pmids_rate"] == 0.5  # 1 of 2 records has a hallucination


def test_compute_citation_metrics_no_rag_placeholder():
    records = [
        _rec("rec (PMID: XXXXXXXX)"),          # placeholder, no retrieval
        _rec("rec with no citation at all"),    # nothing
        _rec("see [1] for details"),            # bracket style
    ]
    m = compute_citation_metrics(records)
    assert m["placeholder_citation_rate"] == 1 / 3
    assert m["no_citation_rate"] == 1 / 3
    assert m["bracket_citation_rate"] == 1 / 3
    assert m["pmid_citation_rate"] == 0.0
    # No retrieval anywhere → grounding precision undefined.
    assert math.isnan(m["citation_grounding_precision"])


def test_abstained_and_errored_excluded_from_answered():
    records = [
        _rec("(PMID: 111111)", retrieved=["111111"]),
        _rec(None, abstained=True),
        _rec(None, error="boom"),
    ]
    m = compute_citation_metrics(records)
    # Only 1 answered record → rate over answered, not total.
    assert m["pmid_citation_rate"] == 1.0
