import datetime
import re

# Evidence-quality scores ordered highest-to-lowest following the EBM pyramid.
# Dict insertion order matters: _pub_type_score walks entries top-to-bottom and
# returns the first match, so a doc tagged as both "review" and "meta-analysis"
# is correctly credited as a meta-analysis.
_PUB_TYPE_SCORES: dict[str, float] = {
    "practice guideline": 1.0,
    "guideline": 1.0,
    "systematic review": 0.9,
    "meta-analysis": 0.9,
    "randomized controlled trial": 0.75,
    "clinical trial": 0.70,  # includes phase I/II — weaker than RCT
    "review": 0.55,  # narrative review; quality varies widely
    "journal article": 0.30,
}


def _pub_type_score(pub_types: list[str]) -> float:
    """Return the highest EBM-pyramid score for a document's publication types.

    Args:
        pub_types: Raw publication-type strings from the PubMed payload
            (e.g. ``["Journal Article", "Randomized Controlled Trial"]``).

    Returns:
        Score in [0.20, 1.0]. Returns 0.20 when no type matches the lookup
        table (conservative fallback for unclassified documents).
    """
    lower = {p.lower() for p in pub_types}
    # Walk highest-to-lowest so the best matching type wins for multi-typed docs.
    for label, score in _PUB_TYPE_SCORES.items():
        if label in lower:
            return score
    return 0.20


def _recency_score(year: int, current_year: int) -> float:
    """Map publication age to a recency score using a step-decay function.

    Thresholds reflect typical guideline update cycles in clinical medicine:
    guidelines are usually revised every 3–5 years, and evidence older than
    12 years is rarely cited in current practice.

    Args:
        year: Four-digit publication year, or 0 / None when unknown.
        current_year: Reference year for computing age (injected for testability).

    Returns:
        Score in [0.20, 1.0].  Unknown year returns 0.20.
    """
    if not year:
        # Undated docs are as conservative as the oldest tier.
        return 0.20
    age = current_year - int(year)
    if age <= 3:
        return 1.00
    if age <= 7:
        return 0.75
    if age <= 12:
        return 0.60
    return 0.20


def _token_overlap(query_tokens: set[str], target_tokens: set[str]) -> float:
    """Compute query-recall overlap: fraction of query tokens found in target.

    Measures how much of the *query* is covered by the target.

    Args:
        query_tokens: Tokenized query.
        target_tokens: Tokenized field from the document (e.g. MeSH or title).

    Returns:
        Value in [0.0, 1.0], or 0.0 for an empty query.
    """
    if not query_tokens:
        return 0.0
    return len(query_tokens & target_tokens) / len(query_tokens)


# Matches runs of lowercase letters and digits after lowercasing.
# Strips punctuation, hyphens, and non-ASCII characters (e.g. "β") so that
# "Diabetes Mellitus, Type 2" and "diabetes mellitus type 2" produce identical
# token sets.  This is intentional: the overlap signals are coarse keyword
# features, not precise NLP.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    """Lower-case and extract alphanumeric tokens from *text*.

    Args:
        text: Raw string (query, title, or MeSH term).

    Returns:
        Set of lowercase alphanumeric tokens; empty set for blank input.
    """
    return set(_TOKEN_RE.findall(text.lower()))


class ClinicalScorer:
    """Weighted linear scorer combining MedCPT and clinical domain signals.

    Computes a ``final_score`` for each retrieved document after cross-encoder
    reranking.  The score drives both result ordering and the abstention gate in
    ``RAGPipeline`` (documents below ``min_final_score`` are filtered out before
    the LLM is called).

    The five components and their roles:

    - **cross_encoder** — MedCPT semantic relevance; it is the dominant
      weight as it is the most expensive and accurate signal.
    - **mesh_overlap** — fraction of query tokens found in the document's MeSH
      descriptors.
    - **publication_type** — EBM evidence-quality tier (guideline > RCT > review).
    - **recency** — step-decay on publication age; penalises outdated evidence.
    - **title_keyword** — lightweight title match; kept low to avoid double-
      counting with the cross-encoder.

    Weights must sum to 1.0 so that ``final_score ∈ [0, 1]`` and the threshold
    in ``RAGPipeline.min_final_score`` retains a stable, interpretable meaning.
    """

    def __init__(self, weights: dict, current_year: int | None = None):
        """
        Args:
            weights: Dict with keys ``cross_encoder``, ``mesh_overlap``,
                ``publication_type``, ``recency``, ``title_keyword``.
                Values must sum to 1.0.
            current_year: Override for the reference year used in recency
                scoring.  Defaults to the current calendar year.
        """
        self.weights = weights
        self.current_year = current_year or datetime.date.today().year

    def score(self, query: str, doc: dict) -> float:
        """Compute the final clinical relevance score for a single document.

        Args:
            query: The original clinical question string.
            doc: Document dict as returned by ``MedCPTReranker.rerank``.
                Expected keys: ``cross_encoder_score`` (float), ``mesh_terms``
                (list[str]), ``title`` (str), ``publication_types`` (list[str]),
                ``year`` (int).

        Returns:
            Weighted linear combination of the five signals, rounded to 4
            decimal places.
        """
        query_tokens = _tokenize(query)
        # Flatten MeSH phrases ("Diabetes Mellitus, Type 2") into individual
        # tokens so single query words can intersect with multi-word descriptors.
        mesh_tokens = {tok for term in doc.get("mesh_terms", []) for tok in _tokenize(term)}
        title_tokens = _tokenize(doc.get("title", ""))
        w = self.weights
        return round(
            w["cross_encoder"] * doc.get("cross_encoder_score", 0.0)
            + w["mesh_overlap"] * _token_overlap(query_tokens, mesh_tokens)
            + w["publication_type"] * _pub_type_score(doc.get("publication_types", []))
            + w["recency"] * _recency_score(doc.get("year", 0), self.current_year)
            + w["title_keyword"] * _token_overlap(query_tokens, title_tokens),
            4,
        )


# Matches "PMID", optional separator, then 4–9 digits.  The range [4,9] avoids
# false positives on other numeric strings while covering all valid PMIDs.
_PMID_RE = re.compile(r"PMID[:\s\-]*(\d{4,9})", re.IGNORECASE)


def verify_citations(response: str, retrieved_pmids: list[str]) -> dict:
    """Check whether every PMID cited in the LLM response was actually retrieved.

    Hallucinated citations — PMIDs present in the response but not in the
    retrieval set — are a safety concern in clinical contexts.  ``RAGPipeline``
    includes this check in every ``run()`` result so callers can log or filter
    responses with fabricated references.

    Args:
        response: Raw LLM-generated text, potentially containing PMID references
            in formats like ``PMID: 12345678``, ``PMID-12345678``, or
            ``PMID 12345678``.
        retrieved_pmids: PMIDs of the documents that were passed to the LLM as
            context (the ground truth for what citations are legitimate).

    Returns:
        Dict with three keys:

        - ``cited_pmids`` (list[str]): All PMIDs found in the response, sorted.
        - ``hallucinated_pmids`` (list[str]): PMIDs cited but not retrieved.
        - ``citation_ok`` (bool): ``True`` iff no hallucinated PMIDs are present.
    """
    cited = set(_PMID_RE.findall(response))
    retrieved = set(str(p) for p in retrieved_pmids)
    hallucinated = cited - retrieved
    return {
        "cited_pmids": sorted(cited),
        "hallucinated_pmids": sorted(hallucinated),
        "citation_ok": len(hallucinated) == 0,
    }
