from src.llm_clients import LLMClient
from src.logging_config import get_logger

from .reranker import MedCPTReranker
from .retriever import Retriever
from .scorer import ClinicalScorer, verify_citations

logger = get_logger(__name__)


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline for clinical queries.

    Orchestrates hybrid document retrieval, cross-encoder reranking, clinical
    relevance scoring, LLM response generation, and citation verification.
    Supports abstention when insufficient high-quality evidence is available.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: MedCPTReranker,
        scorer: ClinicalScorer,
        client: LLMClient,
        system_prompt: str,
        candidate_k: int = 80,
        final_k: int = 6,
        min_year: int = 2015,
        min_final_score: float = 0.35,
        min_relevant_docs: int = 2,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ):
        """Initialize the RAG pipeline with retrieval, reranking, and LLM components.

        Args:
            retriever: Hybrid semantic+lexical retrieval engine (Qdrant-backed).
            reranker: MedCPT cross-encoder for query-document relevance scoring.
            scorer: Clinical relevance scorer combining publication type, recency,
                and semantic match.
            client: LLM client for response generation (e.g., Claude, Llama).
            system_prompt: System instructions for the LLM.
            candidate_k: Number of initial retrieval candidates (default 80).
            final_k: Number of documents to include in the LLM prompt (default 6).
            min_year: Minimum publication year filter for retrieval (default 2015).
            min_final_score: Relevance score threshold [0, 1]; documents below
                this are excluded from response generation (default 0.35).
            min_relevant_docs: Minimum number of documents scoring >= min_final_score
                required to proceed; below this triggers abstention (default 2).
            max_tokens: Maximum tokens in LLM response (default 1024).
            temperature: Sampling temperature for LLM (default 0.1, near-deterministic).

        """
        self.retriever = retriever
        self.reranker = reranker
        self.scorer = scorer
        self.candidate_k = candidate_k
        self.final_k = final_k
        self.min_year = min_year
        self.min_final_score = min_final_score
        self.min_relevant_docs = min_relevant_docs
        self._client = client
        self._system = system_prompt
        self._max_tokens = max_tokens
        self._temperature = temperature

    def _generate(self, prompt: str) -> str:
        """Generate an LLM response for the given prompt.

        Args:
            prompt: Formatted prompt containing clinical query and retrieved evidence.

        Returns:
            Generated response text from the LLM.

        """
        return self._client.generate(
            system=self._system,
            user=prompt,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

    def _build_prompt(self, query: str, docs: list[dict]) -> str:
        """Format retrieved documents and clinical query into a structured prompt.

        Args:
            query: Clinical question or query string.
            docs: List of scored and ranked documents, each containing rank, pmid,
                year, mesh_category, title, and abstract.

        Returns:
            Formatted prompt string with evidence section and clinical query.

        """
        parts = "\n\n---\n\n".join(
            f"[{d['rank']}] PMID: {d['pmid']} | Year: {d['year']} | MeSH: {d['mesh_category']}\n"
            f"Title: {d['title']}\nAbstract: {d['abstract']}"
            for d in docs
        )
        return (
            f"=== RETRIEVED EVIDENCE ===\n\n{parts}\n\n"
            f"=== END OF EVIDENCE ===\n\n"
            f"CLINICAL QUERY: {query}\n\nEVIDENCE-BASED RESPONSE:"
        )

    def retrieve(self, query: str) -> list[dict]:
        """Retrieve and score documents via hybrid search, reranking, and clinical relevance.

        Pipeline: hybrid semantic+lexical retrieval → MedCPT cross-encoder reranking →
        ClinicalScorer (combining publication type, recency, and semantic match).

        Args:
            query: Clinical question or search string.

        Returns:
            List of all scored documents sorted by final_score (descending). Each dict
            contains the original document metadata plus a 'final_score' field [0, 1].

        """
        logger.debug("Retrieving candidates for query: %.120s", query)
        candidates = self.retriever.search(
            query, candidate_k=self.candidate_k, min_year=self.min_year
        )
        reranked = self.reranker.rerank(query, candidates)
        scored = [{**doc, "final_score": self.scorer.score(query, doc)} for doc in reranked]
        scored.sort(key=lambda x: x["final_score"], reverse=True)
        logger.debug(
            "Retrieved %d candidates (top score: %.4f)",
            len(scored),
            scored[0]["final_score"] if scored else 0.0,
        )
        return scored

    def run(self, query: str) -> dict:
        """Execute full RAG pipeline with evidence-based response generation.

        Retrieves documents, filters by min_final_score and min_relevant_docs,
        and either generates an evidence-based response or abstrains (returns None)
        when insufficient high-quality documents are available.

        Args:
            query: Clinical question.

        Returns:
            Dict with keys:
                - query: The input query.
                - abstained: Boolean; True if insufficient evidence, False otherwise.
                - reason: (Only if abstained=True) Explanation of abstention.
                - candidates_before_rerank: Total documents retrieved initially.
                - top_candidates: (Only if abstained=True) Top 5 candidates for reference.
                - docs: (Only if abstained=False) List of final_k documents with rank,
                    final_score, and all metadata; empty list if abstained.
                - response: Generated response string, or None if abstained.
                - citation_check: Dict with keys 'citation_ok' (bool) and
                    'hallucinated_pmids' (list of PMID strings); or None if abstained.
                - avg_final_score: Average final_score of selected documents; or None
                    if abstained.

        """
        logger.info("RAG run started — query: %.120s", query)
        scored = self.retrieve(query)
        relevant = [d for d in scored if d["final_score"] >= self.min_final_score]

        if len(relevant) < self.min_relevant_docs:
            logger.warning(
                "Abstaining: only %d/%d docs scored >= %.2f (need %d)",
                len(relevant),
                len(scored),
                self.min_final_score,
                self.min_relevant_docs,
            )
            return {
                "query": query,
                "abstained": True,
                "reason": (
                    f"Fewer than {self.min_relevant_docs} documents scored "
                    f">= {self.min_final_score} after clinical reranking."
                ),
                "candidates_before_rerank": len(scored),
                "top_candidates": scored[:5],
                "docs": [],
                "response": None,
                "citation_check": None,
                "avg_final_score": None,
            }

        final_docs = [{**d, "rank": i + 1} for i, d in enumerate(scored[: self.final_k])]

        prompt = self._build_prompt(query, final_docs)
        response = self._generate(prompt)
        citation_check = verify_citations(response, [d["pmid"] for d in final_docs])

        avg_score = round(sum(d["final_score"] for d in final_docs) / len(final_docs), 4)
        logger.info(
            "RAG run complete — docs: %d, avg_score: %.4f, citation_ok: %s, hallucinated: %s",
            len(final_docs),
            avg_score,
            citation_check["citation_ok"],
            citation_check["hallucinated_pmids"] or "none",
        )
        return {
            "query": query,
            "abstained": False,
            "candidates_before_rerank": len(scored),
            "docs": final_docs,
            "response": response,
            "citation_check": citation_check,
            "avg_final_score": avg_score,
        }

    def run_base(self, query: str) -> str:
        """Generate a response using LLM only, without retrieval (baseline).

        Used for ablation studies to measure the contribution of RAG components.
        The LLM receives only the clinical query, no retrieved evidence or citations.

        Args:
            query: Clinical question.

        Returns:
            Generated response text from the LLM.

        """
        return self._generate(f"CLINICAL QUERY: {query}\n\nEVIDENCE-BASED RESPONSE:")
