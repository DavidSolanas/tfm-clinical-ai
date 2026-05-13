from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, Fusion, FusionQuery, Prefetch, Range

from .embedder import DenseEmbedder, SparseEmbedder


class Retriever:
    """Hybrid semantic and lexical retrieval from a Qdrant vector database.

    Combines dense embeddings (semantic similarity) and sparse BM25 embeddings (keyword matching)
    using Reciprocal Rank Fusion (RRF) to produce well-ranked clinical documents. Supports
    optional filtering by publication year for recency constraints.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        dense_embedder: DenseEmbedder,
        sparse_embedder: SparseEmbedder,
    ):
        self._client = client
        self._collection = collection_name
        self._dense = dense_embedder
        self._sparse = sparse_embedder

    def search(
        self,
        query: str,
        candidate_k: int = 80,
        min_year: int | None = None,
    ) -> list[dict]:
        """Retrieve candidate documents using hybrid semantic and lexical search.

        Encodes the query into both dense (Bioformer-16L) and sparse (BM25) embeddings,
        runs separate prefetch searches for each modality, fuses rankings via RRF, and
        returns the top candidate_k results with metadata. Optionally filters by year.

        Args:
            query: Clinical question or search string.
            candidate_k: Number of top results to return (default 80).
            min_year: Minimum publication year filter (inclusive); if None, no filter applied.

        Returns:
            List of dicts, each containing:
                - rank: Result ranking (1-indexed).
                - pmid: PubMed ID.
                - title: Document title.
                - abstract: Full abstract text.
                - mesh_category: MeSH disease category label.
                - mesh_terms: List of MeSH descriptor terms.
                - publication_types: List of publication type labels (e.g., "Journal Article").
                - year: Publication year.
                - journal: Journal name.
                - rrf_score: RRF fusion score (rounded to 4 decimals).
        """
        dense_vec = self._dense.encode([query])[0].tolist()
        sparse_vec = self._sparse.embed_query(query)

        year_filter = (
            Filter(must=[FieldCondition(key="year", range=Range(gte=min_year))])
            if min_year is not None
            else None
        )

        results = self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                Prefetch(query=dense_vec, using="dense", filter=year_filter, limit=candidate_k),
                Prefetch(query=sparse_vec, using="sparse", filter=year_filter, limit=candidate_k),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=candidate_k,
        )

        return [
            {
                "rank": i + 1,
                "pmid": hit.payload.get("pmid", ""),
                "title": hit.payload.get("title", ""),
                "abstract": hit.payload.get("abstract", ""),
                "mesh_category": hit.payload.get("mesh_category", ""),
                "mesh_terms": hit.payload.get("mesh_terms", []),
                "publication_types": hit.payload.get("publication_types", []),
                "year": hit.payload.get("year", 0),
                "journal": hit.payload.get("journal", ""),
                "rrf_score": round(hit.score, 4),
            }
            for i, hit in enumerate(results.points)
        ]
