from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from tqdm.auto import tqdm

from ..logging_config import get_logger
from .embedder import DenseEmbedder, SparseEmbedder

logger = get_logger(__name__)


def parse_mesh_terms(raw: list[str]) -> list[str]:
    """Normalize a list of raw MeSH entries.

    PubMed MeSH strings sometimes include markers such as a leading ``*`` to
    denote "major topic", and may include qualifiers after a slash. This helper
    extracts the base descriptor for each entry.

    Examples:
        >>> parse_mesh_terms(["*Diabetes Mellitus/therapy", "Hypertension"])
        ['Diabetes Mellitus', 'Hypertension']

    Args:
        raw: Raw MeSH strings (e.g., from a PubMed export). Elements may be
            empty, start with ``*``, and/or include qualifiers after ``/``.

    Returns:
        A list of cleaned MeSH descriptor names (no leading ``*``; qualifiers
        removed; whitespace trimmed). Empty entries are skipped.
    """
    return [entry.lstrip("*").split("/")[0].strip() for entry in raw if entry]


def create_collection(client: QdrantClient, name: str, dense_dim: int) -> None:
    """Create (or recreate) a hybrid Qdrant collection.

    The collection is configured with:
    - one dense vector named ``"dense"`` using cosine distance
    - one sparse vector named ``"sparse"``

    Note:
        If a collection with the same name already exists, it is deleted
        first.

    Args:
        client: Initialized Qdrant client.
        name: Collection name in Qdrant.
        dense_dim: Dimensionality of the dense embeddings (e.g., 384).

    Returns:
        None
    """
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": VectorParams(size=dense_dim, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )


def index_corpus(
    client: QdrantClient,
    collection_name: str,
    corpus: list[dict],
    dense_embedder: DenseEmbedder,
    sparse_embedder: SparseEmbedder,
    upsert_batch: int = 128,
    dense_batch: int = 32,
) -> int:
    """Encode and upsert a document corpus into Qdrant.

    Each document becomes a Qdrant point with:
    - vectors:
        - ``"dense"``: dense embedding of ``title + abstract``
        - ``"sparse"``: sparse embedding of ``title + abstract``
    - payload: document metadata used for filtering and citation.

    The function assumes the target collection already exists and is configured
    with matching vector names (see :func:`create_collection`).

    Expected corpus schema (per document):
        - ``pmid``: str or int (optional)
        - ``title``: str (optional)
        - ``abstract``: str (optional)
        - ``mesh_category``: str (optional)
        - ``mesh``: list[str] raw MeSH terms (optional)
        - ``publication_types``: list[str] (optional)
        - ``year``: int or str convertible to int (optional)
        - ``journal``: str (optional)

    Args:
        client: Initialized Qdrant client.
        collection_name: Name of the Qdrant collection to upsert into.
        corpus: List of document dictionaries. Missing fields are filled with
            safe defaults in the stored payload.
        dense_embedder: Dense embedder used to compute semantic vectors.
        sparse_embedder: Sparse embedder used to compute lexical vectors.
        upsert_batch: Number of points to upsert per request.
        dense_batch: Batch size passed to the dense encoder.

    Returns:
        The collection's point count after indexing completes.
    """
    # Use title + abstract for both dense and sparse so disease names in titles
    # can match clinical-note queries on either index.
    index_texts = [f"{p.get('title', '')} {p.get('abstract', '')}".strip() for p in corpus]

    logger.info("Encoding %d dense vectors", len(corpus))
    dense_vecs = dense_embedder.encode(index_texts, batch_size=dense_batch)

    logger.info("Encoding sparse vectors")
    sparse_vecs = sparse_embedder.embed_corpus(index_texts)

    for start in tqdm(range(0, len(corpus), upsert_batch), desc="Upserting"):
        batch = corpus[start : start + upsert_batch]
        d_batch = dense_vecs[start : start + upsert_batch]
        s_batch = sparse_vecs[start : start + upsert_batch]

        points = [
            PointStruct(
                id=start + j,
                vector={
                    "dense": d.tolist(),
                    "sparse": SparseVector(indices=s.indices.tolist(), values=s.values.tolist()),
                },
                payload={
                    "pmid": doc.get("pmid", ""),
                    "title": doc.get("title", ""),
                    "abstract": doc.get("abstract", ""),
                    "mesh_category": doc.get("mesh_category", ""),
                    "mesh_terms": parse_mesh_terms(doc.get("mesh", [])),
                    "publication_types": doc.get("publication_types", []),
                    "year": int(doc.get("year") or 0),
                    "journal": doc.get("journal", ""),
                },
            )
            for j, (doc, d, s) in enumerate(zip(batch, d_batch, s_batch))
        ]
        client.upsert(collection_name=collection_name, points=points)

    count = client.get_collection(collection_name).points_count
    logger.info("Indexed %d points into collection %r", count, collection_name)
    return count
