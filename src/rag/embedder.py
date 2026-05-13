import numpy as np
import transformers
from fastembed import SparseEmbedding, SparseTextEmbedding
from qdrant_client.models import SparseVector
from sentence_transformers import SentenceTransformer, models

transformers.logging.set_verbosity_error()


class DenseEmbedder:
    """Mean-pooled semantic embeddings for biomedical text.

    Wraps a SentenceTransformer with mean-pooling to produce normalized embeddings suitable
    for vector similarity search (e.g., cosine distance in Qdrant). Uses mean-pooling strategy
    instead of [CLS] token to better capture multi-span clinical concepts.
    """

    def __init__(self, model_name: str, max_seq_length: int = 512):
        """Initialize dense embedder with a HuggingFace transformer.

        Args:
            model_name: HuggingFace model identifier (e.g., "bioformers/bioformer-16L").
            max_seq_length: Maximum sequence length (default 512). Longer texts are truncated.

        """
        word_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        pool_model = models.Pooling(
            word_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
        )
        self._model = SentenceTransformer(modules=[word_model, pool_model])

    @property
    def dim(self) -> int:
        """Embedding dimensionality (e.g., 384 for Bioformer-16L)."""
        return self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to normalized dense vectors.

        Args:
            texts: List of text strings (e.g., PubMed title + abstract).
            batch_size: Batch size for inference (default 32).

        Returns:
            (n, dim) float32 array of L2-normalized embeddings for cosine similarity.

        """
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)


class SparseEmbedder:
    """BM25-based sparse lexical embeddings for hybrid retrieval.

    Complements dense embeddings by matching keyword presence and TF-IDF weights. Used in
    Qdrant hybrid search (fusion via RRF) for queries requiring exact term matching.
    """

    def __init__(self, model_name: str = "Qdrant/bm25"):
        """Initialize sparse embedder with a BM25 model.

        Args:
            model_name: FastEmbed model identifier (default "Qdrant/bm25").

        """
        self._model = SparseTextEmbedding(model_name=model_name)

    def embed_corpus(self, texts: list[str]) -> list[SparseEmbedding]:
        """Encode a corpus of texts to sparse BM25 vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of SparseEmbedding objects (indices/values for non-zero terms).

        """
        return list(self._model.embed(texts))

    def embed_query(self, text: str) -> SparseVector:
        """Encode a single query to a sparse BM25 vector for Qdrant search.

        Args:
            text: Query text to embed.

        Returns:
            SparseVector with indices and values ready for Qdrant hybrid search.

        Raises:
            ValueError: If the model returns no sparse vector (unexpected error).

        """
        result = list(self._model.query_embed(text))
        if not result:
            raise ValueError(f"SparseTextEmbedding returned no sparse vector for query: {text!r}")

        s = result[0]
        return SparseVector(indices=s.indices.tolist(), values=s.values.tolist())
