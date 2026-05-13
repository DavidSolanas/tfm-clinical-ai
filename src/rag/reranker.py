import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..logging_config import get_logger

logger = get_logger(__name__)


class MedCPTReranker:
    """Cross-encoder reranker for medical documents using MedCPT.

    MedCPT is a cross-encoder that scores the relevance of a query-document pair
    by passing both through a single encoder. Raw logits are mapped to [0, 1] via
    sigmoid for interpretability and consistency across batches.
    """

    MODEL_NAME = "ncbi/MedCPT-Cross-Encoder"

    def __init__(self, batch_size: int = 16, max_length: int = 512):
        """Initialize the MedCPT reranker.

        Args:
            batch_size: Number of query-document pairs to process per batch.
            max_length: Maximum token length for input sequences (longer sequences
                are truncated).
        """
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading MedCPT reranker on device: %s", self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self._model = (
            AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
            .to(self.device)
            .eval()
        )
        logger.info("MedCPT reranker ready")

    def _raw_scores(self, query: str, docs: list[dict]) -> list[float]:
        """Compute raw logit scores for query-document pairs.

        Args:
            query: The query string.
            docs: List of document dicts, each with 'title' and 'abstract' keys.

        Returns:
            List of raw logit scores (unbounded floats; higher = more relevant).
        """
        scores: list[float] = []
        texts = [f"{d['title']}. {d['abstract']}" for d in docs]
        for i in range(0, len(texts), self.batch_size):
            # Cross-encoder requires [query, doc] pairs
            pairs = [[query, t] for t in texts[i : i + self.batch_size]]
            encoded = self._tokenizer(
                pairs,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=self.max_length,
            ).to(self.device)
            with torch.no_grad():
                logits = self._model(**encoded).logits.squeeze(-1)
            scores.extend(logits.cpu().tolist())
        return scores

    def rerank(self, query: str, docs: list[dict]) -> list[dict]:
        """Rerank documents by relevance to a query using MedCPT cross-encoder.

        Computes cross-encoder scores for each query-document pair, applies sigmoid
        normalization, and returns documents sorted by descending relevance.

        Args:
            query: The query string.
            docs: List of document dicts with 'title' and 'abstract' keys.

        Returns:
            Reranked list of documents with added 'cross_encoder_score' field
            (float in [0, 1]), sorted by score descending.
        """
        logger.debug("Reranking %d docs with MedCPT", len(docs))
        raw = self._raw_scores(query, docs)

        # MedCPT outputs raw logits (unbounded floats; higher = more relevant).
        # Sigmoid maps them to [0, 1] via a fixed, monotonic function — unlike
        # min-max normalization, it does NOT re-anchor per batch, so a score of
        # 0.7 means the same thing across every query. This is required for the
        # abstention gate in ClinicalScorer (min_final_score threshold) to be
        # consistent: a batch of weak candidates will stay clustered below 0.5
        # and correctly trigger abstention instead of inflating to 1.0.
        scores = torch.sigmoid(torch.tensor(raw, dtype=torch.float32)).tolist()

        result = [{**d, "cross_encoder_score": round(s, 4)} for d, s in zip(docs, scores)]
        result.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
        return result
