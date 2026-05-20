"""Index PubMed corpus into Qdrant for the clinical RAG pipeline.

Reads the corpus produced by ``scripts/pubmed_ingestion.py``, encodes each
document with Bioformer-16L (dense) and BM25 (sparse), and upserts the
resulting points into a Qdrant collection configured for hybrid search.

The corpus file may be a JSON array or a newline-delimited JSON (NDJSON) file.

Usage::

    uv run python scripts/index_corpus.py [options]

Options:
    --corpus PATH         Corpus file to index        (default: data/raw/pubmed_bulk_corpus.json)
    --collection NAME     Qdrant collection name       (default: from configs/rag.yaml)
    --recreate            Delete and recreate the collection if it already exists
    --dense-batch INT     Batch size for dense encoder (default: from configs/rag.yaml)
    --upsert-batch INT    Batch size for Qdrant upsert (default: 128)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from src.config import load_config
from src.logging_config import get_logger
from src.rag.embedder import DenseEmbedder, SparseEmbedder
from src.rag.indexer import create_collection, index_corpus

logger = get_logger(__name__)


def _load_corpus(path: Path) -> list[dict]:
    """Load a corpus file that is either a JSON array or NDJSON."""
    text = path.read_text(encoding="utf-8").lstrip()
    if text.startswith("["):
        return json.loads(text)
    docs = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            docs.append(json.loads(line))
    return docs


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Index PubMed corpus into Qdrant for hybrid RAG retrieval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/raw/pubmed_bulk_corpus.json"),
        help="Path to the corpus JSON or NDJSON file",
    )
    p.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Qdrant collection name (default: vector_store.collection_name from rag.yaml)",
    )
    p.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the collection if it already exists",
    )
    p.add_argument(
        "--dense-batch",
        type=int,
        default=None,
        help="Batch size for the dense encoder (default: embedding.batch_size from rag.yaml)",
    )
    p.add_argument(
        "--upsert-batch",
        type=int,
        default=128,
        help="Number of points per Qdrant upsert request",
    )
    return p


def main() -> None:
    load_dotenv(override=True)

    args = _build_parser().parse_args()
    cfg = load_config("rag")

    corpus_path: Path = args.corpus
    collection_name: str = args.collection or cfg["vector_store"]["collection_name"]
    dense_batch: int = args.dense_batch or cfg["embedding"]["batch_size"]

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    logger.info("Loading corpus from %s", corpus_path)
    corpus = _load_corpus(corpus_path)
    logger.info("Loaded %d documents", len(corpus))

    logger.info("Initialising dense embedder: %s", cfg["embedding"]["model"])
    dense_embedder = DenseEmbedder(cfg["embedding"]["model"])
    logger.info("Dense embedder ready | dim=%d", dense_embedder.dim)

    logger.info("Initialising sparse embedder (BM25)")
    sparse_embedder = SparseEmbedder()

    qdrant = QdrantClient(
        host=os.getenv("QDRANT_HOST", cfg["vector_store"]["host"]),
        port=int(os.getenv("QDRANT_PORT", cfg["vector_store"]["port"])),
    )

    if qdrant.collection_exists(collection_name):
        if args.recreate:
            logger.info("--recreate set: deleting existing collection %r", collection_name)
        else:
            logger.warning(
                "Collection %r already exists. Use --recreate to rebuild it. Exiting.",
                collection_name,
            )
            return

    logger.info("Creating collection %r (dense_dim=%d)", collection_name, dense_embedder.dim)
    create_collection(qdrant, collection_name, dense_embedder.dim)

    count = index_corpus(
        client=qdrant,
        collection_name=collection_name,
        corpus=corpus,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        upsert_batch=args.upsert_batch,
        dense_batch=dense_batch,
    )

    logger.info("Done — %d points in collection %r", count, collection_name)


if __name__ == "__main__":
    main()
