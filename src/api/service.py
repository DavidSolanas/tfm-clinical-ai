"""Service layer wrapping :class:`~src.rag.pipeline.RAGPipeline` for the UI.

``ClinicalAssistantService`` is the single, UI-agnostic API the Gradio app talks
to. It owns the RAG pipeline (built once, on CPU) and the llama.cpp subprocess
(swapped between the base and fine-tuned GGUFs on demand), and exposes:

* ``answer(query, config)`` — runs one of the four ablation configs (plan §1.7);
* ``ensure_model(family)`` — reloads the GGUF on a Base↔FT switch (guards in
  :mod:`src.api.llama_server`);
* ``health()`` — drives the UI service-status banner;
* ``prewarm()`` — startup warm-up so the first real query is not cold.

The pipeline wiring is a **copy** of ``scripts/run_evaluation_phase._build_pipeline``
(plan §4): ``scripts/`` is not a package, so importing it would need ``sys.path``
hacks, and that file produced the locked Cap. 6 numbers and must never be edited.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from qdrant_client import QdrantClient

from src.api.llama_server import LlamaServerError, LlamaServerManager, ModelFamily
from src.config import PROJECT_ROOT, load_config
from src.llm_clients import OpenAICompatibleClient
from src.logging_config import get_logger
from src.rag.embedder import DenseEmbedder, SparseEmbedder
from src.rag.pipeline import RAGPipeline
from src.rag.reranker import MedCPTReranker
from src.rag.retriever import Retriever
from src.rag.scorer import ClinicalScorer

logger = get_logger(__name__)

Config = Literal["A", "B", "C", "D"]

# RF-08 four-config decomposition: model family × RAG on/off (plan §1.7).
_CONFIG_ROUTING: dict[Config, tuple[ModelFamily, bool]] = {
    "A": ("base", False),  # base + RAG off  → run_base on base server
    "B": ("ft", False),  # FT   + RAG off  → run_base on FT server
    "C": ("base", True),  # base + RAG on   → run on base server
    "D": ("ft", True),  # FT   + RAG on   → run on FT server
}

_CONFIG_LABELS: dict[Config, str] = {
    "A": "Base, no RAG",
    "B": "Fine-tuned, no RAG",
    "C": "Base + RAG",
    "D": "Fine-tuned + RAG",
}


@dataclass
class AssistantResult:
    """UI-agnostic result of one ``answer()`` call."""

    config: Config
    config_label: str
    model_family: ModelFamily
    use_rag: bool
    response: str | None
    abstained: bool = False
    reason: str | None = None
    docs: list[dict] = field(default_factory=list)
    citation_check: dict | None = None
    avg_final_score: float | None = None
    candidates_before_rerank: int | None = None
    wall_time_s: float = 0.0
    error: str | None = None


@dataclass
class HealthStatus:
    """Snapshot of dependency health for the UI status banner."""

    qdrant: bool
    llamacpp: bool
    model_family: ModelFamily | None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.qdrant and self.llamacpp


class ClinicalAssistantService:
    """Owns the RAG pipeline + llama.cpp server and serves the four configs."""

    def __init__(self, default_family: ModelFamily = "ft"):
        """Build the pipeline (CPU torch) and configure the server manager.

        Heavy models load lazily in :meth:`prewarm`; the constructor only reads
        config and wires objects so it stays cheap and import-safe.

        Args:
            default_family: Model family loaded at startup (plan default = FT).
        """
        self._rag_cfg = load_config("rag")
        self._repo_root = Path(PROJECT_ROOT)
        self._default_family = default_family
        self._server = LlamaServerManager(repo_root=self._repo_root)
        self._pipeline: RAGPipeline | None = None
        self._qdrant: QdrantClient | None = None

    # ------------------------------------------------------------------ build

    def _build_pipeline(self) -> RAGPipeline:
        """Construct the RAG pipeline.

        Copied from ``scripts/run_evaluation_phase._build_pipeline`` (plan §4),
        with the LLM client pointed at the service-managed server's base URL.
        """
        cfg = self._rag_cfg
        qdrant = QdrantClient(
            host=os.getenv("QDRANT_HOST", cfg["vector_store"]["host"]),
            port=int(os.getenv("QDRANT_PORT", cfg["vector_store"]["port"])),
        )
        self._qdrant = qdrant
        dense_embedder = DenseEmbedder(cfg["embedding"]["model"])
        sparse_embedder = SparseEmbedder()
        retriever = Retriever(
            client=qdrant,
            collection_name=cfg["vector_store"]["collection_name"],
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
        )
        reranker = MedCPTReranker(
            batch_size=cfg["retrieval"]["reranker"]["batch_size"],
            max_length=cfg["retrieval"]["reranker"]["max_length"],
        )
        scorer = ClinicalScorer(weights=cfg["retrieval"]["clinical_rerank"]["weights"])
        client = OpenAICompatibleClient(base_url=self._server.base_url)
        return RAGPipeline(
            retriever=retriever,
            reranker=reranker,
            scorer=scorer,
            client=client,
            candidate_k=cfg["retrieval"]["candidate_k"],
            final_k=cfg["retrieval"]["final_k"],
            min_year=cfg["retrieval"]["metadata_filters"]["min_year"],
            min_final_score=cfg["retrieval"]["clinical_rerank"]["min_final_score"],
            max_tokens=cfg["generation"]["max_tokens"],
            temperature=cfg["generation"]["temperature"],
        )

    # ---------------------------------------------------------------- lifecycle

    def prewarm(self) -> None:
        """Load the CPU torch models and start the default GGUF server.

        Called once at app startup so the first user query is not cold (plan §3).
        Raises through if the server cannot start, so the launcher fails loudly.
        """
        logger.info("Pre-warming service: building pipeline (CPU torch models)…")
        self._pipeline = self._build_pipeline()
        logger.info("Pipeline ready; starting default model server (%s)…", self._default_family)
        self._server.ensure(self._default_family)
        logger.info("Pre-warm complete (model_family=%s)", self._server.loaded_family)

    @property
    def pipeline(self) -> RAGPipeline:
        if self._pipeline is None:
            self._pipeline = self._build_pipeline()
        return self._pipeline

    def ensure_model(self, family: ModelFamily) -> None:
        """Ensure the requested GGUF is loaded, reloading on a Base↔FT switch."""
        self._server.ensure(family)

    # ------------------------------------------------------------------ health

    def health(self) -> HealthStatus:
        """Return current dependency health for the UI banner."""
        qdrant_ok = False
        try:
            client = self._qdrant or QdrantClient(
                host=os.getenv("QDRANT_HOST", self._rag_cfg["vector_store"]["host"]),
                port=int(os.getenv("QDRANT_PORT", self._rag_cfg["vector_store"]["port"])),
            )
            info = client.get_collection(
                collection_name=self._rag_cfg["vector_store"]["collection_name"]
            )
            qdrant_ok = (info.points_count or 0) > 0
        except Exception as exc:  # noqa: BLE001 (health probe must never raise)
            logger.debug("Qdrant health probe failed: %s", exc)

        llamacpp_ok = self._server.is_healthy()
        detail = ""
        if not llamacpp_ok:
            detail = "The evidence model isn't running. Start it with scripts/run_app.sh."
        elif not qdrant_ok:
            detail = "Qdrant is unavailable. Start it with `docker compose up -d qdrant`."
        return HealthStatus(
            qdrant=qdrant_ok,
            llamacpp=llamacpp_ok,
            model_family=self._server.loaded_family,
            detail=detail,
        )

    # ------------------------------------------------------------------- answer

    def answer(self, query: str, config: Config) -> AssistantResult:
        """Run one ablation config end-to-end and return a typed result.

        Maps ``config`` → ``(model_family, use_rag)``, reloads the GGUF if the
        model boundary is crossed, then dispatches to ``pipeline.run`` (RAG) or
        ``pipeline.run_base`` (no RAG). Never raises: dependency/server failures
        are captured into ``AssistantResult.error`` so the UI shows a banner.

        Args:
            query: The patient clinical note / transcription.
            config: One of ``"A"``, ``"B"``, ``"C"``, ``"D"``.

        Returns:
            An :class:`AssistantResult`.
        """
        family, use_rag = _CONFIG_ROUTING[config]
        label = _CONFIG_LABELS[config]
        result = AssistantResult(
            config=config,
            config_label=label,
            model_family=family,
            use_rag=use_rag,
            response=None,
        )

        query = (query or "").strip()
        if not query:
            result.error = "Enter a patient note before analyzing."
            return result

        start = time.monotonic()
        try:
            self.ensure_model(family)
        except LlamaServerError as exc:
            result.error = str(exc)
            result.wall_time_s = round(time.monotonic() - start, 2)
            return result

        try:
            if use_rag:
                out = self.pipeline.run(query)
                result.abstained = out["abstained"]
                result.candidates_before_rerank = out.get("candidates_before_rerank")
                if out["abstained"]:
                    result.reason = out.get("reason")
                    result.docs = out.get("top_candidates", []) or []
                else:
                    result.response = out["response"]
                    result.docs = out["docs"]
                    result.citation_check = out["citation_check"]
                    result.avg_final_score = out["avg_final_score"]
            else:
                result.response = self.pipeline.run_base(query)
        except Exception as exc:  # noqa: BLE001 (surface to UI, never crash app)
            logger.exception("answer() failed for config %s", config)
            result.error = (
                f"Generation failed: {exc}. Check that the model server and Qdrant are running."
            )
        result.wall_time_s = round(time.monotonic() - start, 2)
        return result

    def shutdown(self) -> None:
        """Stop the managed llama.cpp server (idempotent)."""
        self._server.stop()
