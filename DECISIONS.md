# Technical Decision Log

Chronological record of technical decisions made during the development of the thesis project.
Each entry documents what was decided, why, and which alternatives were considered.

---

## Template

### [DATE] -- Decision title

**Decision:** What was decided.

**Context:** Why this decision needed to be made.

**Alternatives considered:**
- Option A: description. Pros/cons.
- Option B: description. Pros/cons.

**Rationale:** Why this option was chosen over the others.

**Consequences:** What this decision implies for the rest of the project.

---

## Decisions


### 2026-05-21 — Evaluation methodology: ablation study orchestration, prompt fairness, abstention accounting, RAGAS judging

**Decision:** The ablation study comparing configurations A (Base) / B (FT-only) / C (RAG-only) / D (FT+RAG) is structured as follows: (1) **Phased execution** — Phase 1 evaluates A+C with the base GGUF loaded in llama.cpp, Phase 2 evaluates B+D after a manual swap to the FT GGUF (single-server hardware constraint). (2) **Prompt fairness** — configs A/B reuse the exact training template (`_SYSTEM_PROMPT` + `_USER_QUESTION` from `dataset_builder.py`) with an empty evidence section, rather than `RAGPipeline.run_base()`'s minimal prompt, so the FT model sees its in-distribution scaffold. (3) **Retrieval query for C/D** is the bare MTSamples transcription, matching what was used at training-data-build time. (4) **Abstention** is excluded from RAGAS denominators and reported as a separate `abstention_rate`; abstained samples count as 0 hallucinated PMIDs (correct — no response = no fabrication). (5) **RAGAS judge** is Claude Sonnet 4.6 (independent of the system under test); embeddings for AnswerRelevancy use OpenAI `text-embedding-3-small`. (6) **Custom metrics** added beyond RAGAS: `hallucinated_pmids_rate` (already supported by `verify_citations`) and `format_adherence_among_answered` (regex check for the 3-section structure or the no-evidence refusal template). (7) **Sample size** is 50 examples stratified by `medical_specialty`, seed=42, no-evidence training examples excluded, indices frozen to `data/processed/eval_sample_indices.json`. (8) **MLflow structure** is one parent run per phase + 2 nested children, plus one separate rollup run for cross-config deltas, all sharing a `tags.ablation_id` for grouping. (9) **Orchestration lives in `scripts/`** (`build_eval_sample.py`, `run_evaluation_phase.py --phase`, `run_evaluation_rollup.py`); `notebooks/06_evaluation.ipynb` is inspection-only. (10) **Plots are written to both** MLflow artifacts and `docs/thesis_latex/figures/eval_*.png` for direct LaTeX inclusion.

**Context:** The 4-config ablation is the central empirical contribution of the thesis. Getting the methodology wrong invalidates the study. Two specific risks shaped the design: (a) the FT model is trained with a rigid prompt template, so evaluating it under any other prompt format would conflate "FT changed model behavior" with "FT model degraded on OOD prompts"; (b) the dataset contains ~15% no-evidence training examples (`_apply_no_evidence_cap`), which means the headline narrative "FT-only hallucinates PMIDs" (from the 2026-05-07 decision) may not hold — the FT model may instead emit structured refusals. Format adherence becomes the real signal that "FT taught structured citation behavior" independent of whether PMIDs are fabricated.

**Alternatives considered:**

- **Single-script all-configs orchestration (auto-restart llama.cpp between configs) vs phased manual swap:** Auto-restart would require subprocess management of the llama.cpp server, port checks, and waiting for `/v1/models` to come back up — fragile across hours-long runs. Manual swap with `server_check.py` validation at script startup is one explicit human step, zero fragility.

- **Two llama.cpp servers on different ports (one base, one FT):** Cleanest API (no swap, no phasing) but doubles VRAM — 8B+8B with 4-bit quant ≈ 12GB on a 10GB card. Infeasible.

- **OpenAI/local-Qwen judge vs Claude Sonnet 4.6:** GPT-4o-mini saves ~€4 but published faithfulness-grading comparisons show it is noticeably less stable on clinical text. Local Qwen3-35B reuse (no API cost) would extend judge wall time to hours and reintroduce judge instability. Opus 4.7 (~€20) is overkill for grading. Sonnet 4.6 at ~€5 is the methodological + cost sweet spot.

- **`RAGPipeline.run_base()` vs training-template prompt for A/B:** `run_base` emits `"CLINICAL QUERY: ... EVIDENCE-BASED RESPONSE:"` — a scaffold the FT model has never seen. Using it would produce OOD format collapse in config B, conflating prompt mismatch with FT behavior. Rejected; A/B build a training-template prompt with `RETRIEVED PUBMED EVIDENCE:\n(none)`.

- **Abstention scoring strategies:** Treating abstained samples as `faithfulness=1.0, answer_relevancy=0` invents scores RAGAS didn't compute (methodologically shaky). Treating them as `0` penalizes correct behavior (abstention is the design intent when evidence is insufficient). Excluding them from the RAGAS denominator and reporting `abstention_rate` separately is the only honest accounting.

- **ContextRecall gold = teacher response (cached-retrieval bias) vs skip ContextRecall vs build hand-curated gold:** The teacher response was conditioned on a different retrieved set than live retrieval will produce, biasing ContextRecall downward in absolute terms. Skipping the metric drops a RAGAS dimension. Hand-curating gold for 20+ queries is an annotation project the thesis schedule doesn't fit. Chosen: keep the metric, interpret comparatively (the bias cancels across C vs D since they share gold), document the ceiling caveat in Cap. 6.

- **Filesystem manifest (`evaluation_runs/<ablation_id>/`) vs MLflow-only artifact store:** A filesystem index creates two sources of truth that can diverge. MLflow already stores per-run artifacts and supports tag-based queries (`tags.ablation_id`), so the rollup script can find phase parents and walk to their children via `tags.mlflow.parentRunId` with no filesystem coordination.

- **Sample size 50 vs 100 vs full test split:** 50 fits in ~2h per phase (4h total), tolerable RAGAS variance for thesis-grade comparison reported with bootstrap 95% CIs. 100 doubles wall time and crash risk. Full test split (~250) risks GPU/llama.cpp instability over ~11h.

- **Including no-evidence training examples in the eval sample vs excluding:** Including them muddies answer-quality metrics with non-answerable cases (the teacher's gold is itself a refusal). Excluding focuses metrics on the answering behavior the ablation actually studies; the system's refusal/abstention behavior is already captured by `abstention_rate` and `format_adherence`.

**Rationale:** Every component of the methodology is justifiable in isolation, but the coupling between them matters: phased execution makes the per-phase MLflow parent structure natural (no run kept open across the swap); training-template prompts for A/B make `format_adherence` a meaningful cross-config metric; abstention exclusion + `abstention_rate` makes RAGAS scores interpretable as "when the system answered, how good was it?" rather than mixing two questions; live-retrieval ContextRecall is unbiased *as a delta* across C vs D, which is what the ablation cares about. The system-under-test code (`RAGPipeline`, `dataset_builder`) is the source of truth for prompts, generation params, and citation verification — the evaluation code imports from it rather than duplicating, eliminating drift.

**Consequences:**

- New module `src/evaluation/` with focused files: `sample.py`, `prompts.py`, `runner.py`, `metrics.py`, `aggregation.py`, `mlflow_logging.py`, `server_check.py`. No fat modules; each file has one responsibility tied to a locked decision.
- New scripts: `scripts/build_eval_sample.py`, `scripts/run_evaluation_phase.py`, `scripts/run_evaluation_rollup.py`. The notebook `06_evaluation.ipynb` is rewritten as inspection-only (loads MLflow artifacts, draws comparison plots, drafts Cap. 6 narrative).
- New dependencies likely required: `ragas` (version-pinned), `openai` (for RAGAS embeddings), `scipy` (bootstrap CIs), `matplotlib`, `pandas`. Verify against `pyproject.toml`.
- `configs/evaluation.yaml` is unchanged in spirit; the new code reads `num_questions`, `seed`, `mlflow_experiment` from it.
- Sample indices file `data/processed/eval_sample_indices.json` becomes a versioned thesis artifact (commit it).
- Plot outputs land in `docs/thesis_latex/figures/eval_*.png`; the thesis Cap. 6 `\includegraphics` commands reference them directly.
- `CONTEXT.md` added at repo root to glossary the terms that emerged during this design (Abstention vs Refusal, Ablation configuration A/B/C/D, Phase 1/2, Eval sample, Ablation ID, Judge vs Teacher) — terms previously implicit in code and easy to confuse in thesis prose.
- Full implementation spec: `plans/06_evaluation_implementation_plan.md` (signatures, CLI workflow, failure modes, sanity-check thresholds).
- Open implementation questions deferred to coding: exact llama.cpp model ID strings reported by `/v1/models`, RAGAS version to pin, bootstrap CI implementation choice, plot styling to match thesis figures.

---

### 2026-05-07 — Fine-tuning strategy: behavioral alignment via synthetic per-note data generation

**Decision:** Fine-tune Llama 3.1 8B Instruct with QLoRA on a single task — given a patient clinical transcription (MTSamples) + a generic evidence question + retrieved PubMed abstracts, generate a grounded structured response citing only provided PMIDs. Training examples are generated synthetically by a local teacher LLM (Qwen3-35B-A3B served via llama.cpp; `--teacher local`) using real per-note retrieved context. Training is **predominantly** with-context (≥85%), with up to 15% no-evidence examples where the teacher emits a structured refusal — see `max_no_evidence_ratio=0.15` in `dataset_builder.py`. The goal is behavioral alignment, not knowledge injection.

*(Updated 2026-05-20: teacher changed from Claude Opus 4.7 to local Qwen3-35B-A3B-UD-Q5\_K\_XL. Rationale: local inference eliminates API cost at ~2K calls and avoids network dependency; thinking-block stripping already implemented in `dataset_builder.py` (`_strip_thinking`). The `--teacher local` flag passes an OpenAI-compatible endpoint at `http://localhost:8001/v1`.)*

**Context:** The ablation study requires four configurations (Base / FT-only / RAG-only / FT+RAG). For the comparison to be meaningful, fine-tuning must teach a *behavior* the base model does not already exhibit reliably — specifically, the discipline of citing PMIDs from provided context and structuring responses as Recommendation / Evidence basis / Uncertainty. The base Llama 3.1 8B Instruct already has clinical domain knowledge; injecting more via SFT is neither feasible at this scale nor the thesis goal. The behavioral target is: "when retrieved context with PMIDs is provided, always cite them; structure the response consistently."

The critical implication is that the fine-tuned model learns a strong PMID-citation pattern *and* a structured refusal template (the no-evidence response). When deployed without context (FT-only config B), the model may either hallucinate PMIDs (if the with-context behavior dominates) or emit the refusal template out-of-distribution (if the no-evidence behavior generalizes). Which behavior wins is an empirical question for the 06_evaluation ablation. The `hallucinated_pmids` rate and the new `format_adherence` rate (see 2026-05-21 entry) jointly characterize what FT actually taught — and which behavior separates configs B and D.

**Alternatives considered:**

- **Multi-task fine-tuning (summarization + classification + keyword extraction + RAG):** Originally planned, influenced by multi-task medical SFT literature. Rejected because: (1) the downstream use case is a single chatbot query, not multi-task inference; (2) classification and keyword tasks would dilute the training signal for the main task; (3) LIMA (Zhou et al., 2023) shows 1K curated single-task examples outperform 52K+ noisy multi-task examples for behavioral alignment.

- **JSON structured output vs structured free text:** JSON allows Gradio parsing but an 8B model under clinical reasoning produces malformed JSON often enough to break the parser, and hallucinated PMIDs would fill `evidence_used` fields in FT-only mode. Structured free text (1. Recommendation / 2. Evidence basis / 3. Uncertainty) is regex-parseable, always well-formed, and works identically for all query types.

- **Derived questions from `sample_name` vs generic question:** Derived would be "What are evidence-based treatments for Kawasaki Disease?" Generic is "Based on this patient's clinical presentation, what evidence-based treatments are recommended?" Derived queries require cleaning messy `sample_name` values (procedure-only entries, administrative notes, enumerated suffixes like "Abscess I&D", "1-year-old Exam") that break templating. The generic format is simpler and works for all note types. Clinical specificity is carried by the transcription content, not the question wording.

- **Per-specialty vs per-note retrieved context:** Per-specialty context (same PubMed pool for all cardiology notes) is too weak — it teaches the model that generic specialty background is acceptable evidence. Per-note context retrieves abstracts specific to the individual patient case, producing training examples that mirror the actual inference scenario. Feasible because the RAG pipeline is working before training data is built.

- **With-context-only vs mixed (with + ≤15% no-evidence) training:** Strict with-context-only would maximize PMID-hallucination signal in config B but would also leave the model with no learned refusal behavior — risky for production. The chosen mix (≤15% no-evidence) teaches refusal as a safety floor, at the cost of weakening the hallucination demonstration. The trade-off is acknowledged: which behavior dominates in config B is now an empirical finding of the ablation rather than a predicted outcome.

- **Existing clinical instruction datasets (MedAlpaca, PubMedQA, ClinicalCamel) vs synthetic generation from MTSamples:** Existing datasets target general medical QA, not the specific transcription → evidence → grounded response format. Synthetic generation from MTSamples + Claude produces examples in the exact format the system uses at inference, which is the critical alignment property.

**Rationale:** LIMA principle (Zhou et al., 2023): for instruction fine-tuning, quality and format consistency dominate quantity. ~2,000 examples of consistent, high-quality transcription → grounded response pairs are sufficient for behavioral alignment. Generating them synthetically via a capable local teacher (Qwen3-35B) using real per-note retrieved context ensures: (1) PMID citations in training responses are always real and retrievable; (2) the training distribution matches inference exactly; (3) the ablation story is clean and quantitatively measurable via the existing citation verifier.

**Consequences:**

- `src/training/dataset_builder.py`: drives the full data generation pipeline (MTSamples load → RAG retrieve → quality filter → teacher LLM generate → cache → DatasetDict). Cache is written to disk after each call to allow resumable generation.
- Quality filter: notes where RAG abstains or `min_doc_score < 0.40` are excluded. Note: 0.40 is the *dataset-build* threshold (stricter than the 0.35 inference threshold in `pipeline.py`) — training examples must rest on high-confidence evidence. Notes where teacher hallucinates PMIDs are also skipped. Expecting ~1,500–2,400 usable training examples after filtering.
- Split happens at the medical-specialty level (stratified) before generating any examples, preventing leakage. Implemented in `src/data/mtsamples.py::split_by_medical_specialty`.
- `src/training/finetune.py`: Unsloth + QLoRA (rank=16, alpha=32) + SFTTrainer, MLflow logging via `report_to="mlflow"`. Adapter saved to `checkpoints/llama-3.1-8b-mtsamples-qlora/`.
- Evaluation metric added to `06_evaluation.ipynb`: `hallucinated_pmids` rate per config. Expected result: B (FT-only) high hallucination, D (FT+RAG) near-zero.
- No-evidence examples capped at 15% of the total (`max_no_evidence_ratio=0.15`) to prevent imbalance.

---

### 2026-05-06 — RAG pipeline upgrade: hybrid search, MedCPT reranking, ClinicalScorer, abstention gating, citation verification

**Decision:** Replace the original single-stage dense-only retrieval pipeline with a five-layer architecture: (1) hybrid dense + BM25 search fused with Reciprocal Rank Fusion (RRF), (2) MedCPT cross-encoder reranking, (3) ClinicalScorer weighted combination of five signals, (4) abstention gating, (5) citation verification.

**Context:** Qualitative evaluation of the initial RAG pipeline (`notebooks/05_rag_pipeline.ipynb`) revealed three systematic failure modes: (a) retrieved documents were semantically adjacent but clinically irrelevant (e.g. CRKP bacteremia query returning generic gram-negative sepsis papers that lack the specific antimicrobial evidence needed); (b) the LLM cited PMIDs not present in the retrieved set (hallucinated citations); (c) there was no quality gate — the generator answered even when no relevant document was retrieved. The original pipeline retrieved wide (candidate_k determined by embedding similarity alone) and had no downstream filter, so all failures propagated directly into the generated response.

**Alternatives considered:**

- **SPLADE sparse encoder vs BM25 (FastEmbed `Qdrant/bm25`):** SPLADE learns vocabulary expansion and is theoretically better for recall on paraphrased queries. Rejected because: (1) no production-ready biomedical SPLADE checkpoint exists; (2) the primary clinical failure mode is exact-term miss on rare identifiers (CRKP, PSA, H. pylori, DOACs) where BM25's IDF naturally upweights low-frequency clinical strings; (3) BM25 via FastEmbed requires no additional model at inference time and uses Qdrant's native sparse vector support.

- **PICO query parser (query-expansion layer):** An NLP layer to extract Population / Intervention / Comparison / Outcome from the user query. Originally in the plan. Rejected because clinical queries in this system are user-generated from patient history records and symptoms — they rarely conform to PICO structure and structured extraction would introduce errors. ClinicalScorer achieves the same signal (publication type relevance, MeSH overlap, title keyword match) using raw query token overlap without requiring structured extraction.

- **Cross-encoder alternatives (MonoT5, MS-MARCO fine-tuned):** General-purpose rerankers. Rejected in favour of `ncbi/MedCPT-Cross-Encoder`: purpose-trained on biomedical query–article pairs using contrastive learning, directly optimised for the retrieval task performed at inference time (clinical query vs PubMed abstract).

- **Score-threshold retrieval cutoff vs abstention gating:** Hard score cutoff (drop any doc below threshold from context) vs returning `abstained=True` when fewer than 2 docs reach `min_final_score=0.35`. Chose explicit abstention: the generator receives no context and returns a structured refusal, which is logged, measurable in evaluation, and clearly communicated to the user. A silent hard cutoff would generate from zero context without signalling the gap.

**Rationale:**

- **Wide retrieval + strict reranking** (`candidate_k=80 → final_k=6`): Dense retrieval alone conflates semantic similarity with clinical relevance. Retrieving 80 candidates and reranking to 6 decouples the "find related literature" step from the "select clinically useful evidence" step.
- **RRF fusion**: Dense scores (cosine, bounded) and sparse scores (BM25, unbounded) are not directly comparable — raw score combination would require tuning a weight that generalises poorly. RRF combines by position rank (`1/(k+rank)`, k=60), bypassing score-scale incompatibility entirely. Implemented natively via Qdrant `FusionQuery(Fusion.RRF)` with `Prefetch` for each vector space.
- **BM25 for clinical exact-term matching**: Clinical literature retrieval has a long-tail vocabulary problem — multi-drug-resistant organism names, procedure acronyms, specific drug classes. Dense retrieval returns semantically similar documents; BM25 returns documents that literally contain the query terms. Together they cover complementary failure modes.
- **ClinicalScorer weights**: cross_encoder (0.60) dominates as the most direct relevance signal; mesh_overlap (0.15) ensures topical alignment with the corpus structure; publication_type (0.12) up-weights systematic reviews and RCTs over case reports; recency (0.10) penalises outdated evidence more strongly than the initial estimate; title_keyword (0.03) adds a light exact-match bonus. Weights sum to 1.0, are interpretable, configurable in `configs/rag.yaml`, and match the clinical evidence hierarchy (meta-analyses > RCTs > observational). *(Updated 2026-05-20: mesh_overlap revised from 0.20→0.15, recency from 0.05→0.10 after calibration; see `configs/rag.yaml` as the authoritative source.)*
- **Citation verification**: Regex extraction of PMIDs from the generated response, checked against the retrieved set. Hallucinated citations are flagged in the result dict and logged to MLflow. This makes hallucination rate a directly measurable metric in `06_evaluation.ipynb` without requiring a separate NLI-based faithfulness model.

**Consequences:**

- **Re-ingestion required**: `publication_types` (MEDLINE `PT` field) was fetched in `notebooks/02_pubmed_ingestion.ipynb` but discarded in the output mapping. One-line fix (`"publication_types": record.get("PT", [])`). Must re-run ingestion to populate `data/raw/pubmed_bulk_corpus.json` with this field.
- **Re-indexing required**: Qdrant collection schema changes from a single dense vector space to named vector spaces (`"dense"`: VectorParams 384d cosine, `"sparse"`: SparseVectorParams). Existing collection must be dropped and recreated. BM25 model must be `fit()` on the full corpus at index time (IDF computation) before indexing, then `query_embed()` used at query time (TF=1).
- **Payload enrichment**: `mesh_terms` (stripped of asterisks and qualifiers from raw `mesh` field), `publication_types`, and `year` (int) added to every Qdrant point payload to support ClinicalScorer signals without additional lookups.
- **Config update**: `configs/rag.yaml` gains `hybrid_search`, `reranker`, and `clinical_rerank` sections; `candidate_k` and `final_k` replace the single `top_k`; `min_final_score` added.
- **MLflow logging update**: Run `hybrid_rag_qualitative_demo_v2` logs abstention flags, per-query avg/top1 final and CE scores, citation_ok, and hallucinated PMID counts — enabling automated comparison across ablation configurations in `06_evaluation.ipynb`.
- **Abstention is a first-class outcome**: Downstream evaluation must account for abstained queries separately (they are neither correct nor incorrect answers). Document as a scope boundary in Cap. 6.

---

### 2026-05-05 — Embedding model revised: Bioformer-16L selected over PubMedBERT based on empirical benchmark

**Decision:** Replace `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` with `bioformers/bioformer-16L` as the RAG embedding model.

**Context:** The initial model selection chose PubMedBERT based on domain alignment theory: pre-trained exclusively on PubMed abstracts, same domain as the retrieval corpus (Gu et al., 2021). After running the full proxy benchmark (`same_category@k`, `MRR@10`) with stratified random sampling and bootstrap confidence intervals, Bioformer-16L outperformed PubMedBERT in every metric across all 24 MeSH categories. The domain alignment argument was also weakened by a previously unacknowledged query-document mismatch: production queries are clinical notes (MTSamples), not abstracts.

**Evidence:**
- same_category@1: Bioformer 0.395 vs BiomedBERT 0.352 (+0.043)
- same_category@5: Bioformer 0.316 vs BiomedBERT 0.269 (+0.047)
- MRR@10: Bioformer 0.530 vs BiomedBERT 0.484 (+0.046); 95% bootstrap CIs do not overlap
- Bioformer wins 24/24 MeSH categories after stratified random sampling
- Confusion analysis: all cross-category errors in both models are clinically sensible (Musculoskeletal↔Wounds, Mental↔Nervous System, etc.) — failure mode is MeSH label noise, not model error, and is identical between models

**Alternatives considered:**
- **Keep PubMedBERT:** The domain alignment argument (Gu et al., 2021) was the original justification. Still valid in principle, but assumes query ≈ abstract. Production queries are clinical notes — the alignment assumption doesn't hold.
- **Switch to Bioformer-16L (chosen):** 24/24 category win is structural, not noise. Broader PMC pretraining may help with clinical query understanding. 2x faster embedding, 384d vs 768d (half Qdrant memory). Empirical evidence + practical wins with no identified downside.

**Rationale:** When empirical evidence consistently contradicts a theoretical prior, and the theoretical prior rested on an assumption (query = abstract) that doesn't hold in production, the empirical result takes precedence. The confusion analysis confirms Bioformer makes the same types of errors as PubMedBERT — it doesn't introduce new failure modes.

**Consequences:**
- `configs/rag.yaml`: `embedding_model` field must be updated to `bioformers/bioformer-16L`
- Cap. 3 and Cap. 5 thesis sections: update embedding model name, dimension (384), and justification
- Qdrant collection: embedding dimension changes from 768 to 384 — collection must be recreated on next ingestion run

---

### 2026-04-28 — Broad MeSH-anchored PubMed corpus (final)

**Decision:** Build a single broad reference corpus by querying PubMed once per top-level MeSH descriptor in the disease tree (Tree C, ~20 entries), the mental disorders tree (F03), and key procedure trees (Surgical Procedures, Diagnostic Techniques, Therapeutics). Each query targets `[MeSH Major Topic]` (descriptor must be the central focus, not incidentally indexed), filtered by recency (≥ 2015), publication-type whitelist (Review / Guideline / RCT / Meta-Analysis / Clinical Trial / Systematic Review), and publication-type blacklist (Editorial / Letter / Comment / News / Biography / Personal Narrative). Result: a flat `pubmed_bulk_corpus.json` of ~320K abstracts, each tagged with `mesh_category`. Supersedes the 2026-04-20 "Bulk Ingestion Strategy" and the (same-day) per-`sample_name` attempt.

**Context:** Two prior strategies failed for the same underlying reason — PubMed indexes literature by **controlled vocabulary** (MeSH), not by free-text strings drawn from MTSamples. (1) Per-specialty queries (`query = spec`, e.g. `"Allergy / Immunology"`) returned specialty-meta papers (training, ethics, journal year-in-review, country reports) that don't represent disease-specific clinical evidence; ~80% of golden-set candidates were irrelevant. (2) Per-`sample_name` queries (e.g. `"AVM with Hemorrhage"`, `"Abdominal Abscess I"`, `"1-year-old Exam"`) returned 0 hits because case-instance titles aren't strings PubMed indexes — and loosening the query reintroduces the meta-paper noise.

The fundamental issue is **trying to align corpus construction with case-level queries via free-text matching**. Standard practice in clinical RAG (Almanac, Clinfo.ai, MedRAG benchmark) is to **decouple corpus from queries**: build a broad MeSH-anchored knowledge base once, and let dense retrieval (BioBERT + Qdrant) handle per-query relevance.

**Alternatives considered:**
- **Per-MTSamples-row free-text queries:** rejected on 2026-04-20 (API load, redundancy).
- **Per-unique-`sample_name` queries with PT filter:** attempted 2026-04-28; rejected — case titles are not searchable strings.
- **MeSH curation per specialty (manual mapping):** ~40 specialty → MeSH-list mappings. Rejected: high curation cost, brittle maintenance, indirect alignment with the MeSH tree.
- **Concept extraction from note text via scispaCy + UMLS:** considered as an enhancement on top of broad ingestion, deferred — the broad MeSH corpus + dense retriever handles the same need without an extra dependency. Revisit if retrieval precision plateaus.
- **Broad MeSH-tree-driven corpus (chosen):** ~25 queries (one per top-level MeSH category), each tightly filtered by `[MeSH Major Topic]` and PT whitelist. Produces a clinically curated, stratifiable corpus that aligns with how peer clinical RAG systems are built.

**Rationale:** This approach (a) eliminates noise from specialty-meta and document-template queries, (b) leverages PubMed's controlled vocabulary directly instead of fighting it with free-text matching, (c) decouples corpus construction from query construction (so query-side improvements don't require re-ingestion), and (d) produces a methodologically defensible artifact for the thesis ("broad MeSH-anchored corpus + dense retrieval", standard in clinical RAG). The `mesh_category` tag on every abstract preserves stratification capability for analysis.

**Consequences:**
- `pubmed_bulk_corpus.json` schema changes from `{specialty: [papers]}` to a **flat list of papers**, each with a `mesh_category` field. Every consumer of this file must be updated.
- `notebooks/02_pubmed_ingestion.ipynb` Section 6 holds the new ingestion code; Section 5 (v1) and the per-`sample_name` cells (v2) are superseded but left in place as historical context.
- `pubmed_corpus_by_concept.json` (a v2 by-product) is no longer produced; safe to delete.
- `notebooks/03a_golden_dataset.ipynb` must be refactored: the current "per-specialty candidate pool" assumption no longer holds. New design: BM25 over the full corpus to select top-N candidates per query, optionally filtered by `mesh_category` derived from the MTSamples specialty.
- `notebooks/03b_embedding_benchmark.ipynb` Section-6 sanity check must switch from `specialty` labels to `mesh_category` labels.
- The 3 already-labeled queries in `data/raw/golden_retrieval_labels.json` are obsolete (different candidate pools); discard before re-annotation.
- New ingestion volume is ~320K abstracts — verify Qdrant collection sizing (384d × 320K ≈ ~500MB embeddings). *(2026-05-21: actual final corpus is ~320K in `data/raw/pubmed_bulk_corpus.json`. The file `data/raw/pubmed_bulk_corpus_180k.json` is an older intermediate from a partial ingestion run and is safe to delete.)*

---

### 2026-04-28 — PubMed ingestion: per-`sample_name` queries with publication-type filtering [SUPERSEDED 2026-04-28 — see "Broad MeSH-anchored PubMed corpus" above]

**Decision:** Replace per-specialty bulk PubMed queries (`query = spec`) with per-unique-`sample_name` clinical-concept queries, filtered by Publication Type (whitelist Reviews / Guidelines / RCTs / Meta-Analyses / Clinical Trials / Systematic Reviews; blacklist Editorial / Letter / Comment / News / Biography / Personal Narrative). Adds a recency filter (≥ 2015) and a looser fallback query when the strict form returns < 5 hits. Supersedes the query construction from the 2026-04-20 "Bulk Ingestion Strategy" decision.

**Context:** During golden-set annotation, a sample of 13 queries across 11 specialties showed that ~80% had zero relevant candidates because the per-specialty corpus was dominated by specialty-meta papers (fellowship training, ethics, leadership, journal year highlights, country-level reviews). The PubMed query was literally the specialty name (e.g. `"Allergy / Immunology"`), and PubMed's relevance sort surfaces papers that mention the specialty in the title — biasing strongly toward meta-content rather than disease-specific clinical evidence. Continuing annotation on this corpus would yield a gold set with mostly empty positive sets, making P@k / Recall@k / nDCG meaningless.

**Alternatives considered:**
- **Per-MTSamples-row queries (~5,000 calls):** Originally rejected on 2026-04-20 for API load and redundancy. Still rejected — most rows share `sample_name`, so concept-level dedup is sufficient.
- **Per-specialty + curated MeSH-term lists:** Build a 40-entry mapping (e.g. `Allergy/Immunology → ["Asthma"[MeSH] OR "Anaphylaxis"[MeSH] OR ...]`). Pros: still ~40 queries. Cons: requires manual curation per specialty; alignment with MTSamples queries is indirect.
- **Per-unique-`sample_name` + PT filter (chosen):** A few hundred unique cleaned names (suffix-stripped: "Kawasaki Disease - Discharge Summary" → "Kawasaki Disease"). Each query is `("<concept>"[Title/Abstract] OR "<concept>"[MeSH Terms]) AND <recency> AND <PT_whitelist> NOT <PT_blacklist>`. Aligns the corpus directly with the gold-set queries (which are MTSamples descriptions tied to the same `sample_name`).

**Rationale:** The gold-set queries are derived from MTSamples notes, so the corpus should be retrieved using terms drawn from those same notes. `sample_name` has the highest signal-to-noise of the available fields — it encodes disease + procedure. The PT whitelist eliminates the meta-paper dominance directly: Editorials, Letters, Comments, and "introduction to specialty" reviews produced almost all of the noise observed during annotation. Title/Abstract + MeSH dual matching leverages PubMed's auto-mapping while still catching un-coded papers.

**Consequences:**
- New cell (Section 6) in `notebooks/02_pubmed_ingestion.ipynb` produces two outputs: `data/raw/pubmed_corpus_by_concept.json` (concept → papers, used for per-MTSamples-note retrieval) and `data/raw/pubmed_bulk_corpus.json` (specialty → deduped papers, replacing the v1 file).
- BM25 candidate generation in `notebooks/03a_golden_dataset.ipynb` must be re-run on the new corpus before annotation resumes.
- The 3 already-labeled queries in `data/raw/golden_retrieval_labels.json` should be discarded after re-ingestion (different candidate pools → labels invalid).
- Annotation methodology is unchanged: criteria + multi-pass resumable workflow (`scripts/annotation/dump_batch.py`, `save_labels.py`, `apply_labels.py`) remain valid.
- `MIN_RESULTS_FALLBACK = 5` and the loose fallback query may still surface some Editorials/Letters via the blacklist-only path; spot-check empty-or-fallback concepts after the run.

---

### 2026-04-28 — Embedding benchmark: two-tier evaluation strategy [PARTIALLY SUPERSEDED 2026-05-05 — Tier 2 deferred; selection rests on Tier 1 proxy]

**Decision:** Evaluate embedding models using two tiers: (1) MeSH-category-proxy sanity check and (2) golden dataset with standard IR metrics as the primary benchmark.

*(Updated 2026-05-21: Tier 2 was deferred — no `03a_golden_dataset.ipynb` was built, no `golden_retrieval_labels.json` exists. The 2026-05-05 Bioformer-vs-PubMedBERT decision was made on Tier 1 (proxy) evidence alone: 24/24 MeSH-category wins, MRR@10 deltas with non-overlapping bootstrap CIs, and confusion analysis. The 2026-04-28 "Golden retrieval dataset" entry below is also superseded. Cap. 5.2 will document embedding selection from Tier 1 only and acknowledge the absence of a golden IR benchmark as a scope limitation in Cap. 6 — standard caveat for thesis-scale work where annotation cost outweighs the marginal evidential value once a clear proxy-level winner is identified.)*

**Context:** The original plan was a simple embedding comparison. A pure category proxy measures embedding-space organisation, not clinical retrieval relevance. A rigorous thesis needs quantitative IR evidence for the model selection decision in Cap. 5. *(Updated 2026-04-28: Tier-1 proxy was originally specialty-based, derived from the per-specialty corpus. Following the broad MeSH-anchored corpus pivot, the proxy label switches to `mesh_category` — same methodology, different label source.)*

**Alternatives considered:**
- **Category proxy only:** No annotation required, fast. Measures same-category@k and MRR@10. Rejected as primary benchmark because a model memorising category vocabulary would score well without useful clinical retrieval.
- **Golden dataset only:** Blocks all results until annotation is complete.
- **Two-tier (chosen):** Proxy runs immediately and catches obviously broken models. Golden dataset provides the primary result once annotated.

**Rationale:** The two-tier design separates a fast sanity check (is the embedding space structured by clinical topic?) from the actual benchmark (does the model retrieve clinically relevant abstracts for real clinical queries?). Only the second tier is cited in the thesis as evidence.

**Consequences (as originally planned — Tier 2 not executed):**
- ~~`03a_golden_dataset.ipynb` is annotation-creation only; all benchmark logic lives in `03b`.~~
- ~~`03b_embedding_benchmark.ipynb` Section 6 = proxy; Section 10 = golden evaluation.~~
- ~~Annotation of ~3K relevance judgments (BM25-pooled, ~25 per query) is required before Cap. 5.2 can be written.~~
- **Actual outcome:** consolidated into a single `03_embedding_benchmark.ipynb` containing only Tier 1 (proxy). Golden-dataset annotation was deferred and not produced. Cap. 5.2 cites the proxy benchmark; Cap. 6 documents the absence of golden IR evaluation as a scope limitation.

---

### 2026-04-28 — SentenceTransformer with explicit pooling modules

**Decision:** Use `SentenceTransformer(modules=[Transformer, Pooling])` instead of `SentenceTransformer(model_name)` for embedding generation.

**Context:** Neither BiomedBERT nor Bioformer-16L was released as a sentence-embedding model — both are MLM-pretrained. `SentenceTransformer(model_name)` infers pooling from config (behaviour has changed across ST versions) and emits `UNEXPECTED key` warnings from the MLM head.

**Alternatives considered:**
- `SentenceTransformer(model_name)`: Shorter but pooling strategy inferred at runtime, not documented in code. Warnings pollute output.
- `AutoModel` + manual `mean_pool()`: Most explicit, no ST dependency, but more boilerplate code.
- `SentenceTransformer(modules=[Transformer, Pooling])` (chosen): Pooling strategy is a visible code argument (`pooling_mode_mean_tokens=True`). L2 normalisation declared with `normalize_embeddings=True`. Reuses existing project dependency. Warnings suppressed via `transformers.logging.set_verbosity_error()`.

**Rationale:** The thesis must document the embedding methodology precisely. The ST modules API makes every choice auditable in the code rather than inferred from library defaults.

**Consequences:**
- `embed_texts()` calls `make_st_model()` which builds the pipeline explicitly.
- Device management delegated to ST; `torch` not imported directly in the embedding cell.
- `sentence-transformers>=3.0` remains the only embedding dependency.

---

### 2026-04-28 — Golden retrieval dataset: MTSamples descriptions as queries, BM25-pooled candidates from the broad MeSH corpus [SUPERSEDED 2026-05-05 — golden dataset deferred; Bioformer selected on Tier 1 proxy alone]

*(2026-05-21 status: this entry documents work that was planned but not executed. No `03a_golden_dataset.ipynb`, no `golden_retrieval_labels.json`, no annotation was performed. The Bioformer-vs-PubMedBERT selection (2026-05-05) used Tier 1 proxy evidence only, which proved decisive (24/24 categories, non-overlapping CIs). The entry is retained for historical context — it documents the IR-rigorous path that was considered and descoped.)*


**Decision:** Build the golden retrieval dataset using MTSamples `description` fields as clinical queries; for each query, select the top-N candidates by BM25 over the **broad MeSH-anchored corpus** (optionally pre-filtered by the `mesh_category` aligned with the query's MTSamples specialty). Annotate the resulting pool. BM25 is used only to select candidates for the annotator, not as the evaluated system.

**Context:** A golden IR dataset requires human-labeled query–document pairs. MTSamples `description` fields are 1–3 sentence case summaries written by clinicians — they closely match how a practitioner would phrase a literature search. *(Updated 2026-04-28: this entry was originally written assuming the per-specialty corpus from the 2026-04-20 ingestion strategy. Following the pivot to a broad MeSH-anchored corpus, the candidate-pool design changes from "complete pool annotation per specialty" to "BM25-pooled top-N over the broad corpus" — standard TREC pooling, since the full ~320K corpus cannot be annotated end-to-end.)*

**Alternatives considered:**
- **Queries from transcriptions:** More realistic distribution but harder to formulate consistently; transcriptions are long and noisy.
- **Complete pool annotation per specialty:** Original design under the per-specialty corpus. No longer feasible against a 320K-paper broad corpus.
- **Dense retrieval pooling (use the candidate retriever to select pool):** Couples gold set to the system being evaluated → biased.
- **BM25 pooling (chosen):** Standard TREC methodology. BM25 is independent of the dense retrievers under evaluation, so the gold set remains a fair benchmark for them.

**Rationale:** BM25 pooling is the established practice for IR evaluation when the candidate space is too large for exhaustive annotation. Selecting candidates with a method (BM25) different from the systems under test (dense retrievers) avoids self-confirmation bias. Optional pre-filtering by `mesh_category` keeps pools clinically focused without coupling them to dense retrieval.

**Consequences:**
- Pool size per query: ~25 BM25 top hits. Total judgments: ~5 queries × ~25 specialties × 25 candidates ≈ 3K (vs ~5K under the prior per-specialty design — comparable annotation effort).
- `notebooks/03a_golden_dataset.ipynb` must be refactored to: (1) load the flat broad corpus, (2) optionally filter candidates by `mesh_category` mapped from each query's MTSamples specialty, (3) run BM25 over that filtered (or full) subset, (4) keep the annotation export schema unchanged.
- Annotation planned via Claude Opus 4.7 + author spot-check (workflow scripts unchanged).
- Pooling bias is acknowledged as a known limitation in Cap. 6 thesis discussion (standard caveat for TREC-style evaluations).

---

### 2026-04-20 -- Bulk Ingestion Strategy for RAG Corpus [SUPERSEDED 2026-04-28 — see "Broad MeSH-anchored PubMed corpus" above]

**Decision:** Query PubMed abstracts per *medical specialty* (extracted from MTSamples) rather than per individual MTSamples patient record, forming a corpus of 50-100 high-relevance papers per specialty.

**Context:** We need a robust corpus of medical literature for the RAG vector database (Qdrant). MTSamples contains ~5,000 rows. Executing live PubMed queries for each individual row to build the RAG context would be extremely inefficient.

**Alternatives considered:**
- **Option A: Query per individual MTSamples row.** Pros: Highly specific context tailored to each patient note. Cons: ~5,000 API requests (taking hours), massive redundancy (hundreds of records share the exact same specialty), and high risk of NCBI API timeouts/bans.
- **Option B: Query per unique medical specialty (chosen).** Pros: Requires only ~40 queries total (one for each unique MTSamples specialty). Yields a clean, dense knowledge base of ~2,000-4,000 high-quality abstracts. Cons: Slightly broader context, though this can be mitigated in the future by enriching queries with top specialty keywords.

**Rationale:** Building the corpus per specialty directly supports the `rag_augmented` schema designed during the EDA phase. It drastically reduces API load and network I/O while still providing clustered, evidence-based medical literature. The resulting JSON corpus will serve as a stable, static dataset for all downstream vector indexing.

**Consequences:** 
- The bulk download logic will initially be prototyped in the ingestion notebook, and eventually transitioned into a standalone script (`scripts/ingest_pubmed.py`) for robustness.
- All downstream embedding steps (`03_embedding_benchmark.ipynb`) will read from the static JSON file (`pubmed_bulk_corpus.json`) instead of hitting the NCBI API live.

---

### 2026-04-20 -- RAG Knowledge Source: Abstract-Only Retrieval from PubMed

**Date:** 2026-04-20  
**Decision:** Retrieve and index only abstract-level fields from PubMed via Entrez, 
not full article text.

**Fields ingested per record:**
- Title (TI)
- Abstract (AB)
- MeSH Terms (MH) — appended as structured metadata
- Author Keywords (OT)
- Journal, Year, PMID, DOI

**Alternatives considered:**
1. Full text via PubMed Central (PMC) API
   - Rejected: only available for ~40% of articles (open-access subset with a PMC ID)
   - Would create asymmetric coverage: newer OA articles over-represented
   - Full text includes methods boilerplate and supplementary noise that degrades 
     retrieval precision for clinical Q&A

2. Hybrid: abstract for all + full text when PMC available
   - Rejected for v1: adds pipeline complexity without clear quality gain given the 
     use case (bibliographic consultation, not protocol extraction)
   - Revisit in future work if evaluation shows abstract truncation as a failure mode

**Justification:**  
Per NLM official documentation (MEDLINE/PubMed Data Element Field Descriptions), 
full text is not part of MEDLINE records. Abstracts are consistently available across 
all indexed articles, structured (BACKGROUND/METHODS/RESULTS/CONCLUSIONS), and 
contain the information density needed for clinical bibliographic assistance.  
Abstract-only retrieval maximizes corpus coverage while maintaining retrieval precision.

**Known limitation:**  
Specific procedural details, exact dosages, or data from figures/tables in the body 
of articles are not captured. This is documented as a scope boundary, not a defect.

---

### 2026-04-13 -- Structured bibliography reference guide as a standalone operational document

**Decision:** Maintain a dedicated `bibliography_reference_guide.md` document alongside `bibliography.bib`, providing for each reference: a summary, the thesis section(s) where it applies, the specific argument it supports, and a quick-reference index organized by thesis chapter.

**Context:** The bibliography has grown to 28 references spanning multiple thematic blocks (foundational architecture, fine-tuning, RAG, biomedical embeddings, vector databases, hallucinations, clinical applications, evaluation, and data management). The `bibliography.bib` file, while authoritative for citation metadata, is not usable as a cognitive navigation tool during writing. Without a structured guide, the risk is either over-citing (inserting redundant references) or under-citing (missing mandatory citations in key sections), both of which create friction during thesis review.

**Alternatives considered:**
- **Rely solely on Zotero annotations:** Pros — keeps metadata and notes co-located; supports tagging and search. Cons — not portable, not version-controlled alongside the thesis repository, requires GUI access, and cannot be reviewed in the same editing session as the manuscript.
- **Inline comments in `bibliography.bib`:** Pros — no separate file to maintain; notes travel with the citation. Cons — BibTeX comment syntax is non-standard across parsers, notes cannot be structured as a table, and the file is already dense with metadata fields.
- **Standalone `bibliography_reference_guide.md` (chosen):** Pros — version-controlled in the repository, renderable as a table in any Markdown viewer, structured by thesis chapter for direct lookup during writing, and easy to update as new references are added. Cons — requires manual synchronization with `bibliography.bib` when entries are added or removed.

**Rationale:** The guide addresses a real operational bottleneck: during thesis writing, the question is not "what is the DOI of this paper?" (Zotero answers that) but "which paper do I cite here, and what argument does it support?" The chapter-indexed quick-reference table at the end of the guide directly answers that question without requiring context-switching. The synchronization cost is acceptable given that the bibliography grows infrequently and in discrete batches.

**Consequences:**
- `bibliography_reference_guide.md` becomes a living document: every new reference added to `bibliography.bib` must have a corresponding row added to the guide, including its thematic block assignment, summary, and section mapping.
- The guide serves as an audit trail for bibliography coverage gaps: if a thesis section has no references listed in the quick-reference table, that is a signal to review whether citations are missing.
- During the thesis writing phase, the guide should be the first document consulted before inserting a `\cite{}` command.

---

### 2026-03-24 -- Local LLM Base Model and Fine-Tuning Stack

**Decision:** Adopt Meta Llama 3.1 (8B) as the base model, loaded in 4-bit quantization using the Unsloth framework for both inference and QLoRA fine-tuning.

**Context:** The project requires a local LLM capable of strict clinical reasoning and adherence to RAG contexts for Evidence-Based Medicine. The entire pipeline (Vector DB, Embeddings, LLM inference, and LoRA fine-tuning) must operate within the strict hardware constraints of a single Nvidia RTX 3080 with 10GB of VRAM.

**Alternatives considered:**
- Option A: *Microsoft Phi-4-Mini (3.8B)*. Pros: Low memory footprint (~3.5GB). Cons: Insufficient reasoning depth for complex clinical tasks.
- Option B: *Nvidia Nemotron-3 (4B) / Qwen 3.5 (4B/9B)*. Pros: Highly performant. Cons: Hybrid architectures (Mamba/MoE) require `trust_remote_code=True` and complex C++ dependencies, causing environment instability.
- Option C: *Google Gemma 3 (4B)*. Pros: Large context window. Cons: High memory overhead limits RAG capacity.
- Option D: *Standard Hugging Face `transformers` + `bitsandbytes`*. Pros: Industry standard. Cons: Slower generation and higher memory overhead during training.
- Option E (chosen): *Meta Llama 3.1 (8B) + Unsloth (4-bit)*. Pros: Proven clinical reasoning, strong instruction-following, broad community support. 4-bit quantization reduces footprint to ~6GB VRAM, leaving sufficient memory for RAG and fine-tuning.

**Rationale:** Llama 3.1 8B offers superior clinical reasoning capabilities compared to smaller alternatives while remaining within VRAM constraints through 4-bit quantization. Unsloth's optimized kernels deliver fast inference and seamless QLoRA fine-tuning. This balances performance with hardware pragmatism.

**Consequences:** 
- Model loading must use `unsloth.FastLanguageModel` rather than standard `transformers.AutoModelForCausalLM`.
- Imports must be strictly ordered (`unsloth` imported before `torch` and `transformers`) to ensure internal library patching applies correctly.
- The host OS must have Python development headers (`python3-dev`) and `build-essential` installed for Triton kernel compilation.

---

### 2026-03-24 -- Package manager: uv

**Decision:** Use `uv` as the package and virtual environment manager instead of pip + requirements.txt.

**Context:** The project needs a dependency manager that guarantees reproducibility (lockfile) and is fast enough for iterative development.

**Alternatives considered:**
- pip + requirements.txt: Standard approach, but no native lockfile, slow dependency resolution, and `pip freeze` produces brittle requirement files.
- Poetry: Provides a lockfile and pyproject.toml, but is significantly slower than uv and more complex to configure.
- uv (chosen): Deterministic lockfile, ultra-fast resolution, integrated venv management, compatible with the standard pyproject.toml format.

**Rationale:** uv combines the simplicity of pip with the reproducibility guarantees of Poetry, at speeds 10-100x faster. The lockfile (`uv.lock`) is versioned in Git, ensuring any contributor can replicate the exact environment.

**Consequences:** Anyone cloning the repository needs to install uv first (`curl -LsSf https://astral.sh/uv/install.sh | sh`). Dependencies are added with `uv add <package>` rather than by editing a requirements file manually.

---

### 2026-03-23 -- Repository structure

**Decision:** Separate exploratory code (notebooks/) from production code (src/), with centralised configuration in YAML files.

**Context:** The project needs a structure that supports fast experimentation while also demonstrating sound software engineering to the examination committee.

**Alternatives considered:**
- Everything in notebooks: Faster to prototype, but hard to maintain and not professional enough.
- Scripts only: Cleaner, but loses the exploratory narrative useful for the thesis document.
- Hybrid (chosen): notebooks for exploration, src/ for finalised, reusable code.

**Rationale:** The hybrid approach allows fast iteration in notebooks, followed by refactoring into reusable modules once code stabilises. The src/ subpackage structure (data, training, rag, evaluation) mirrors the project phases.

**Consequences:** Discipline is required to migrate code from notebooks to src/ once it stabilises.



<!-- Add new decisions above this line -->
