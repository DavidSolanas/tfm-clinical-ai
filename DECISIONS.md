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
- `CLAUDE.md`: architecture description and notebook entry updated
- Cap. 3 and Cap. 5 thesis sections: update embedding model name, dimension (384), and justification
- Qdrant collection: embedding dimension changes from 768 to 384 — collection must be recreated on next ingestion run

---

### 2026-04-28 — Broad MeSH-anchored PubMed corpus (final)

**Decision:** Build a single broad reference corpus by querying PubMed once per top-level MeSH descriptor in the disease tree (Tree C, ~20 entries), the mental disorders tree (F03), and key procedure trees (Surgical Procedures, Diagnostic Techniques, Therapeutics). Each query targets `[MeSH Major Topic]` (descriptor must be the central focus, not incidentally indexed), filtered by recency (≥ 2015), publication-type whitelist (Review / Guideline / RCT / Meta-Analysis / Clinical Trial / Systematic Review), and publication-type blacklist (Editorial / Letter / Comment / News / Biography / Personal Narrative). Result: a flat `pubmed_bulk_corpus.json` of ~30–50K abstracts, each tagged with `mesh_category`. Supersedes the 2026-04-20 "Bulk Ingestion Strategy" and the (same-day) per-`sample_name` attempt.

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
- New ingestion volume is ~50K abstracts vs ~1K previously — verify Qdrant collection sizing (still trivial at <100MB embeddings).

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

### 2026-04-28 — Embedding benchmark: two-tier evaluation strategy

**Decision:** Evaluate embedding models using two tiers: (1) MeSH-category-proxy sanity check and (2) golden dataset with standard IR metrics as the primary benchmark.

**Context:** The original plan was a simple embedding comparison. A pure category proxy measures embedding-space organisation, not clinical retrieval relevance. A rigorous thesis needs quantitative IR evidence for the model selection decision in Cap. 5. *(Updated 2026-04-28: Tier-1 proxy was originally specialty-based, derived from the per-specialty corpus. Following the broad MeSH-anchored corpus pivot, the proxy label switches to `mesh_category` — same methodology, different label source.)*

**Alternatives considered:**
- **Category proxy only:** No annotation required, fast. Measures same-category@k and MRR@10. Rejected as primary benchmark because a model memorising category vocabulary would score well without useful clinical retrieval.
- **Golden dataset only:** Blocks all results until annotation is complete.
- **Two-tier (chosen):** Proxy runs immediately and catches obviously broken models. Golden dataset provides the primary result once annotated.

**Rationale:** The two-tier design separates a fast sanity check (is the embedding space structured by clinical topic?) from the actual benchmark (does the model retrieve clinically relevant abstracts for real clinical queries?). Only the second tier is cited in the thesis as evidence.

**Consequences:**
- `03a_golden_dataset.ipynb` is annotation-creation only; all benchmark logic lives in `03b`.
- `03b_embedding_benchmark.ipynb` Section 6 = proxy (always runnable, label = `mesh_category`); Section 10 = golden evaluation (conditional on annotation).
- Annotation of ~3K relevance judgments (BM25-pooled, ~25 per query) is required before Cap. 5.2 can be written.

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

### 2026-04-28 — Golden retrieval dataset: MTSamples descriptions as queries, BM25-pooled candidates from the broad MeSH corpus

**Decision:** Build the golden retrieval dataset using MTSamples `description` fields as clinical queries; for each query, select the top-N candidates by BM25 over the **broad MeSH-anchored corpus** (optionally pre-filtered by the `mesh_category` aligned with the query's MTSamples specialty). Annotate the resulting pool. BM25 is used only to select candidates for the annotator, not as the evaluated system.

**Context:** A golden IR dataset requires human-labeled query–document pairs. MTSamples `description` fields are 1–3 sentence case summaries written by clinicians — they closely match how a practitioner would phrase a literature search. *(Updated 2026-04-28: this entry was originally written assuming the per-specialty corpus from the 2026-04-20 ingestion strategy. Following the pivot to a broad MeSH-anchored corpus, the candidate-pool design changes from "complete pool annotation per specialty" to "BM25-pooled top-N over the broad corpus" — standard TREC pooling, since the full ~50K corpus cannot be annotated end-to-end.)*

**Alternatives considered:**
- **Queries from transcriptions:** More realistic distribution but harder to formulate consistently; transcriptions are long and noisy.
- **Complete pool annotation per specialty:** Original design under the per-specialty corpus. No longer feasible against a 50K-paper broad corpus.
- **Dense retrieval pooling (use the candidate retriever to select pool):** Couples gold set to the system being evaluated → biased.
- **BM25 pooling (chosen):** Standard TREC methodology. BM25 is independent of the dense retrievers under evaluation, so the gold set remains a fair benchmark for them.

**Rationale:** BM25 pooling is the established practice for IR evaluation when the candidate space is too large for exhaustive annotation. Selecting candidates with a method (BM25) different from the systems under test (dense retrievers) avoids self-confirmation bias. Optional pre-filtering by `mesh_category` keeps pools clinically focused without coupling them to dense retrieval.

**Consequences:**
- Pool size per query: ~25 BM25 top hits. Total judgments: ~5 queries × ~25 specialties × 25 candidates ≈ 3K (vs ~5K under the prior per-specialty design — comparable annotation effort).
- `notebooks/03a_golden_dataset.ipynb` must be refactored to: (1) load the flat broad corpus, (2) optionally filter candidates by `mesh_category` mapped from each query's MTSamples specialty, (3) run BM25 over that filtered (or full) subset, (4) keep the annotation export schema unchanged.
- Annotation planned via Claude Opus 4.7 + author spot-check (workflow scripts unchanged).
- Pooling bias is acknowledged as a known limitation in Cap. 6 thesis discussion (standard caveat for TREC-style evaluations).

---

### 2026-04-20 -- Bulk Ingestion Strategy for RAG Corpus

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
