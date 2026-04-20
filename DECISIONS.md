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

---

<!-- Add new decisions above this line -->
