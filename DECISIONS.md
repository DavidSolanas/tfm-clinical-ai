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

### 2026-03-24 -- Local LLM Base Model and Fine-Tuning Stack

**Decision:** Adopt Microsoft Phi-4-Mini (3.8B) as the base model, loaded in 4-bit quantization using the Unsloth framework for both inference and QLoRA fine-tuning.

**Context:** The project requires a local LLM capable of strict clinical reasoning and adherence to RAG contexts for Evidence-Based Medicine. Crucially, the entire pipeline (Vector DB, Embeddings, LLM inference, and LoRA fine-tuning) must operate within the strict hardware constraints of a single Nvidia RTX 3080 with 10GB of VRAM.

**Alternatives considered:**
- Option A: *Nvidia Nemotron-3 (4B) / Qwen 3.5 (4B/9B)*. Pros: Highly performant. Cons: These utilize hybrid architectures (Mamba/MoE) requiring `trust_remote_code=True` and complex C++ dependencies (`mamba-ssm`, `causal-conv1d`), which caused severe environment instability and compilation failures.
- Option B: *DeepSeek-R1-Distill-Llama (8B)*. Pros: State-of-the-art chain-of-thought clinical reasoning. Cons: Even with quantization, an 8B model consumes too much VRAM, leaving insufficient memory for large RAG medical contexts and QLoRA gradients on a 10GB GPU, leading to out-of-memory (OOM) crashes.
- Option C: *Google Gemma 3 (4B)*. Pros: Massive 128K context window. Cons: Native 16-bit loading requires ~8.6GB of VRAM, severely bottlenecking RAG retrieval space.
- Option D: *Standard Hugging Face `transformers` + `bitsandbytes`*. Pros: Industry standard. Cons: Slower generation speeds and higher memory overhead during training compared to optimized kernels.
- Option E (chosen): *Microsoft Phi-4-Mini (3.8B) + Unsloth (4-bit)*. Pros: Pure Transformer architecture avoids C++ dependency hell. Phi is uniquely trained on "textbook-quality" logic, making it exceptional at clinical deduction. Unsloth handles 4-bit quantization natively, shrinking the model footprint to ~3.5GB VRAM.

**Rationale:** This combination perfectly balances clinical reasoning capabilities with strict hardware pragmatism. Phi-4-Mini provides the necessary logic for medical deduction, while Unsloth's optimized Triton kernels deliver 2x faster inference and frictionless QLoRA fine-tuning. Shrinking the model to 3.5GB VRAM guarantees ~6.5GB of free memory remains available for Qdrant, embeddings, and massive clinical document prompts without crashing.

**Consequences:** - The host OS must have Python development headers (`python3-dev`) and `build-essential` installed to allow Unsloth to dynamically compile Triton kernels.
- Model loading code must use `unsloth.FastLanguageModel` rather than standard `transformers.AutoModelForCausalLM`.
- Imports must be strictly ordered (`unsloth` imported before `torch` and `transformers`) to ensure internal library patching applies correctly.

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
