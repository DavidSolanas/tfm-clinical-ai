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
