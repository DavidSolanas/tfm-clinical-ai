# Hybrid Clinical Assistance System

**Open-Source LLM Fine-Tuning + RAG for Evidence-Based Medicine**

Master's Thesis — Master's Degree in Artificial Intelligence (UNIR)

---

## Description

A prototype system combining:
- **Fine-tuning** (QLoRA) of an open-source LLM on clinical notes (MTSamples)
- **RAG** (Retrieval-Augmented Generation) connected to PubMed for evidence-grounded responses
- **MLOps** with MLflow for experiment tracking and reproducibility

## Architecture

```
Clinical query
       |
       v
+---------------+     +--------------+
|   Embedding   |---->|    Qdrant    |
|  (BioBERT)    |     |   (top-k)    |
+---------------+     +------+-------+
                              | context
                              v
                       +--------------+
                       |  LLM (FT)   |
                       |  + augmented|----> Response + references
                       |    prompt   |
                       +--------------+
```

## Quick Start

### Prerequisites
- Python 3.10+
- Docker and Docker Compose
- (Optional) GPU with 16 GB+ VRAM for local fine-tuning

### 1. Clone and set up the environment

```bash
git clone <repo-url>
cd tfm-clinical-ai

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync --extra dev
```

> `uv sync` creates the `.venv`, resolves dependencies, and installs them from the lockfile.
> To add packages: `uv add <package>`. To activate the venv: `source .venv/bin/activate`.

### 2. Start services

```bash
docker compose up -d
```

This starts:
- **Qdrant** at `http://localhost:6333` (vector database)
- **MLflow** at `http://localhost:5000` (experiment tracking)

### 3. Verify the installation

```bash
uv run python scripts/verify_setup.py
```

### 4. Run the exploration notebook

```bash
uv run jupyter lab notebooks/
```

Start with `00_vertical_slice.ipynb` for an end-to-end "hello world" of the full system.

## Project Structure

```
tfm-clinical-ai/
|
+-- README.md                   # This file
+-- DECISIONS.md                # Technical decision log
+-- pyproject.toml              # Project dependencies and config (uv)
+-- uv.lock                     # Dependency lockfile (auto-generated)
+-- docker-compose.yml          # Qdrant + MLflow
+-- .env.example                # Environment variable template
+-- .gitignore
|
+-- configs/                    # Centralised configuration
|   +-- training.yaml           # Fine-tuning hyperparameters
|   +-- rag.yaml                # RAG pipeline configuration
|   +-- evaluation.yaml         # Evaluation configuration
|
+-- notebooks/                  # Exploration and experimentation
|   +-- 00_vertical_slice.ipynb # End-to-end hello world
|   +-- 01_mtsamples_eda.ipynb  # MTSamples exploratory analysis
|   +-- 02_pubmed_ingestion.ipynb
|   +-- 03_embedding_benchmark.ipynb
|   +-- 04_finetuning.ipynb
|   +-- 05_rag_pipeline.ipynb
|   +-- 06_evaluation.ipynb
|
+-- src/                        # Production code
|   +-- __init__.py
|   +-- config.py               # Configuration loader
|   +-- data/                   # ETL and data processing
|   |   +-- __init__.py
|   |   +-- mtsamples.py        # MTSamples pipeline
|   |   +-- pubmed.py           # PubMed pipeline
|   +-- training/               # Fine-tuning
|   |   +-- __init__.py
|   |   +-- prepare.py          # Training data preparation
|   |   +-- finetune.py         # Fine-tuning script (Unsloth)
|   +-- rag/                    # RAG pipeline
|   |   +-- __init__.py
|   |   +-- indexer.py          # Qdrant indexing
|   |   +-- retriever.py        # Semantic retrieval
|   |   +-- generator.py        # Augmented generation
|   +-- evaluation/             # Metrics and evaluation
|   |   +-- __init__.py
|   |   +-- benchmark.py        # Comparative benchmark
|   +-- api/                    # Interface
|       +-- __init__.py
|       +-- app.py              # Gradio application
|
+-- scripts/                    # Utility scripts
|   +-- verify_setup.py         # Environment verification
|   +-- ingest_pubmed.py        # Batch PubMed ingestion
|   +-- run_evaluation.py       # Full evaluation run
|
+-- data/                       # Data (NOT versioned in Git)
|   +-- raw/                    # Original downloaded data
|   +-- processed/              # Processed data
|   +-- embeddings/             # Pre-computed embeddings (cache)
|
+-- tests/                      # Tests
|   +-- __init__.py
|   +-- test_data.py
|   +-- test_rag.py
|   +-- test_evaluation.py
|
+-- docs/                       # Additional documentation
|   +-- architecture.md         # Detailed architecture diagram
|
+-- .github/                    # CI (optional)
    +-- workflows/
```

## Evaluation Configurations

The benchmark compares four configurations:

| Config | Fine-Tuning | RAG | Purpose                      |
|--------|:-----------:|:---:|------------------------------|
| A      |             |     | Baseline                     |
| B      | x           |     | Impact of fine-tuning        |
| C      |             | x   | Impact of RAG                |
| D      | x           | x   | Full system                  |

## Experiment Tracking

All experiments are logged in MLflow:

```bash
# Open the dashboard
open http://localhost:5000
```

## License

Academic project — Master's Thesis.
