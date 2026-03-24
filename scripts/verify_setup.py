#!/usr/bin/env python3
"""
Verifies that the development environment is correctly configured.
Usage: python scripts/verify_setup.py
"""

import sys
from pathlib import Path


def check(name: str, fn) -> bool:
    """Run a verification check and report the result."""
    try:
        result = fn()
        print(f"  [OK] {name}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False


def check_python():
    v = sys.version_info
    assert v.major == 3 and v.minor >= 10, (
        f"Python 3.10+ is required; found {v.major}.{v.minor}"
    )
    return f"Python {v.major}.{v.minor}.{v.micro}"


def check_torch():
    import torch
    cuda = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if cuda else "CPU only"
    return f"PyTorch {torch.__version__} | CUDA: {cuda} | {device}"


def check_transformers():
    import transformers
    return f"v{transformers.__version__}"


def check_llama_index():
    from llama_index.core import __version__
    return f"v{__version__}"


def check_qdrant():
    from qdrant_client import QdrantClient
    client = QdrantClient(host="localhost", port=6333, timeout=5)
    info = client.get_collections()
    return f"Connected | {len(info.collections)} collection(s)"


def check_mlflow():
    import mlflow
    mlflow.set_tracking_uri("http://localhost:5000")
    experiments = mlflow.search_experiments()
    return f"v{mlflow.__version__} | Connected | {len(experiments)} experiment(s)"


def check_env():
    from dotenv import dotenv_values
    env = dotenv_values(".env")
    assert "NCBI_EMAIL" in env, "NCBI_EMAIL is missing from .env"
    assert "HF_TOKEN" in env, "HF_TOKEN is missing from .env"
    return f"{len(env)} variable(s) configured"


def check_dirs():
    required = ["data/raw", "data/processed", "data/embeddings", "configs", "src", "notebooks"]
    missing = [d for d in required if not Path(d).exists()]
    assert not missing, f"Missing directories: {missing}"
    return "All required directories present"


def main():
    print("\nVerifying development environment...\n")

    checks = [
        ("Python version", check_python),
        ("Directory structure", check_dirs),
        ("Environment variables (.env)", check_env),
        ("PyTorch + GPU", check_torch),
        ("Transformers", check_transformers),
        ("LlamaIndex", check_llama_index),
        ("Qdrant (Docker)", check_qdrant),
        ("MLflow (Docker)", check_mlflow),
    ]

    results = []
    for name, fn in checks:
        results.append(check(name, fn))

    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 50}")
    print(f"Result: {passed}/{total} checks passed")

    if passed == total:
        print("Environment ready. Start with notebooks/00_vertical_slice.ipynb")
    else:
        print("Review the errors above before continuing.")
        print("  - Qdrant / MLflow: did you run 'docker compose up -d'?")
        print("  - .env: did you copy .env.example to .env and fill in the values?")
        print("  - Dependencies: did you run 'uv sync --extra dev'?")

    print()
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
