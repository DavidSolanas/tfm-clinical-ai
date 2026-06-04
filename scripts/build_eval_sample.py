"""Build the frozen evaluation sample (idempotent).

Usage:
    uv run python scripts/build_eval_sample.py \
        --dataset data/processed/dataset \
        --n 50 --seed 42 \
        --out data/processed/eval_sample_indices.json
"""

from __future__ import annotations

import argparse
import json

from src.evaluation.sample import build_eval_sample
from src.logging_config import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/processed/dataset")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/processed/eval_sample_indices.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    indices = build_eval_sample(
        dataset_path=args.dataset,
        n=args.n,
        seed=args.seed,
        out_path=args.out,
        force=args.force,
    )

    with open(args.out) as f:
        payload = json.load(f)
    logger.info("Eval sample: %d indices", len(indices))
    logger.info("Per-specialty counts: %s", payload["specialty_counts"])
    logger.info("Eligible pool: %d (no-evidence excluded: %d)",
                payload["eligible_pool_size"], payload["no_evidence_excluded"])


if __name__ == "__main__":
    main()
