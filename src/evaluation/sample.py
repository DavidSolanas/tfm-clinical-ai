"""Build and load the frozen evaluation sample for the ablation study.

The evaluation sample is a stratified-by-`medical_specialty` random subset of the
saved dataset's `test` split with no-evidence training examples excluded. The
selected indices are persisted to disk so that all four ablation configurations
evaluate the exact same queries.

Medical specialty is not stored in the saved dataset (only `messages` is), so it
is recovered by re-running `split_by_medical_specialty` with the same seed used
at dataset-build time (42) and joining on the literal transcription string.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_from_disk

from src.data.mtsamples import load_mtsamples, split_by_medical_specialty
from src.evaluation.prompts import extract_transcription
from src.logging_config import get_logger
from src.training.dataset_builder import _is_no_evidence_response

logger = get_logger(__name__)

_TRAINING_SPLIT_SEED = 42
_TRAINING_MIN_CLASS_SIZE = 10


def _assistant_response(example: dict) -> str:
    return next(m["content"] for m in example["messages"] if m["role"] == "assistant")


def _build_transcription_to_specialty() -> dict[str, str]:
    """Re-derives the training-time test split to recover medical_specialty.

    Returns:
        Map from stripped transcription string to medical_specialty.
    """
    df = load_mtsamples()
    _, _, test_df = split_by_medical_specialty(
        df, seed=_TRAINING_SPLIT_SEED, min_class_size=_TRAINING_MIN_CLASS_SIZE
    )
    return {
        str(row["transcription"]).strip(): str(row["medical_specialty"])
        for _, row in test_df.iterrows()
    }


def _stratified_indices(
    indices_by_specialty: dict[str, list[int]],
    n: int,
    seed: int,
) -> list[int]:
    """Allocate exactly `n` indices proportional to per-specialty counts.

    Largest-remainder rounding corrects drift from integer quotas. Within each
    specialty, indices are sampled without replacement using a seeded generator.
    """
    total = sum(len(v) for v in indices_by_specialty.values())
    if total < n:
        raise ValueError(f"Eligible pool ({total}) smaller than requested sample ({n})")

    specialties = sorted(indices_by_specialty.keys())
    raw = {s: n * len(indices_by_specialty[s]) / total for s in specialties}
    quotas = {s: int(np.floor(raw[s])) for s in specialties}
    remainder = n - sum(quotas.values())
    # Distribute remainder to specialties with the largest fractional part.
    order = sorted(specialties, key=lambda s: raw[s] - quotas[s], reverse=True)
    for s in order[:remainder]:
        quotas[s] += 1

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for s in specialties:
        pool = indices_by_specialty[s]
        k = min(quotas[s], len(pool))
        if k == 0:
            continue
        picks = rng.choice(len(pool), size=k, replace=False)
        selected.extend(int(pool[i]) for i in picks)
    selected.sort()
    return selected


def build_eval_sample(
    dataset_path: str = "data/processed/dataset",
    n: int = 50,
    seed: int = 42,
    out_path: str = "data/processed/eval_sample_indices.json",
    force: bool = False,
) -> list[int]:
    """Build (or return cached) the frozen evaluation index list.

    Stratifies the saved `test` split by `medical_specialty` (recovered by re-running
    the training-time split on mtsamples) after excluding no-evidence examples.

    Args:
        dataset_path: Path passed to ``datasets.load_from_disk``.
        n: Sample size. Locked to 50 by configs/evaluation.yaml.
        seed: RNG seed for the per-specialty draw. Locked to 42.
        out_path: Destination JSON file.
        force: If True, rebuild even when ``out_path`` already exists.

    Returns:
        Sorted list of integer indices into ``dataset['test']``.
    """
    out = Path(out_path)
    if out.exists() and not force:
        logger.info("Eval sample already exists at %s — returning cached indices", out)
        with open(out) as f:
            return list(json.load(f)["indices"])

    dataset = load_from_disk(dataset_path)
    if "test" not in dataset:
        raise ValueError(f"Dataset at {dataset_path} has no 'test' split")
    test = dataset["test"]

    transcription_to_specialty = _build_transcription_to_specialty()

    indices_by_specialty: dict[str, list[int]] = {}
    n_no_evidence = 0
    unresolved: list[int] = []
    for i, example in enumerate(test):
        if _is_no_evidence_response(example):
            n_no_evidence += 1
            continue
        trans = extract_transcription(example)
        specialty = transcription_to_specialty.get(trans)
        if specialty is None:
            unresolved.append(i)
            continue
        indices_by_specialty.setdefault(specialty, []).append(i)

    if unresolved:
        raise RuntimeError(
            f"Could not resolve medical_specialty for {len(unresolved)} test examples "
            f"(first few idx: {unresolved[:5]}). Training-time split is out of sync "
            f"with the saved dataset."
        )

    eligible = sum(len(v) for v in indices_by_specialty.values())
    logger.info(
        "Eligible pool: %d (test=%d, no_evidence_excluded=%d, specialties=%d)",
        eligible,
        len(test),
        n_no_evidence,
        len(indices_by_specialty),
    )

    indices = _stratified_indices(indices_by_specialty, n=n, seed=seed)

    specialty_counts = Counter(
        s for s, idxs in indices_by_specialty.items() for i in idxs if i in set(indices)
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "indices": indices,
        "n": n,
        "seed": seed,
        "dataset_path": dataset_path,
        "specialty_counts": dict(sorted(specialty_counts.items())),
        "eligible_pool_size": eligible,
        "no_evidence_excluded": n_no_evidence,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote %d eval indices to %s", len(indices), out)
    return indices


def load_eval_sample(
    indices_path: str = "data/processed/eval_sample_indices.json",
    dataset_path: str = "data/processed/dataset",
) -> list[dict]:
    """Load the frozen eval sample as ready-to-iterate records.

    Args:
        indices_path: JSON file produced by ``build_eval_sample``.
        dataset_path: Path passed to ``datasets.load_from_disk``.

    Returns:
        List of ``{idx, transcription, ground_truth_response, medical_specialty}``,
        ordered by ``idx``.
    """
    with open(indices_path) as f:
        payload = json.load(f)
    indices: list[int] = list(payload["indices"])

    dataset = load_from_disk(dataset_path)
    test = dataset["test"]
    transcription_to_specialty = _build_transcription_to_specialty()

    records: list[dict] = []
    for i in indices:
        example = test[int(i)]
        trans = extract_transcription(example)
        specialty = transcription_to_specialty.get(trans)
        if specialty is None:
            raise RuntimeError(
                f"Could not resolve medical_specialty for idx={i}; eval sample is "
                f"out of sync with the dataset."
            )
        records.append(
            {
                "idx": int(i),
                "transcription": trans,
                "ground_truth_response": _assistant_response(example),
                "medical_specialty": specialty,
            }
        )
    return records
