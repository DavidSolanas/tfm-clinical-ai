import json
import random
import re
from pathlib import Path

import mlflow
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm

from src.data.mtsamples import split_by_medical_specialty
from src.llm_clients import LLMClient
from src.logging_config import get_logger
from src.prompts import (
    _NO_EVIDENCE_PREFIX,
    _SYSTEM_PROMPT,
    _USER_QUESTION,
    _format_evidence,
    _is_no_evidence_response,
    _user_message,
)
from src.rag.pipeline import RAGPipeline
from src.tracking import mlflow_run

logger = get_logger(__name__)

# Re-exports kept so external callers (`src.evaluation.*`) can keep importing
# the training constants from this module per the evaluation plan.
__all__ = [
    "_NO_EVIDENCE_PREFIX",
    "_SYSTEM_PROMPT",
    "_USER_QUESTION",
    "_format_evidence",
    "_is_no_evidence_response",
    "_user_message",
    "build_dataset",
]

# Strips thinking blocks from Gemma-4 (<|channel>thought...<channel|>) and Qwen3/DeepSeek-R1 (<think>...</think>)
_THINKING_RE = re.compile(
    r"(<\|channel>thought.*?<channel\|>|<think>.*?</think>)", re.DOTALL
)

_PMID_CITATION_RE = re.compile(r"\(PMID:\s*(\d+)\)")


def _strip_thinking(text: str) -> str:
    """Remove thinking blocks (Gemma-4 and Qwen3/DeepSeek-R1) from generated text."""
    return _THINKING_RE.sub("", text).strip()


def _invalid_pmids(response: str, docs: list[dict]) -> set[str]:
    allowed = {str(d["pmid"]) for d in docs}
    cited = set(_PMID_CITATION_RE.findall(response))
    return cited - allowed


def _make_example(transcription: str, docs: list[dict], response: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _user_message(transcription, docs)},
            {"role": "assistant", "content": response},
        ]
    }


def _apply_no_evidence_cap(
    examples: list[dict],
    max_ratio: float,
    seed: int,
) -> tuple[list[dict], int]:
    """Cap no-evidence examples to max_ratio of total. Returns (capped_list, n_dropped)."""
    if max_ratio >= 1.0:
        return examples, 0

    has_evidence = [e for e in examples if not _is_no_evidence_response(e)]
    no_evidence = [e for e in examples if _is_no_evidence_response(e)]

    max_allowed = round(len(has_evidence) * max_ratio / max(1e-9, 1.0 - max_ratio))

    if len(no_evidence) <= max_allowed:
        return examples, 0

    kept = random.Random(seed).sample(no_evidence, max_allowed)
    dropped = len(no_evidence) - max_allowed
    logger.info(
        "No-evidence cap applied — kept: %d, dropped: %d (ratio cap: %.0f%%)",
        max_allowed,
        dropped,
        max_ratio * 100,
    )
    return has_evidence + kept, dropped


def _process_split(
    split_df: pd.DataFrame,
    pipeline: RAGPipeline,
    client: LLMClient,
    min_doc_score: float,
    max_tokens: int,
    cache: dict,
    cache_path: Path | None,
    cache_flush_every: int,
    use_thinking: bool = False,
    split_name: str = "split",
) -> tuple[list[dict], dict]:
    examples: list[dict] = []
    new_since_flush = 0
    cache_hits = 0
    skipped = 0

    bar = tqdm(
        split_df.iterrows(),
        total=len(split_df),
        desc=split_name,
        unit="note",
        dynamic_ncols=True,
    )
    for idx, row in bar:
        note_id = str(row.get("note_id") or row.get("record_id") or idx)
        transcription = str(row.get("transcription", "")).strip()

        if note_id in cache:
            entry = cache[note_id]
            examples.append(_make_example(transcription, entry["docs"], entry["response"]))
            cache_hits += 1
            bar.set_postfix(generated=len(examples) - cache_hits, cache=cache_hits, skip=skipped)
            continue

        scored = pipeline.retrieve(transcription)
        relevant = [d for d in scored if d["final_score"] >= min_doc_score]
        if len(relevant) < pipeline.min_relevant_docs:
            skipped += 1
            bar.set_postfix(generated=len(examples) - cache_hits, cache=cache_hits, skip=skipped)
            logger.debug("Skipped note %s — only %d relevant docs", note_id, len(relevant))
            continue

        final_docs = [{**d, "rank": i + 1} for i, d in enumerate(relevant[: pipeline.final_k])]

        # Thinking blocks consume tokens before the actual response; reserve extra budget.
        effective_max_tokens = max_tokens * 3 if use_thinking else max_tokens
        response = _strip_thinking(
            client.generate(
                system=_SYSTEM_PROMPT,
                user=_user_message(transcription, final_docs),
                max_tokens=effective_max_tokens,
            )
        )

        bad_pmids = _invalid_pmids(response, final_docs)
        if bad_pmids:
            logger.warning("Skipped note %s — hallucinated PMIDs: %s", note_id, bad_pmids)
            skipped += 1
            bar.set_postfix(generated=len(examples) - cache_hits, cache=cache_hits, skip=skipped)
            continue

        cache[note_id] = {"docs": final_docs, "response": response}
        new_since_flush += 1
        if cache_path and new_since_flush >= cache_flush_every:
            with open(cache_path, "w") as f:
                json.dump(cache, f)
            new_since_flush = 0

        examples.append(_make_example(transcription, final_docs, response))
        bar.set_postfix(generated=len(examples) - cache_hits, cache=cache_hits, skip=skipped)

    if cache_path and new_since_flush > 0:
        with open(cache_path, "w") as f:
            json.dump(cache, f)

    stats = {"examples": len(examples), "cache_hits": cache_hits, "skipped": skipped}
    logger.info(
        "Split processed — examples: %d, cache_hits: %d, skipped: %d",
        len(examples),
        cache_hits,
        skipped,
    )
    return examples, stats


def build_dataset(
    df: pd.DataFrame,
    pipeline: RAGPipeline,
    client: LLMClient,
    min_doc_score: float = 0.40,
    max_tokens: int = 1024,
    cache_path: Path | None = None,
    cache_flush_every: int = 25,
    seed: int = 42,
    min_class_size: int = 10,
    max_no_evidence_ratio: float = 0.15,
    use_thinking: bool = False,
    output_path: Path | None = Path("data/processed/dataset"),
    mlflow_experiment: str = "tfm-dataset-building",
    teacher_model: str | None = None,
) -> DatasetDict:
    if output_path and output_path.exists():
        logger.info("Loading pre-built dataset from %s", output_path)
        return load_from_disk(str(output_path))

    logger.info(
        "Building dataset — total notes: %d, seed: %d, min_class_size: %d, cache: %s",
        len(df),
        seed,
        min_class_size,
        cache_path or "none",
    )
    train_df, val_df, test_df = split_by_medical_specialty(df, seed=seed, min_class_size=min_class_size)
    logger.info(
        "Split sizes — train: %d, val: %d, test: %d", len(train_df), len(val_df), len(test_df)
    )

    cache: dict = {}
    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)

    kwargs = dict(
        pipeline=pipeline,
        client=client,
        min_doc_score=min_doc_score,
        max_tokens=max_tokens,
        cache=cache,
        cache_path=cache_path,
        cache_flush_every=cache_flush_every,
        use_thinking=use_thinking,
    )

    with mlflow_run(
        experiment=mlflow_experiment,
        run_name="build-dataset",
        tags={"stage": "data", "teacher_model": teacher_model or "unknown"},
    ):
        mlflow.log_params({
            "min_doc_score": min_doc_score,
            "max_tokens": max_tokens,
            "seed": seed,
            "min_class_size": min_class_size,
            "cache_flush_every": cache_flush_every,
            "num_input_notes": len(df),
            "cache_path": str(cache_path) if cache_path else "none",
            "teacher_model": teacher_model or "unknown",
            "max_no_evidence_ratio": max_no_evidence_ratio,
            "use_thinking": use_thinking,
        })

        train_examples, train_stats = _process_split(train_df, split_name="train", **kwargs)
        val_examples, val_stats = _process_split(val_df, split_name="val", **kwargs)
        test_examples, test_stats = _process_split(test_df, split_name="test", **kwargs)

        train_examples, train_dropped = _apply_no_evidence_cap(train_examples, max_no_evidence_ratio, seed)
        val_examples, val_dropped = _apply_no_evidence_cap(val_examples, max_no_evidence_ratio, seed)
        test_examples, test_dropped = _apply_no_evidence_cap(test_examples, max_no_evidence_ratio, seed)

        dataset = DatasetDict(
            {
                "train": Dataset.from_list(train_examples).shuffle(seed=seed),
                "validation": Dataset.from_list(val_examples),
                "test": Dataset.from_list(test_examples),
            }
        )

        mlflow.log_metrics({
            "train_size": len(dataset["train"]),
            "val_size": len(dataset["validation"]),
            "test_size": len(dataset["test"]),
            "train_cache_hits": train_stats["cache_hits"],
            "val_cache_hits": val_stats["cache_hits"],
            "test_cache_hits": test_stats["cache_hits"],
            "train_skipped": train_stats["skipped"],
            "val_skipped": val_stats["skipped"],
            "test_skipped": test_stats["skipped"],
            "total_skipped": train_stats["skipped"] + val_stats["skipped"] + test_stats["skipped"],
            "total_examples": len(dataset["train"]) + len(dataset["validation"]) + len(dataset["test"]),
            "train_no_evidence_dropped": train_dropped,
            "val_no_evidence_dropped": val_dropped,
            "test_no_evidence_dropped": test_dropped,
        })

    logger.info(
        "Dataset ready — train: %d, validation: %d, test: %d",
        len(dataset["train"]),
        len(dataset["validation"]),
        len(dataset["test"]),
    )

    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_path))
        logger.info("Dataset saved to %s", output_path)

    return dataset
