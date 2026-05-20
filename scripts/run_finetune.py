"""Run QLoRA fine-tuning on the pre-built SFT dataset.

Loads the HuggingFace DatasetDict produced by ``scripts/build_dataset.py``,
applies QLoRA via Unsloth, and saves the merged adapter to disk.
Requires a CUDA GPU and Unsloth installed (see project README).

Usage::

    uv run python scripts/run_finetune.py [options]

Options:
    --dataset PATH          Pre-built DatasetDict directory       (default: data/processed/dataset)
    --output-dir PATH       Where to save the fine-tuned model   (default: models/llama-3.1-8b-clinical)
    --base-model STR        HuggingFace model ID or local path   (default: from configs/training.yaml)
    --lora-rank INT         LoRA rank r                          (default: from configs/training.yaml)
    --lora-alpha INT        LoRA alpha                           (default: from configs/training.yaml)
    --epochs INT            Number of training epochs            (default: from configs/training.yaml)
    --lr FLOAT              Learning rate                        (default: from configs/training.yaml)
    --batch-size INT        Per-device train batch size          (default: from configs/training.yaml)
    --grad-accum INT        Gradient accumulation steps          (default: from configs/training.yaml)
    --seed INT              Random seed                          (default: 42)
    --mlflow-experiment STR MLflow experiment name              (default: tfm-finetuning)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_from_disk
from dotenv import load_dotenv

from src.config import load_config
from src.logging_config import get_logger
from src.training.finetune import finetune

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="QLoRA fine-tuning of Llama 3.1 on the clinical SFT dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/dataset"),
        help="Directory containing the HuggingFace DatasetDict (output of build_dataset.py)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/llama-3.1-8b-clinical"),
        help="Directory where the fine-tuned model and tokenizer will be saved",
    )
    p.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model identifier (default: model.base from configs/training.yaml)",
    )
    p.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        help="LoRA rank r (default: qlora.r from configs/training.yaml)",
    )
    p.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="LoRA alpha (default: qlora.lora_alpha from configs/training.yaml)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs (default: training.num_epochs from configs/training.yaml)",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: training.learning_rate from configs/training.yaml)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Per-device train batch size (default: training.per_device_batch_size from configs/training.yaml)",
    )
    p.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        help="Gradient accumulation steps (default: training.gradient_accumulation_steps from configs/training.yaml)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    p.add_argument(
        "--mlflow-experiment",
        type=str,
        default="tfm-finetuning",
        help="MLflow experiment name where this run will be recorded",
    )
    return p


def main() -> None:
    load_dotenv(override=True)

    args = _build_parser().parse_args()
    cfg = load_config("training")

    if not args.dataset.exists():
        raise FileNotFoundError(
            f"Dataset not found at {args.dataset}. "
            "Run scripts/build_dataset.py first."
        )

    logger.info("Loading dataset from %s", args.dataset)
    dataset = load_from_disk(str(args.dataset))
    logger.info(
        "Dataset loaded — train: %d, validation: %d, test: %d",
        len(dataset["train"]),
        len(dataset["validation"]),
        len(dataset["test"]),
    )

    finetune(
        dataset=dataset,
        output_dir=args.output_dir,
        base_model=args.base_model or cfg["model"]["base"],
        lora_rank=args.lora_rank or cfg["qlora"]["r"],
        lora_alpha=args.lora_alpha or cfg["qlora"]["lora_alpha"],
        lora_dropout=cfg["qlora"]["lora_dropout"],
        target_modules=tuple(cfg["qlora"]["target_modules"]),
        per_device_batch_size=args.batch_size or cfg["training"]["per_device_batch_size"],
        gradient_accumulation_steps=args.grad_accum or cfg["training"]["gradient_accumulation_steps"],
        num_train_epochs=args.epochs or cfg["training"]["num_epochs"],
        learning_rate=args.lr or cfg["training"]["learning_rate"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        weight_decay=cfg["training"]["weight_decay"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        logging_steps=cfg["training"]["logging_steps"],
        seed=args.seed,
        mlflow_experiment=args.mlflow_experiment,
    )


if __name__ == "__main__":
    main()
