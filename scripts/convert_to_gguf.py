"""Convert a model to a llama.cpp GGUF for the ablation-study evaluation server.

Produces the two GGUFs the evaluation phases load into llama.cpp (port 8001):

* **FT GGUF** (Phase 2, configs B/D): pass ``--adapter <dir>`` pointing at the
  directory ``run_finetune.py`` wrote (a LoRA adapter). Unsloth loads the base
  model named in ``adapter_config.json``, merges the adapter to fp16, then
  quantizes.
* **Base GGUF** (Phase 1, configs A/C): pass ``--base-only``. Converts the
  untouched base model.

Both must be produced with the **same** ``--quant`` and ideally in the same pod
session so they share one llama.cpp build — then base and FT differ only in the
LoRA weights, which is what the ablation isolates (see DECISIONS.md 2026-05-21).

Requires a CUDA GPU and Unsloth (clones/builds llama.cpp on first run).

Unsloth writes the final ``.gguf`` files to ``{out}_gguf/`` (it appends ``_gguf`` to
``--out``). Intermediate merged HuggingFace weights land in ``--out`` itself.

Usage::

    # FT model (loads base + adapter, merges, quantizes)
    uv run python scripts/convert_to_gguf.py --adapter models/llama-3.1-8b-clinical \\
        --out gguf/ft --quant q4_k_m

    # Base model
    uv run python scripts/convert_to_gguf.py --base-only --out gguf/base --quant q4_k_m

Options:
    --adapter PATH          Saved LoRA adapter dir (output of run_finetune.py)
    --base-only             Convert the untouched base model instead of an adapter
    --base-model STR        Base model id/path        (default: from configs/training.yaml)
    --out PATH              Output directory for the GGUF
    --quant STR             llama.cpp quantization    (default: q4_k_m)
    --max-seq-length INT    Max sequence length       (default: from configs/training.yaml)
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

from dotenv import load_dotenv

# Imported after dotenv; unsloth must be imported before torch/transformers
# (see DECISIONS.md 2026-03-24). FastLanguageModel pulls them in internally.
from unsloth import FastLanguageModel

import torch

from src.config import load_config
from src.logging_config import get_logger

logger = get_logger(__name__)


def _ensure_llama_cpp_on_path() -> None:
    """Put Unsloth's llama.cpp tree on PYTHONPATH for the ``conversion`` package."""
    llama_cpp = Path.home() / ".unsloth" / "llama.cpp"
    if not llama_cpp.is_dir():
        return
    root = str(llama_cpp)
    existing = os.environ.get("PYTHONPATH", "")
    if root not in existing.split(os.pathsep):
        os.environ["PYTHONPATH"] = f"{root}{os.pathsep}{existing}" if existing else root


def _free_gpu(model) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _to_gguf(model, tokenizer, out: Path, quant: str) -> Path:
    model.save_pretrained_gguf(str(out), tokenizer, quantization_method=quant)
    # Unsloth always appends ``_gguf`` to the save directory.
    return Path(f"{out}_gguf")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert a fine-tuned adapter or the base model to a llama.cpp GGUF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Directory containing the saved LoRA adapter (output of run_finetune.py)",
    )
    p.add_argument(
        "--base-only",
        action="store_true",
        help="Convert the untouched base model instead of an adapter (Phase 1 base GGUF)",
    )
    p.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model identifier (default: model.base from configs/training.yaml)",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory where the GGUF will be written",
    )
    p.add_argument(
        "--quant",
        type=str,
        default="q4_k_m",
        help="llama.cpp quantization method (base and FT must match for a fair ablation)",
    )
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Maximum sequence length (default: model.max_seq_length from configs/training.yaml)",
    )
    return p


def main() -> None:
    load_dotenv(override=True)

    args = _build_parser().parse_args()
    cfg = load_config("training")

    if args.base_only == (args.adapter is not None):
        raise SystemExit("Pass exactly one of --adapter <dir> or --base-only.")

    if args.adapter is not None and not args.adapter.exists():
        raise FileNotFoundError(
            f"Adapter directory not found at {args.adapter}. Run scripts/run_finetune.py first."
        )

    max_seq_length = args.max_seq_length or cfg["model"]["max_seq_length"]
    # For --base-only, load the base model id; otherwise load the adapter dir,
    # which Unsloth resolves to base + LoRA via adapter_config.json.
    model_name = (args.base_model or cfg["model"]["base"]) if args.base_only else str(args.adapter)

    logger.info(
        "Converting to GGUF — source: %s, quant: %s, max_seq_length: %d, out: %s",
        model_name,
        args.quant,
        max_seq_length,
        args.out,
    )

    _ensure_llama_cpp_on_path()
    args.out.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    if args.base_only:
        # Non-PEFT models cannot go straight to save_pretrained_gguf when loaded in
        # 4-bit: Transformers 5.x raises NotImplementedError in revert_weight_conversion.
        # Export fp16 to disk first (same fp16 pull Unsloth uses for adapter merges),
        # then reload and quantize.
        logger.info("Base model: exporting fp16 merged weights to %s", args.out)
        model.save_pretrained_merged(str(args.out), tokenizer, save_method="merged_16bit")
        _free_gpu(model)
        logger.info("Base model: reloading fp16 weights for GGUF conversion")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(args.out),
            max_seq_length=max_seq_length,
            load_in_4bit=False,
        )

    gguf_dir = _to_gguf(model, tokenizer, args.out, args.quant)
    logger.info("GGUF written to %s", gguf_dir)


if __name__ == "__main__":
    main()
