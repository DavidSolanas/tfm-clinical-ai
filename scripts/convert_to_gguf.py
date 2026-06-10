"""Convert a model to a llama.cpp GGUF for the ablation-study evaluation server.

Produces the two GGUFs the evaluation phases load into llama.cpp (port 8001):

* **FT GGUF** (Phase 2, configs B/D): pass ``--adapter <dir>`` pointing at the
  directory ``run_finetune.py`` wrote (a LoRA adapter). Unsloth loads the base
  model named in ``adapter_config.json``, merges the adapter to fp16, then
  quantizes.
* **Base GGUF** (Phase 1, configs A/C): pass ``--base-only``. Pulls the fp16
  HuggingFace snapshot and quantizes via the same llama.cpp toolchain (no GPU).

Both must be produced with the **same** ``--quant`` and ideally in the same pod
session so they share one llama.cpp build — then base and FT differ only in the
LoRA weights, which is what the ablation isolates (see DECISIONS.md 2026-05-21).

Adapter conversion requires a CUDA GPU and Unsloth (clones/builds llama.cpp on
first run). Base conversion is CPU-only once llama.cpp is built.

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
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

from src.config import load_config
from src.logging_config import get_logger

logger = get_logger(__name__)

_LLAMA_CPP = Path.home() / ".unsloth" / "llama.cpp"


def _ensure_llama_cpp_on_path() -> None:
    """Put Unsloth's llama.cpp tree on PYTHONPATH for the ``conversion`` package."""
    if not _LLAMA_CPP.is_dir():
        return
    root = str(_LLAMA_CPP)
    existing = os.environ.get("PYTHONPATH", "")
    if root not in existing.split(os.pathsep):
        os.environ["PYTHONPATH"] = f"{root}{os.pathsep}{existing}" if existing else root


def _llama_cpp_env() -> dict[str, str]:
    _ensure_llama_cpp_on_path()
    return os.environ.copy()


def _require_llama_cpp() -> Path:
    if not _LLAMA_CPP.is_dir():
        raise RuntimeError(
            f"llama.cpp not found at {_LLAMA_CPP}. "
            "Run an adapter conversion first (it builds llama.cpp), or install manually."
        )
    return _LLAMA_CPP


def _find_llama_cpp_binary(name: str) -> Path:
    llama_cpp = _require_llama_cpp()
    for candidate in (llama_cpp / name, llama_cpp / "build" / "bin" / name):
        if candidate.is_file():
            return candidate
    matches = list(llama_cpp.rglob(name))
    if not matches:
        raise FileNotFoundError(f"{name} not found under {llama_cpp}")
    return matches[0]


def _convert_script() -> Path:
    llama_cpp = _require_llama_cpp()
    for name in ("convert_hf_to_gguf.py", "unsloth_convert_hf_to_gguf.py"):
        path = llama_cpp / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No HF→GGUF converter found under {llama_cpp}")


def _quant_gguf_suffix(quant: str) -> str:
    return quant.upper().replace("-", "_")


def _convert_hf_dir_to_gguf(hf_dir: Path, gguf_dir: Path, model_slug: str, quant: str) -> None:
    """Run llama.cpp HF→GGUF conversion and optional quantization."""
    llama_cpp = _require_llama_cpp()
    convert_script = _convert_script()
    gguf_dir.mkdir(parents=True, exist_ok=True)

    bf16_gguf = gguf_dir / f"{model_slug}.BF16.gguf"
    logger.info("Converting HF weights at %s → bf16 GGUF", hf_dir)
    subprocess.run(
        [
            sys.executable,
            str(convert_script),
            str(hf_dir),
            "--outfile",
            str(bf16_gguf),
            "--outtype",
            "bf16",
            "--split-max-size",
            "50G",
        ],
        check=True,
        env=_llama_cpp_env(),
        cwd=str(llama_cpp),
    )

    if quant in ("f16", "bf16", "f32"):
        return

    final_gguf = gguf_dir / f"{model_slug}.{_quant_gguf_suffix(quant)}.gguf"
    quantize_bin = _find_llama_cpp_binary("llama-quantize")
    logger.info("Quantizing GGUF → %s", quant)
    subprocess.run(
        [str(quantize_bin), str(bf16_gguf), str(final_gguf), quant],
        check=True,
        cwd=str(llama_cpp),
    )
    bf16_gguf.unlink(missing_ok=True)


def _convert_base_to_gguf(model_id: str, out: Path, quant: str) -> Path:
    """Base model: snapshot fp16 weights from HF hub, convert with llama.cpp (CPU)."""
    logger.info("Base model: resolving fp16 HuggingFace snapshot for %s", model_id)
    hf_dir = Path(snapshot_download(model_id))
    gguf_dir = Path(f"{out}_gguf")
    model_slug = Path(model_id).name.lower()
    _convert_hf_dir_to_gguf(hf_dir, gguf_dir, model_slug, quant)
    return gguf_dir


def _convert_adapter_to_gguf(
    adapter: Path,
    out: Path,
    quant: str,
    max_seq_length: int,
) -> Path:
    """FT model: Unsloth loads adapter, merges LoRA to fp16, quantizes via llama.cpp."""
    # Imported here so base-only conversion stays CPU-only (no CUDA init).
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter),
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    model.save_pretrained_gguf(str(out), tokenizer, quantization_method=quant)
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
    model_name = (args.base_model or cfg["model"]["base"]) if args.base_only else str(args.adapter)

    logger.info(
        "Converting to GGUF — source: %s, quant: %s, max_seq_length: %d, out: %s",
        model_name,
        args.quant,
        max_seq_length,
        args.out,
    )

    args.out.mkdir(parents=True, exist_ok=True)

    if args.base_only:
        gguf_dir = _convert_base_to_gguf(model_name, args.out, args.quant)
    else:
        _ensure_llama_cpp_on_path()
        gguf_dir = _convert_adapter_to_gguf(args.adapter, args.out, args.quant, max_seq_length)

    logger.info("GGUF written to %s", gguf_dir)


if __name__ == "__main__":
    main()
