#!/usr/bin/env bash
# One-command launcher for the Clinical Evidence Console (plan §6.5, Option A).
#
# Brings up the whole local stack:
#   1. Qdrant via docker compose (vector DB on :6333).
#   2. The Gradio app (uv run) with the CPU/GPU env split (plan §3):
#        - the PARENT process gets CUDA_VISIBLE_DEVICES="" so torch
#          (Bioformer embedder + MedCPT reranker) runs on CPU and leaves VRAM
#          for generation;
#        - the service spawns the llama.cpp child with CUDA_VISIBLE_DEVICES=0
#          (set inside src/api/llama_server.py) so generation stays on the GPU.
#
# The service owns the llama.cpp subprocess and cleans it up via atexit/signal;
# this script's trap only needs to cover what it starts directly.
#
# Usage:  scripts/run_app.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> [1/3] Starting Qdrant (docker compose)…"
docker compose up -d qdrant

echo "==> [2/3] Waiting for Qdrant on :6333…"
for _ in $(seq 1 30); do
  if curl -sf http://localhost:6333/healthz >/dev/null 2>&1 \
     || curl -sf http://localhost:6333/readyz >/dev/null 2>&1; then
    echo "    Qdrant is up."
    break
  fi
  sleep 1
done

# Optional: run the context-size probe once to confirm/refresh LLAMA_CTX_SIZE.
# (Skipped by default; see scripts/probe_context_size.py.)

echo "==> [3/3] Launching Gradio app (parent torch on CPU; llama.cpp child on GPU)…"
echo "    Open http://localhost:7860 once the model has pre-warmed."

# Parent torch runs on CPU. The service resets CUDA_VISIBLE_DEVICES=0 for the
# llama.cpp child so generation runs on the GPU (see src/api/llama_server.py).
APP_PID=""
cleanup() {
  echo ""
  echo "==> Shutting down app…"
  [[ -n "$APP_PID" ]] && kill "$APP_PID" 2>/dev/null || true
  # Qdrant is left running (docker compose down qdrant to stop it).
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="" uv run python -m src.api.app &
APP_PID=$!
wait "$APP_PID"
