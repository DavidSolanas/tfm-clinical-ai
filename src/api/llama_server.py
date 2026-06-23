"""Manager for the llama.cpp ``llama-server`` subprocess that serves the GGUFs.

A single OpenAI-compatible server runs on ``:8001/v1``; the UI's four ablation
configs share it. Base vs fine-tuned is a **GGUF reload** of this one server
(plan §1.7, §4): only one ~4.9 GB model is resident at a time, which is what
keeps the demo inside the 10 GB VRAM budget.

The manager implements the reload safety guards the plan requires (§4):

* spawn the child with ``CUDA_VISIBLE_DEVICES=0`` so generation stays on the GPU
  even though the parent Gradio process runs torch on CPU (``CUDA_VISIBLE_DEVICES=""``);
* on reload, terminate the old process and **wait for the port to actually free**
  before respawning, to avoid bind races and orphans;
* bound the health-wait with a timeout that surfaces a clear error instead of
  hanging;
* register ``atexit``/signal cleanup so the service-spawned server is always
  killed on app exit (the ``run_app.sh`` ``trap`` only covers script-spawned
  processes).
"""

from __future__ import annotations

import atexit
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Literal

import httpx

from src.logging_config import get_logger

logger = get_logger(__name__)

ModelFamily = Literal["base", "ft"]

# Validated Q4_K_M GGUFs (plan §1.3). Relative to repo root.
_GGUF_PATHS: dict[ModelFamily, str] = {
    "base": "gguf/base/base_gguf/meta-llama-3.1-8b-instruct.Q4_K_M.gguf",
    "ft": "gguf/ft/ft_gguf/meta-llama-3.1-8b-instruct.Q4_K_M.gguf",
}

_DEFAULT_BINARY = str(Path.home() / ".unsloth" / "llama.cpp" / "llama-server")
# Grounded by scripts/probe_context_size.py (max prompt 9018 → 9216). Not 14336.
_DEFAULT_CTX = 9216


class LlamaServerError(RuntimeError):
    """Raised when the llama.cpp server fails to start or become healthy."""


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


class LlamaServerManager:
    """Starts, health-checks, and reloads a single ``llama-server`` subprocess."""

    def __init__(
        self,
        repo_root: Path,
        host: str = "127.0.0.1",
        port: int = 8001,
        binary: str | None = None,
        ctx_size: int | None = None,
        n_gpu_layers: int = 99,
        health_timeout_s: float = 120.0,
        port_free_timeout_s: float = 30.0,
    ):
        """Configure the manager (does not start anything yet).

        Args:
            repo_root: Repository root used to resolve GGUF paths.
            host: Bind host for the server.
            port: Bind port (served at ``http://host:port/v1``).
            binary: Path to ``llama-server``. Defaults to the Unsloth build, or
                the ``LLAMA_SERVER_BIN`` env var if set.
            ctx_size: llama.cpp ``-c`` context window. Defaults to the probed
                value, or the ``LLAMA_CTX_SIZE`` env var if set.
            n_gpu_layers: ``-ngl`` layers offloaded to GPU (99 = all).
            health_timeout_s: Max seconds to wait for ``/v1/models`` after start.
            port_free_timeout_s: Max seconds to wait for the port to free on reload.
        """
        self.repo_root = repo_root
        self.host = host
        self.port = port
        self.binary = binary or os.getenv("LLAMA_SERVER_BIN", _DEFAULT_BINARY)
        self.ctx_size = ctx_size or int(os.getenv("LLAMA_CTX_SIZE", _DEFAULT_CTX))
        self.n_gpu_layers = n_gpu_layers
        self.health_timeout_s = health_timeout_s
        self.port_free_timeout_s = port_free_timeout_s

        self._proc: subprocess.Popen | None = None
        self._family: ModelFamily | None = None
        self._cleanup_registered = False

    @property
    def base_url(self) -> str:
        """OpenAI-compatible base URL the LLM client should target."""
        return f"http://{self.host}:{self.port}/v1"

    @property
    def loaded_family(self) -> ModelFamily | None:
        """The model family currently loaded, or ``None`` if not started."""
        return self._family

    def _gguf_path(self, family: ModelFamily) -> Path:
        path = self.repo_root / _GGUF_PATHS[family]
        if not path.exists():
            raise LlamaServerError(f"GGUF not found for {family!r}: {path}")
        return path

    def is_healthy(self) -> bool:
        """Return True if the server answers ``/v1/models`` quickly."""
        try:
            resp = httpx.get(f"{self.base_url}/models", timeout=2.0)
            return resp.status_code == 200
        except (httpx.HTTPError, OSError):
            return False

    def ensure(self, family: ModelFamily) -> None:
        """Ensure the requested model family is loaded and healthy (idempotent).

        Reloads the GGUF (full guard sequence) only when crossing the model
        boundary; otherwise a no-op if the server is already healthy.

        Args:
            family: ``"base"`` or ``"ft"``.

        Raises:
            LlamaServerError: If the server cannot become healthy in time.
        """
        if self._family == family and self._proc and self._proc.poll() is None:
            if self.is_healthy():
                return
            logger.warning("Server for %s is up but unhealthy — restarting", family)
        self._start(family)

    def _start(self, family: ModelFamily) -> None:
        """Stop any existing server, wait for the port, then spawn for `family`."""
        gguf = self._gguf_path(family)
        self.stop()
        self._wait_port_free()

        cmd = [
            self.binary,
            "-m",
            str(gguf),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "-c",
            str(self.ctx_size),
            "-ngl",
            str(self.n_gpu_layers),
            "--no-webui",
        ]
        # CRITICAL (plan §3): parent torch runs on CPU via CUDA_VISIBLE_DEVICES="",
        # but the llama.cpp child must use the GPU or generation takes minutes.
        child_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
        logger.info("Starting llama-server (%s) on :%d, ctx=%d", family, self.port, self.ctx_size)
        self._proc = subprocess.Popen(  # noqa: S603 (trusted binary, fixed args)
            cmd,
            env=child_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._family = family
        self._register_cleanup()
        self._wait_healthy(family)

    def _wait_port_free(self) -> None:
        deadline = time.monotonic() + self.port_free_timeout_s
        while _port_in_use(self.host, self.port):
            if time.monotonic() > deadline:
                raise LlamaServerError(
                    f"Port {self.port} still in use after {self.port_free_timeout_s:.0f}s; "
                    "another llama-server may be running — free it and retry."
                )
            time.sleep(0.3)

    def _wait_healthy(self, family: ModelFamily) -> None:
        deadline = time.monotonic() + self.health_timeout_s
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise LlamaServerError(
                    f"llama-server for {family!r} exited with code {self._proc.returncode} "
                    "during startup — check the binary, GGUF path, and VRAM."
                )
            if self.is_healthy():
                logger.info("llama-server (%s) healthy at %s", family, self.base_url)
                return
            time.sleep(0.5)
        self.stop()
        raise LlamaServerError(
            f"llama-server for {family!r} did not become healthy within "
            f"{self.health_timeout_s:.0f}s."
        )

    def stop(self) -> None:
        """Terminate the managed subprocess if running (graceful, then kill)."""
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            logger.info("Stopping llama-server (pid=%s)", proc.pid)
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("llama-server did not stop gracefully — killing")
                proc.kill()
                proc.wait(timeout=5)
        self._proc = None
        self._family = None

    def _register_cleanup(self) -> None:
        if self._cleanup_registered:
            return
        atexit.register(self.stop)
        # Best-effort: also clean up on SIGTERM/SIGINT if no handler is set.
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                if signal.getsignal(sig) in (signal.SIG_DFL, None):
                    signal.signal(sig, self._on_signal)
            except (ValueError, OSError):
                # Not on the main thread (e.g. Gradio worker) — atexit covers it.
                pass
        self._cleanup_registered = True

    def _on_signal(self, signum, frame) -> None:  # noqa: ANN001
        logger.info("Signal %s received — stopping llama-server", signum)
        self.stop()
        raise SystemExit(128 + signum)
