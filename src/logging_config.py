"""Centralised logging configuration.

Usage:
    from src.logging_config import get_logger
    logger = get_logger(__name__)

All modules obtain a child logger under the ``clinical_ai`` hierarchy.
A rotating file handler writes DEBUG-and-above to ``logs/clinical_ai.log``;
a console handler emits INFO-and-above.
"""

import logging
import logging.handlers
from pathlib import Path

_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "clinical_ai.log"
_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.DEBUG) -> None:
    root = logging.getLogger("clinical_ai")
    if root.handlers:
        return
    root.setLevel(level)

    fmt = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    _LOG_DIR.mkdir(exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        _LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"clinical_ai.{name}")
