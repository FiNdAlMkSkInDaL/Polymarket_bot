"""
Structured logging configuration using structlog.

Produces JSON lines to both stdout and a rotating file for post-hoc analysis.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog

_CONFIGURED = False


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    """Initialise structured logging.  Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # File handler — JSON lines
    file_handler = logging.FileHandler(log_path / "bot.jsonl", encoding="utf-8")
    file_handler.setLevel(level)

    # Stdout handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)

    logging.basicConfig(
        format="%(message)s",
        level=level,
        handlers=[stream_handler, file_handler],
    )

    # Silence noisy HTTP-level request logs from httpx / httpcore
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a named structured logger."""
    setup_logging()
    return structlog.get_logger(name)
