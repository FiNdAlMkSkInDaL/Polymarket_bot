"""
Structured logging configuration using structlog.

Produces JSON lines to both stdout and a rotating file for post-hoc analysis.
All disk I/O is offloaded to a background thread via QueueHandler/QueueListener
so that log calls never block the asyncio event loop.
"""

from __future__ import annotations

import atexit
import logging
import logging.handlers
import queue
import sys
from pathlib import Path

import structlog

_CONFIGURED = False
_LISTENER: logging.handlers.QueueListener | None = None


def setup_logging(
    log_dir: str = "logs",
    level: int = logging.INFO,
    log_file: str = "bot.jsonl",
) -> None:
    """Initialise structured logging.  Safe to call multiple times."""
    global _CONFIGURED, _LISTENER
    if _CONFIGURED:
        return

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Rotating file handler — 10 MB per file, 5 backups
    file_handler = logging.handlers.RotatingFileHandler(
        log_path / log_file,
        encoding="utf-8",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
    )
    file_handler.setLevel(level)

    # Stdout handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)

    # Offload all handler I/O to a background thread via QueueHandler.
    # This guarantees log calls never block the asyncio event loop.
    log_queue: queue.Queue[logging.LogRecord] = queue.Queue(-1)
    queue_handler = logging.handlers.QueueHandler(log_queue)

    _LISTENER = logging.handlers.QueueListener(
        log_queue, file_handler, stream_handler, respect_handler_level=True,
    )
    _LISTENER.start()
    atexit.register(_LISTENER.stop)

    logging.basicConfig(
        format="%(message)s",
        level=level,
        handlers=[queue_handler],
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
