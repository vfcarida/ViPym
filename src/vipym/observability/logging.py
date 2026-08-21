"""Structured Logging and Observability Subsystem with Structlog."""

from __future__ import annotations

import logging
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import structlog
from structlog.contextvars import (
    bind_contextvars,
    clear_contextvars,
    get_contextvars,
    merge_contextvars,
    unbind_contextvars,
)

_LOGGING_CONFIGURED = False
_CURRENT_MODE = "console"


def configure_logging(
    mode: str = "console",  # "console" or "json"
    log_level: str = "INFO",
    log_file: Path | str | None = None,
) -> None:
    """Configure structlog for either colored console (development) or structured JSON (production)."""
    global _LOGGING_CONFIGURED, _CURRENT_MODE
    _CURRENT_MODE = mode.lower()

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors: list[Any] = [
        merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if _CURRENT_MODE == "json":
        # Pure JSON output for production CI/CD / log aggregators
        processors = shared_processors + [
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Colored human-readable console output for local development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True, pad_event_to=25),
        ]

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )

    # Configure root logging level
    logging.basicConfig(
        format="%(message)s",
        level=numeric_level,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    if log_file:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(p, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        logging.getLogger().addHandler(file_handler)

    _LOGGING_CONFIGURED = True


def get_logger(name: str = "vipym") -> structlog.BoundLogger:
    """Get a structured logger instance with context binding capability."""
    if not _LOGGING_CONFIGURED:
        configure_logging(mode="console")
    return structlog.get_logger(name)


def bind_context(**kwargs: Any) -> None:
    """Bind contextual attributes (e.g. experiment_id, pipeline_id, stage_name, model_name) to all subsequent log lines."""
    bind_contextvars(**kwargs)


def unbind_context(*keys: str) -> None:
    """Remove specific context variables."""
    unbind_contextvars(*keys)


def clear_context() -> None:
    """Clear all bound context variables."""
    clear_contextvars()


def get_context() -> dict[str, Any]:
    """Retrieve currently bound context variables."""
    return get_contextvars()


@contextmanager
def bound_context(**kwargs: Any) -> Generator[dict[str, Any], None, None]:
    """Context manager for temporarily binding contextual variables and restoring previous context on exit."""
    previous_context = dict(get_contextvars())
    bind_contextvars(**kwargs)
    try:
        yield get_contextvars()
    finally:
        # Restore previous state
        clear_contextvars()
        if previous_context:
            bind_contextvars(**previous_context)


def emit_event(
    event_name: str,
    logger_instance: Any | None = None,
    level: str = "info",
    **kwargs: Any,
) -> dict[str, Any]:
    """Emit a structured lifecycle event with bound context and metadata.

    Standard events include:
      - experiment_started, experiment_completed, experiment_failed
      - stage_started, stage_completed, stage_failed
      - gate_result
    """
    log = logger_instance or get_logger("vipym.events")
    event_payload = {
        "event_type": event_name,
        **kwargs,
    }

    log_fn = getattr(log, level.lower(), log.info)
    log_fn(event_name, **kwargs)

    return event_payload


__all__ = [
    "bind_context",
    "bound_context",
    "clear_context",
    "configure_logging",
    "emit_event",
    "get_context",
    "get_logger",
    "unbind_context",
]
