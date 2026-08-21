"""Structured logging subsystem for ViPym."""

from __future__ import annotations

import logging
from typing import Any

from vipym.observability.logging import (
    bind_context,
    clear_context,
    configure_logging,
    emit_event,
    get_context,
    get_logger as get_structlog_logger,
    unbind_context,
)


def get_logger(name: str = "vipym", level: int = logging.INFO) -> Any:
    """Obtain a structured structlog logger instance."""
    return get_structlog_logger(name)


class StructuredLogRecord:
    """Utility helper for structured JSON-like log entries."""

    @staticmethod
    def format_event(event_type: str, data: dict[str, Any]) -> str:
        items = " ".join(f"[bold cyan]{k}[/bold cyan]={v}" for k, v in data.items())
        return f"[[bold green]{event_type}[/bold green]] {items}"


__all__ = [
    "StructuredLogRecord",
    "bind_context",
    "clear_context",
    "configure_logging",
    "emit_event",
    "get_context",
    "get_logger",
    "unbind_context",
]
