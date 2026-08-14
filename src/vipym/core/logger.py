"""Structured logging subsystem for ViPym."""

import logging
import sys
from typing import Any, Dict
from rich.logging import RichHandler


def get_logger(name: str = "vipym", level: int = logging.INFO) -> logging.Logger:
    """Obtain a structured logger configured with Rich output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=False,
        )
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


class StructuredLogRecord:
    """Utility helper for structured JSON-like log entries."""

    @staticmethod
    def format_event(event_type: str, data: Dict[str, Any]) -> str:
        items = " ".join(f"[bold cyan]{k}[/bold cyan]={v}" for k, v in data.items())
        return f"[[bold green]{event_type}[/bold green]] {items}"
