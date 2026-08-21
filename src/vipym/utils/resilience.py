"""Resilience, Retry, and GPU Memory Management Utilities."""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def safe_cuda_memory_cleanup() -> None:
    """Safely triggers garbage collection and releases unoccupied PyTorch CUDA cached memory."""
    try:
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as e:
        logger.debug(f"CUDA memory cleanup skipped or encountered non-fatal error: {e}")


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying transient network/IO failures with exponential backoff."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            last_err: Exception | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as err:
                    last_err = err
                    if attempt == max_retries:
                        logger.error(
                            f"Operation '{func.__name__}' failed after {max_retries} attempts: {err}"
                        )
                        raise
                    logger.warning(
                        f"Attempt {attempt}/{max_retries} for '{func.__name__}' failed ({err}). "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                    delay *= backoff_factor
            if last_err:
                raise last_err
            raise RuntimeError(f"Unexpected retry failure in {func.__name__}")

        return wrapper

    return decorator


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is in OPEN state."""


class CircuitBreaker:
    """Simple circuit breaker to protect against cascading remote service failures."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout_sec: float = 30.0) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout_sec = recovery_timeout_sec
        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.state: str = "CLOSED"  # CLOSED, OPEN, HALF-OPEN

    def record_success(self) -> None:
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.monotonic()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.error(
                f"Circuit breaker OPENED after {self.failure_count} consecutive failures. "
                f"Cooldown: {self.recovery_timeout_sec}s"
            )

    def check(self) -> None:
        if self.state == "OPEN":
            if (
                self.last_failure_time
                and (time.monotonic() - self.last_failure_time) >= self.recovery_timeout_sec
            ):
                self.state = "HALF-OPEN"
                logger.info("Circuit breaker entered HALF-OPEN state (testing recovery).")
            else:
                raise CircuitBreakerOpenError(
                    f"Service circuit breaker is OPEN. Cooldown active for {self.recovery_timeout_sec}s."
                )

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            self.check()
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception:
                self.record_failure()
                raise

        return wrapper
