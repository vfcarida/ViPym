"""Unit tests for resilience, retry, circuit breaker, and memory cleanup utilities."""

from __future__ import annotations

import pytest

from vipym.inference.hf_engine import HuggingFaceInferenceBackend
from vipym.utils.resilience import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    retry_with_backoff,
    safe_cuda_memory_cleanup,
)


class TestResilienceUtilities:
    def test_safe_cuda_memory_cleanup_does_not_raise(self):
        """Verify memory cleanup executes safely on both CPU and CUDA environments."""
        safe_cuda_memory_cleanup()

    def test_retry_with_backoff_success_on_retry(self):
        """Verify transient errors are retried up to max_retries."""
        attempts = 0

        @retry_with_backoff(max_retries=3, initial_delay=0.01, backoff_factor=1.5)
        def flaky_operation():
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise ConnectionResetError("Transient network timeout")
            return "SUCCESS"

        result = flaky_operation()
        assert result == "SUCCESS"
        assert attempts == 2

    def test_retry_with_backoff_exceeds_max_retries(self):
        """Verify persistent errors raise after max attempts."""
        attempts = 0

        @retry_with_backoff(max_retries=2, initial_delay=0.01)
        def failing_operation():
            nonlocal attempts
            attempts += 1
            raise ValueError("Permanent invalid value")

        with pytest.raises(ValueError, match="Permanent invalid value"):
            failing_operation()
        assert attempts == 2

    def test_circuit_breaker_opens_after_threshold(self):
        """Verify circuit breaker opens after consecutive failures and prevents downstream calls."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout_sec=10.0)

        @cb
        def failing_call():
            raise RuntimeError("Backend timeout")

        with pytest.raises(RuntimeError):
            failing_call()
        assert cb.state == "CLOSED"

        with pytest.raises(RuntimeError):
            failing_call()
        assert cb.state == "OPEN"

        # Next call blocked immediately by circuit breaker
        with pytest.raises(CircuitBreakerOpenError):
            failing_call()

    def test_inference_backend_health_check(self):
        """Verify inference backend health checks accurately report ready status."""
        backend = HuggingFaceInferenceBackend()
        assert not backend.health_check()  # Not started yet

        backend.model = "mock_model"
        backend.tokenizer = "mock_tokenizer"
        assert backend.health_check()

        backend.stop()
        assert not backend.health_check()
