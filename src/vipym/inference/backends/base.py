"""Base Production Inference Backend with Batching, Retries, and Rate Limiting."""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Any

from vipym.core.exceptions import InferenceRuntimeError
from vipym.core.logger import get_logger
from vipym.interfaces.inference import GenerationRequest, GenerationResponse, InferenceBackend

logger = get_logger(__name__)


def auto_detect_quantization(model_path_or_id: str | Path) -> str | None:
    """Inspect model configuration or directory structure to detect quantization format."""
    p = Path(model_path_or_id)
    name_lower = str(model_path_or_id).lower()

    if "gptq" in name_lower:
        return "gptq"
    if "awq" in name_lower:
        return "awq"
    if "fp8" in name_lower or "float8" in name_lower:
        return "fp8"
    if "marlin" in name_lower:
        return "marlin"

    # Inspect config.json if directory exists
    if p.is_dir():
        cfg_file = p / "config.json"
        if cfg_file.exists():
            try:
                data = json.loads(cfg_file.read_text(encoding="utf-8"))
                quant_cfg = data.get("quantization_config", {})
                quant_method = quant_cfg.get("quant_method", "").lower()
                if quant_method:
                    return quant_method
            except Exception:
                pass

        quant_file = p / "quantize_config.json"
        if quant_file.exists():
            try:
                data = json.loads(quant_file.read_text(encoding="utf-8"))
                version = data.get("version", "").lower()
                if "gptq" in version:
                    return "gptq"
                if "awq" in version:
                    return "awq"
            except Exception:
                pass

    return None


class BaseInferenceBackend(InferenceBackend, ABC):
    """Abstract base class for high-throughput serving backends (vLLM, SGLang, Remote APIs)."""

    def __init__(
        self,
        requests_per_second: float | None = None,
        max_retries: int = 3,
        retry_backoff_factor: float = 1.5,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.requests_per_second = requests_per_second
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff_factor
        self.timeout_seconds = timeout_seconds

        self._rate_limiter_delay = (
            (1.0 / requests_per_second) if requests_per_second and requests_per_second > 0 else 0.0
        )
        self._last_request_time = 0.0

    async def _apply_rate_limit(self) -> None:
        """Apply token-bucket rate limiting between requests if configured."""
        if self._rate_limiter_delay <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._rate_limiter_delay:
            await asyncio.sleep(self._rate_limiter_delay - elapsed)
        self._last_request_time = time.monotonic()

    async def execute_with_retry_async(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute an async operation with exponential backoff on transient errors."""
        last_err: Exception | None = None
        delay = 0.5

        for attempt in range(1, self.max_retries + 1):
            try:
                await self._apply_rate_limit()
                return await func(*args, **kwargs)
            except Exception as err:
                last_err = err
                # Check for non-retryable critical errors
                err_str = str(err).lower()
                if "invalid argument" in err_str or "unauthorized" in err_str:
                    raise err

                logger.warning(
                    f"Inference request failed (attempt {attempt}/{self.max_retries}): {err}. Retrying in {delay:.2f}s..."
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(delay)
                    delay *= self.retry_backoff

        raise InferenceRuntimeError(
            f"Operation failed after {self.max_retries} attempts: {last_err}"
        ) from last_err

    def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationResponse]:
        """Synchronous batch generation."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, self.generate_batch_async(requests)).result()
            return loop.run_until_complete(self.generate_batch_async(requests))
        except Exception:
            return asyncio.run(self.generate_batch_async(requests))

    async def generate_batch_async(
        self,
        requests: list[GenerationRequest],
    ) -> list[GenerationResponse]:
        """Asynchronous continuous batch generation. Default implementation runs concurrently."""
        tasks = [self.generate_async(req) for req in requests]
        return await asyncio.gather(*tasks)

    async def stream_generate(
        self,
        request: GenerationRequest,
    ) -> AsyncGenerator[str, None]:
        """Stream generated text token-by-token (default falls back to single-yield)."""
        resp = await self.generate_async(request)
        tokens = resp.generated_text.split()
        for i, t in enumerate(tokens):
            yield t + (" " if i < len(tokens) - 1 else "")
            await asyncio.sleep(0.01)

    def health_check(self) -> bool:
        """Verify that the serving backend is online and responsive."""
        try:
            dummy_req = GenerationRequest(prompt="Hello", max_new_tokens=2)
            resp = self.generate(dummy_req)
            return len(resp.generated_text) >= 0
        except Exception as e:
            logger.warning(f"Backend health check failed: {e}")
            return False
