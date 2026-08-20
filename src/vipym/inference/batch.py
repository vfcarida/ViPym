"""High-Throughput Batch Submission and Concurrency Engine."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from vipym.core.logger import get_logger
from vipym.inference.backends.base import BaseInferenceBackend
from vipym.interfaces.inference import GenerationRequest, GenerationResponse, InferenceBackend

logger = get_logger(__name__)


class BatchInferenceRunner:
    """Orchestrates high-volume batch evaluation requests across serving backends."""

    def __init__(
        self,
        backend: InferenceBackend,
        batch_size: int = 32,
        max_concurrency: int = 16,
    ) -> None:
        self.backend = backend
        self.batch_size = max(1, batch_size)
        self.semaphore = asyncio.Semaphore(max(1, max_concurrency))

    def run_batch(
        self,
        requests: list[GenerationRequest],
    ) -> list[GenerationResponse]:
        """Synchronous wrapper for batch evaluation."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, self.run_batch_async(requests)).result()
            return loop.run_until_complete(self.run_batch_async(requests))
        except Exception:
            return asyncio.run(self.run_batch_async(requests))

    async def run_batch_async(
        self,
        requests: list[GenerationRequest],
    ) -> list[GenerationResponse]:
        """Asynchronously process large batches of generation requests with concurrency control."""
        if not requests:
            return []

        logger.info(
            f"Submitting batch evaluation of {len(requests)} requests (chunk_size={self.batch_size})"
        )
        t0 = time.perf_counter()

        all_responses: list[GenerationResponse] = []

        # If the backend is a BaseInferenceBackend, chunk into optimal batch sizes
        if isinstance(self.backend, BaseInferenceBackend):
            chunks = [requests[i : i + self.batch_size] for i in range(0, len(requests), self.batch_size)]
            for chunk in chunks:
                chunk_resp = await self.backend.generate_batch_async(chunk)
                all_responses.extend(chunk_resp)
        else:
            # Otherwise use semaphore-controlled async calls
            async def _process(req: GenerationRequest) -> GenerationResponse:
                async with self.semaphore:
                    return await self.backend.generate_async(req)

            tasks = [_process(r) for r in requests]
            all_responses = list(await asyncio.gather(*tasks))

        elapsed_s = time.perf_counter() - t0
        total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in all_responses)
        throughput = (total_tokens / elapsed_s) if elapsed_s > 0 else 0.0

        logger.info(
            f"Batch evaluation completed in {elapsed_s:.2f}s: {len(all_responses)} requests, "
            f"{total_tokens} tokens ({throughput:.1f} tok/s)"
        )

        return all_responses
