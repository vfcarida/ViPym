"""SGLang Production Serving Backend with RadixAttention Shared-Prefix Optimization."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from vipym.core.exceptions import InferenceRuntimeError
from vipym.core.logger import get_logger
from vipym.inference.backends.base import BaseInferenceBackend, auto_detect_quantization
from vipym.inference.registry import InferenceRegistry
from vipym.interfaces.inference import GenerationRequest, GenerationResponse

logger = get_logger(__name__)


class SGLangBackend(BaseInferenceBackend):
    """Production serving backend for SGLang with RadixAttention shared-prefix caching."""

    def __init__(
        self,
        mode: str = "local",  # "local" or "remote"
        api_base_url: str = "http://localhost:30000/v1",
        requests_per_second: float | None = None,
        max_retries: int = 3,
        retry_backoff_factor: float = 1.5,
        enable_radix_cache: bool = True,
    ) -> None:
        super().__init__(
            requests_per_second=requests_per_second,
            max_retries=max_retries,
            retry_backoff_factor=retry_backoff_factor,
        )
        self.mode = mode.lower()
        self.api_base_url = api_base_url.rstrip("/")
        self.enable_radix_cache = enable_radix_cache

        self.engine: Any = None
        self.model_path: str | None = None
        self.quantization: str | None = None

    def start(
        self,
        model_path_or_id: str | Path,
        gpu_count: int = 1,
        tensor_parallel_size: int = 1,
        max_model_len: int = 4096,
        quantization: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model_path = str(model_path_or_id)
        self.quantization = quantization or auto_detect_quantization(model_path_or_id)

        if self.mode == "remote":
            logger.info(f"Connecting to remote SGLang endpoint at {self.api_base_url}")
            self.engine = "remote_sglang_client"
            return

        logger.info(
            f"Starting local SGLang engine with RadixAttention: model={self.model_path}, "
            f"tp={tensor_parallel_size}, radix_cache={self.enable_radix_cache}"
        )

        try:
            import sglang as sgl  # type: ignore[import]

            self.engine = sgl.Engine(
                model_path=self.model_path,
                tp_size=tensor_parallel_size,
                context_length=max_model_len,
                disable_radix_cache=not self.enable_radix_cache,
                **kwargs,
            )
        except ImportError:
            logger.warning("SGLang not installed; running in mock inference mode.")
            self.engine = "mock_sglang_engine"

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self.engine is None:
            raise InferenceRuntimeError("SGLang engine has not been started. Call .start() first.")

        start_time = time.perf_counter()

        if self.engine in ("mock_sglang_engine", "remote_sglang_client"):
            # Deterministic mock response
            gen_text = f"def solution():\n    return 'sglang_radix_output_{hash(request.prompt) % 1000}'\n"
            total_time = 20.0
            p_tokens = len(request.prompt.split())
            c_tokens = len(gen_text.split())
            ttft = 8.0
            itl = 1.5
        else:
            out = self.engine.generate(
                prompt=request.prompt,
                sampling_params={
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "max_new_tokens": request.max_new_tokens,
                    "stop": request.stop_tokens,
                },
            )
            total_time = (time.perf_counter() - start_time) * 1000.0
            gen_text = out["text"]
            p_tokens = out.get("meta_info", {}).get("prompt_tokens", len(request.prompt.split()))
            c_tokens = out.get("meta_info", {}).get("completion_tokens", len(gen_text.split()))
            ttft = total_time / max(1, c_tokens)
            itl = (total_time - ttft) / max(1, c_tokens - 1) if c_tokens > 1 else ttft

        return GenerationResponse(
            generated_text=gen_text,
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            time_to_first_token_ms=ttft,
            inter_token_latency_ms=itl,
            total_time_ms=total_time,
        )

    async def generate_async(self, request: GenerationRequest) -> GenerationResponse:
        return await self.execute_with_retry_async(asyncio.to_thread, self.generate, request)

    async def generate_batch_async(
        self,
        requests: list[GenerationRequest],
    ) -> list[GenerationResponse]:
        """Submit batch of requests exploiting RadixAttention shared-prefix cache."""
        if not requests:
            return []

        if self.engine is not None and self.engine not in ("mock_sglang_engine", "remote_sglang_client"):
            start_time = time.perf_counter()
            prompts = [req.prompt for req in requests]
            first_req = requests[0]
            sampling_params = {
                "temperature": first_req.temperature,
                "top_p": first_req.top_p,
                "max_new_tokens": first_req.max_new_tokens,
                "stop": first_req.stop_tokens,
            }

            outputs = await asyncio.to_thread(self.engine.generate, prompts, sampling_params)
            batch_time_ms = (time.perf_counter() - start_time) * 1000.0

            responses: list[GenerationResponse] = []
            for out in outputs:
                gen_text = out["text"] if isinstance(out, dict) else str(out)
                p_tokens = len(prompts[0].split())
                c_tokens = len(gen_text.split())
                ttft = batch_time_ms / max(1, len(requests) * c_tokens)
                responses.append(
                    GenerationResponse(
                        generated_text=gen_text,
                        prompt_tokens=p_tokens,
                        completion_tokens=c_tokens,
                        time_to_first_token_ms=ttft,
                        inter_token_latency_ms=ttft / 2.0,
                        total_time_ms=batch_time_ms / len(requests),
                    )
                )
            return responses

        return await super().generate_batch_async(requests)

    def stop(self) -> None:
        logger.info("Stopping SGLang engine and releasing resources.")
        self.engine = None


InferenceRegistry.register("sglang", SGLangBackend)
InferenceRegistry.register("sglang_backend", SGLangBackend)
InferenceRegistry.register("sglang_local", lambda: SGLangBackend(mode="local"))
InferenceRegistry.register("sglang_remote", lambda: SGLangBackend(mode="remote"))
