"""vLLM high-performance inference engine wrapper."""

import asyncio
from pathlib import Path
import time
from typing import Any, Optional
from vipym.core.exceptions import InferenceRuntimeError
from vipym.core.logger import get_logger
from vipym.interfaces.inference import GenerationRequest, GenerationResponse, InferenceBackend
from vipym.inference.registry import InferenceRegistry

logger = get_logger(__name__)


class VLLMInferenceBackend(InferenceBackend):
    """Production serving adapter for vLLM with PagedAttention and Tensor Parallelism."""

    def __init__(self) -> None:
        self.llm = None
        self.sampling_params_cls = None
        self.model_path = None

    def start(
        self,
        model_path_or_id: str | Path,
        gpu_count: int = 1,
        tensor_parallel_size: int = 1,
        kv_cache_dtype: str = "auto",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.90,
        **kwargs: Any,
    ) -> None:
        self.model_path = str(model_path_or_id)
        logger.info(
            f"Starting vLLM engine: model={self.model_path}, tp={tensor_parallel_size}, kv_cache={kv_cache_dtype}"
        )

        try:
            from vllm import LLM, SamplingParams
            self.sampling_params_cls = SamplingParams
            self.llm = LLM(
                model=self.model_path,
                tensor_parallel_size=tensor_parallel_size,
                kv_cache_dtype=kv_cache_dtype,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True,
                **kwargs,
            )
        except ImportError:
            logger.warning("vLLM not installed; running in mock inference mode for smoke testing.")
            self.llm = "mock_vllm_engine"

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self.llm is None:
            raise InferenceRuntimeError("vLLM engine has not been started.")

        start_time = time.perf_counter()

        if self.llm == "mock_vllm_engine":
            # Deterministic mock response for smoke testing
            gen_text = f"def solution():\n    return 'mock_solution_for_{hash(request.prompt) % 1000}'\n"
            ttft = 15.0
            itl = 5.0
            total_time = 40.0
            p_tokens = len(request.prompt.split())
            c_tokens = len(gen_text.split())
        else:
            sampling_params = self.sampling_params_cls(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_new_tokens,
                stop=request.stop_tokens,
            )
            outputs = self.llm.generate([request.prompt], sampling_params)
            total_time = (time.perf_counter() - start_time) * 1000.0
            gen_text = outputs[0].outputs[0].text
            p_tokens = len(outputs[0].prompt_token_ids)
            c_tokens = len(outputs[0].outputs[0].token_ids)
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
        return await asyncio.to_thread(self.generate, request)

    def stop(self) -> None:
        logger.info("Stopping vLLM engine and releasing resources.")
        self.llm = None


InferenceRegistry.register("vllm", VLLMInferenceBackend)
