"""Interfaces for Inference Serving Engines."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pydantic


class GenerationRequest(pydantic.BaseModel):
    """Prompt generation request."""

    prompt: str
    max_new_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    stop_tokens: list[str] = pydantic.Field(default_factory=list)
    timeout_seconds: float = 60.0


class GenerationResponse(pydantic.BaseModel):
    """Model generation output with latency and token telemetry."""

    generated_text: str
    prompt_tokens: int
    completion_tokens: int
    time_to_first_token_ms: float
    inter_token_latency_ms: float
    total_time_ms: float
    speculative_acceptance_rate: float | None = None


class InferenceBackend(ABC):
    """Abstract interface for model serving runtimes (vLLM, SGLang, HF)."""

    @abstractmethod
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
        """Initialize and start serving engine."""
        pass

    def health_check(self) -> bool:
        """Probe serving backend readiness and memory health."""
        return True

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Synchronous prompt generation."""
        pass

    @abstractmethod
    async def generate_async(self, request: GenerationRequest) -> GenerationResponse:
        """Asynchronous prompt generation for batch evaluation."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop backend engine and release GPU memory."""
        pass
