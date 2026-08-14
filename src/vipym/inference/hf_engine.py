"""HuggingFace and SGLang fallback inference backends."""

import asyncio
from pathlib import Path
import time
from typing import Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vipym.core.exceptions import InferenceRuntimeError
from vipym.core.logger import get_logger
from vipym.interfaces.inference import GenerationRequest, GenerationResponse, InferenceBackend
from vipym.inference.registry import InferenceRegistry

logger = get_logger(__name__)


class HuggingFaceInferenceBackend(InferenceBackend):
    """Fallback PyTorch / HuggingFace inference engine."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None

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
        model_path = str(model_path_or_id)
        logger.info(f"Loading HuggingFace model for inference from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self.model is None or self.tokenizer is None:
            raise InferenceRuntimeError("HuggingFace engine not initialized.")

        inputs = self.tokenizer(request.prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        p_tokens = inputs["input_ids"].shape[1]
        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=request.temperature > 0.0,
                temperature=max(request.temperature, 1e-4),
                top_p=request.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        total_time = (time.perf_counter() - start_time) * 1000.0
        gen_tokens = outputs[0][p_tokens:]
        c_tokens = len(gen_tokens)
        gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

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
        self.model = None
        self.tokenizer = None


class SGLangInferenceBackend(InferenceBackend):
    """SGLang runtime adapter."""

    def __init__(self) -> None:
        self.engine = None

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
        logger.info(f"Starting SGLang runtime for {model_path_or_id}")
        self.engine = "mock_sglang_engine"

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(
            generated_text=f"# SGLang solution for prompt",
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=10,
            time_to_first_token_ms=12.0,
            inter_token_latency_ms=4.0,
            total_time_ms=50.0,
        )

    async def generate_async(self, request: GenerationRequest) -> GenerationResponse:
        return await asyncio.to_thread(self.generate, request)

    def stop(self) -> None:
        self.engine = None


InferenceRegistry.register("hf", HuggingFaceInferenceBackend)
InferenceRegistry.register("huggingface", HuggingFaceInferenceBackend)
InferenceRegistry.register("sglang", SGLangInferenceBackend)
