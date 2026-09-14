"""HuggingFace Local PyTorch Serving Backend with Streaming and Fallback Support."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vipym.core.exceptions import InferenceRuntimeError
from vipym.core.logger import get_logger
from vipym.inference.backends.base import BaseInferenceBackend
from vipym.inference.registry import InferenceRegistry
from vipym.interfaces.inference import (
    GenerationChunk,
    GenerationRequest,
    GenerationResponse,
)

logger = get_logger(__name__)


class HuggingFaceInferenceBackend(BaseInferenceBackend):
    """Fallback and local PyTorch / HuggingFace inference engine with streaming support."""

    def __init__(
        self,
        requests_per_second: float | None = None,
        max_retries: int = 3,
        retry_backoff_factor: float = 1.5,
    ) -> None:
        super().__init__(
            requests_per_second=requests_per_second,
            max_retries=max_retries,
            retry_backoff_factor=retry_backoff_factor,
        )
        self.model: Any = None
        self.tokenizer: Any = None

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
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True,
            )
        except Exception as err:
            logger.info(
                f"Standard AutoModel load raised ({err}), attempting fallback without quantization hook..."
            )
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            if hasattr(cfg, "quantization_config"):
                delattr(cfg, "quantization_config")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=cfg,
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

    def health_check(self) -> bool:
        """Verify model and tokenizer readiness."""
        return bool(self.model is not None and self.tokenizer is not None)

    async def generate_async(self, request: GenerationRequest) -> GenerationResponse:
        return await asyncio.to_thread(self.generate, request)

    async def generate_stream_async(self, request: GenerationRequest):
        """Asynchronously stream tokens using TextIteratorStreamer."""
        if self.model is None or self.tokenizer is None:
            raise InferenceRuntimeError("HuggingFace engine not initialized.")

        from threading import Thread

        from transformers import TextIteratorStreamer

        inputs = self.tokenizer(request.prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": request.max_new_tokens,
            "do_sample": request.temperature > 0.0,
            "temperature": max(request.temperature, 1e-4),
            "top_p": request.top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        start_time = time.perf_counter()
        first_token = True
        loop = asyncio.get_running_loop()

        def get_next_token():
            try:
                return next(streamer)
            except StopIteration:
                return None

        while True:
            token_text = await loop.run_in_executor(None, get_next_token)
            if token_text is None:
                break
            now_ms = (time.perf_counter() - start_time) * 1000.0
            yield GenerationChunk(
                delta_text=token_text,
                arrival_time_ms=now_ms,
                is_first_token=first_token,
                is_finish=False,
            )
            first_token = False

        thread.join()
        now_ms = (time.perf_counter() - start_time) * 1000.0
        yield GenerationChunk(
            delta_text="",
            arrival_time_ms=now_ms,
            is_first_token=False,
            is_finish=True,
        )

    def stop(self) -> None:
        self.model = None
        self.tokenizer = None
        from vipym.utils.resilience import safe_cuda_memory_cleanup

        safe_cuda_memory_cleanup()


InferenceRegistry.register("hf", HuggingFaceInferenceBackend)
InferenceRegistry.register("huggingface", HuggingFaceInferenceBackend)
