"""Unit tests for asynchronous token streaming and arrival telemetry across inference engines."""

import pytest

from vipym.core.exceptions import InferenceRuntimeError
from vipym.inference.hf_engine import HuggingFaceInferenceBackend
from vipym.inference.vllm_engine import VLLMInferenceBackend
from vipym.interfaces.inference import (
    GenerationChunk,
    GenerationRequest,
    GenerationResponse,
    InferenceBackend,
)


class DummyCustomBackend(InferenceBackend):
    """Minimal inference backend relying on default generate_stream_async."""

    def start(self, model_path_or_id, **kwargs):
        self.started = True

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(
            generated_text="def solve(): return 42",
            prompt_tokens=4,
            completion_tokens=5,
            time_to_first_token_ms=10.0,
            inter_token_latency_ms=2.0,
            total_time_ms=18.0,
        )

    async def generate_async(self, request: GenerationRequest) -> GenerationResponse:
        return self.generate(request)

    def stop(self):
        self.started = False


def test_generation_chunk_schema():
    chunk = GenerationChunk(
        delta_text="hello",
        token_id=123,
        arrival_time_ms=15.5,
        is_first_token=True,
        is_finish=False,
    )
    assert chunk.delta_text == "hello"
    assert chunk.token_id == 123
    assert chunk.arrival_time_ms == 15.5
    assert chunk.is_first_token is True
    assert chunk.is_finish is False


@pytest.mark.asyncio
async def test_default_stream_generator_backward_compatibility():
    backend = DummyCustomBackend()
    backend.start("test_model")

    req = GenerationRequest(prompt="Write a function")
    chunks = []
    async for chunk in backend.generate_stream_async(req):
        chunks.append(chunk)

    assert len(chunks) > 1
    assert chunks[0].is_first_token is True
    assert chunks[-1].is_finish is True

    # Timestamps should be non-decreasing
    times = [c.arrival_time_ms for c in chunks]
    assert all(times[i] <= times[i + 1] for i in range(len(times) - 1))

    reconstructed = "".join(c.delta_text for c in chunks)
    assert reconstructed == "def solve(): return 42"


@pytest.mark.asyncio
async def test_vllm_engine_streaming():
    backend = VLLMInferenceBackend()
    backend.start("test-model")

    req = GenerationRequest(prompt="Write a fibonacci function")
    chunks = []
    async for chunk in backend.generate_stream_async(req):
        chunks.append(chunk)

    assert len(chunks) > 1
    assert chunks[0].is_first_token is True
    assert chunks[-1].is_finish is True

    # Reconstructed text without final empty finish chunk matches generation
    reconstructed = "".join(c.delta_text for c in chunks)
    assert len(reconstructed) > 0
    assert "mock_solution_for_" in reconstructed

    backend.stop()


@pytest.mark.asyncio
async def test_hf_engine_streaming_uninitialized_raises():
    backend = HuggingFaceInferenceBackend()
    req = GenerationRequest(prompt="test")
    with pytest.raises(InferenceRuntimeError, match="HuggingFace engine not initialized"):
        async for _ in backend.generate_stream_async(req):
            pass


def test_inference_backend_consolidation_and_reexports():
    """Verify inference backend consolidation maintains identical class references."""
    from vipym.inference.backends.hf_backend import (
        HuggingFaceInferenceBackend as CoreHFBackend,
    )
    from vipym.inference.backends.vllm_backend import (
        VLLMBackend,
    )
    from vipym.inference.backends.vllm_backend import (
        VLLMInferenceBackend as CoreVLLMBackend,
    )
    from vipym.inference.hf_engine import (
        HuggingFaceInferenceBackend as FacadeHFBackend,
    )
    from vipym.inference.registry import InferenceRegistry
    from vipym.inference.vllm_engine import (
        VLLMBackend as FacadeVLLMBackend,
    )
    from vipym.inference.vllm_engine import (
        VLLMInferenceBackend as FacadeVLLMInferenceBackend,
    )

    assert CoreHFBackend is FacadeHFBackend
    assert VLLMBackend is CoreVLLMBackend
    assert VLLMBackend is FacadeVLLMBackend
    assert CoreVLLMBackend is FacadeVLLMInferenceBackend

    # Ensure registries resolve properly
    assert isinstance(InferenceRegistry.get("hf"), CoreHFBackend)
    assert isinstance(InferenceRegistry.get("vllm"), VLLMBackend)
    assert isinstance(InferenceRegistry.get("vllm_engine"), VLLMBackend)
    assert InferenceRegistry.get_class("hf") is CoreHFBackend
    assert InferenceRegistry.get_class("vllm") is VLLMBackend
    assert InferenceRegistry.get_class("vllm_engine") is VLLMBackend
