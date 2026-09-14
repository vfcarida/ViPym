"""vLLM high-performance inference engine wrapper (backward-compatible facade)."""

from vipym.inference.backends.vllm_backend import VLLMBackend, VLLMInferenceBackend

__all__ = ["VLLMBackend", "VLLMInferenceBackend"]
