"""Serving Backends Package for vLLM, SGLang, and HuggingFace."""

from vipym.inference.backends.base import BaseInferenceBackend, auto_detect_quantization
from vipym.inference.backends.sglang_backend import SGLangBackend
from vipym.inference.backends.vllm_backend import VLLMBackend

__all__ = [
    "BaseInferenceBackend",
    "SGLangBackend",
    "VLLMBackend",
    "auto_detect_quantization",
]
