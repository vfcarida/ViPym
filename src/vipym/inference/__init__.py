"""Production Serving and Inference Backends."""

from vipym.inference.backends.base import BaseInferenceBackend, auto_detect_quantization
from vipym.inference.backends.hf_backend import HuggingFaceInferenceBackend
from vipym.inference.backends.sglang_backend import SGLangBackend
from vipym.inference.backends.vllm_backend import VLLMBackend, VLLMInferenceBackend
from vipym.inference.batch import BatchInferenceRunner
from vipym.inference.registry import InferenceRegistry
from vipym.inference.speculative import (
    SpeculativeDecodingHarness,
    SpeculativeInferenceBackend,
    SpeculativeMetrics,
)

__all__ = [
    "BaseInferenceBackend",
    "BatchInferenceRunner",
    "HuggingFaceInferenceBackend",
    "InferenceRegistry",
    "SGLangBackend",
    "SpeculativeDecodingHarness",
    "SpeculativeInferenceBackend",
    "SpeculativeMetrics",
    "VLLMBackend",
    "VLLMInferenceBackend",
    "auto_detect_quantization",
]
