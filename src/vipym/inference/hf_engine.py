"""HuggingFace and fallback inference engines (backward-compatible facade)."""

from vipym.inference.backends.hf_backend import HuggingFaceInferenceBackend
from vipym.inference.backends.sglang_backend import SGLangBackend

# Backward compatibility alias
SGLangInferenceBackend = SGLangBackend

__all__ = ["HuggingFaceInferenceBackend", "SGLangBackend", "SGLangInferenceBackend"]
