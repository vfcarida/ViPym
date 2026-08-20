"""Compression package with auto-registration."""

from vipym.compression import (
    distillation as distillation,
)
from vipym.compression import (
    kv_cache as kv_cache,
)
from vipym.compression import (
    methods as methods,
)
from vipym.compression import (
    pruning as pruning,
)
from vipym.compression import (
    quantization as quantization,
)
from vipym.compression import (
    transforms as transforms,
)
from vipym.compression.pipeline import DAGCompressionPipeline
from vipym.compression.registry import CompressionRegistry

__all__ = [
    "CompressionRegistry",
    "DAGCompressionPipeline",
    "distillation",
    "kv_cache",
    "methods",
    "pruning",
    "quantization",
    "transforms",
]
