"""Compression package with auto-registration."""

import vipym.compression.distillation
import vipym.compression.kv_cache
import vipym.compression.pruning
import vipym.compression.quantization
import vipym.compression.transforms
from vipym.compression.pipeline import DAGCompressionPipeline
from vipym.compression.registry import CompressionRegistry

__all__ = ["CompressionRegistry", "DAGCompressionPipeline"]
