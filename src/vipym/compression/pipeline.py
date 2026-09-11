"""DAG-based Compression Pipeline orchestrator.

This module is retained for backwards compatibility. Canonical DAG pipeline
orchestration is implemented in `vipym.pipelines.dag`.
"""

from vipym.pipelines.dag import (
    DAGCompressionPipeline,
    DirectedAcyclicCompressionPipeline,
)
from vipym.pipelines.node import PipelineStageNode

# Alias PipelineNode to canonical PipelineStageNode for backwards compatibility
PipelineNode = PipelineStageNode

__all__ = [
    "DAGCompressionPipeline",
    "DirectedAcyclicCompressionPipeline",
    "PipelineNode",
]
