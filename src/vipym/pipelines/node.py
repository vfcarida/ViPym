"""Pipeline Node and Edge representations."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from vipym.interfaces.compression import CompressionMethod


@dataclass
class PipelineStageNode:
    """A discrete node in the compression Directed Acyclic Graph."""
    stage_id: str
    method: CompressionMethod
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    executed: bool = False
    output_artifact: Optional[Any] = None
    execution_time_sec: float = 0.0
