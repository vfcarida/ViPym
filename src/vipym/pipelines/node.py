"""Pipeline Node and Edge representations."""

from dataclasses import dataclass, field
from typing import Any

from vipym.interfaces.compression import CompressionMethod


@dataclass
class PipelineStageNode:
    """A discrete node in the compression Directed Acyclic Graph."""

    stage_id: str
    method: CompressionMethod
    dependencies: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    executed: bool = False
    output_artifact: Any | None = None
    execution_time_sec: float = 0.0
