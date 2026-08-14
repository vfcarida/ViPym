"""DAG-based Compression Pipeline orchestrator."""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vipym.core.exceptions import InvalidPipelineDAGError
from vipym.core.logger import get_logger
from vipym.interfaces.compression import CompressionArtifact, CompressionMethod, CompressionPipeline
from vipym.interfaces.model import ModelAdapter, ModelMetadata

logger = get_logger(__name__)


@dataclass
class PipelineNode:
    stage_id: str
    method: CompressionMethod
    dependencies: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)


class DAGCompressionPipeline(CompressionPipeline):
    """Execution engine for Directed Acyclic Graph (DAG) compression workflows."""

    def __init__(self) -> None:
        self.nodes: dict[str, PipelineNode] = {}

    def add_stage(
        self,
        stage_id: str,
        method: CompressionMethod,
        dependencies: list[str] | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> "DAGCompressionPipeline":
        if stage_id in self.nodes:
            raise InvalidPipelineDAGError(f"Duplicate stage_id in pipeline: '{stage_id}'")
        self.nodes[stage_id] = PipelineNode(
            stage_id=stage_id,
            method=method,
            dependencies=dependencies or [],
            parameters=parameters or {},
        )
        return self

    def get_topological_order(self) -> list[str]:
        """Compute valid topological execution order using Kahn's algorithm."""
        in_degree: dict[str, int] = dict.fromkeys(self.nodes, 0)
        adj: dict[str, list[str]] = defaultdict(list)

        for node_id, node in self.nodes.items():
            for dep in node.dependencies:
                if dep not in self.nodes:
                    raise InvalidPipelineDAGError(
                        f"Stage '{node_id}' depends on non-existent stage '{dep}'"
                    )
                adj[dep].append(node_id)
                in_degree[node_id] += 1

        queue = deque([node_id for node_id, deg in in_degree.items() if deg == 0])
        topo_order: list[str] = []

        while queue:
            curr = queue.popleft()
            topo_order.append(curr)
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(topo_order) != len(self.nodes):
            raise InvalidPipelineDAGError("Cycle detected in compression pipeline DAG!")

        return topo_order

    def validate_dag(self, initial_metadata: ModelMetadata) -> bool:
        """Validate pipeline feasibility and applicability across all nodes."""
        order = self.get_topological_order()
        for stage_id in order:
            node = self.nodes[stage_id]
            node.method.validate_applicability(initial_metadata)
        return True

    def execute(
        self,
        model_adapter: ModelAdapter,
        model_id: str,
        output_dir: Path,
        revision: str = "main",
    ) -> CompressionArtifact:
        """Execute each DAG stage in topological order."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        order = self.get_topological_order()

        logger.info(f"Executing compression pipeline with {len(order)} stages in order: {order}")

        # Load initial model
        model = model_adapter.load_for_compression(model_id, revision=revision)
        tokenizer = model_adapter.get_tokenizer(model_id, revision=revision)

        current_artifact: CompressionArtifact | None = None
        applied_methods: list[str] = []

        for idx, stage_id in enumerate(order):
            node = self.nodes[stage_id]
            logger.info(
                f"Running compression stage [{idx + 1}/{len(order)}]: '{stage_id}' ({node.method.name})"
            )

            stage_out_dir = output_dir / f"stage_{idx}_{stage_id}"
            stage_out_dir.mkdir(parents=True, exist_ok=True)

            current_artifact = node.method.compress(
                model=model,
                tokenizer=tokenizer,
                output_dir=stage_out_dir,
                **node.parameters,
            )
            applied_methods.append(node.method.name)

        if current_artifact is None:
            # Empty pipeline (baseline case)
            model_save_dir = output_dir / "baseline"
            model_save_dir.mkdir(parents=True, exist_ok=True)
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(model_save_dir)
            if hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(model_save_dir)

            current_artifact = CompressionArtifact(
                output_path=model_save_dir,
                format="safetensors",
                compressed_size_bytes=0,
                applied_methods=["baseline"],
            )

        current_artifact.applied_methods = applied_methods
        return current_artifact
