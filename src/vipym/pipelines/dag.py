"""Topological DAG compression pipeline engine."""

import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from vipym.config.exceptions import InvalidPipelineDAGError
from vipym.core.logger import get_logger
from vipym.interfaces.compression import CompressionArtifact, CompressionMethod, CompressionPipeline
from vipym.interfaces.model import ModelAdapter, ModelMetadata
from vipym.pipelines.node import PipelineStageNode

logger = get_logger(__name__)


class DirectedAcyclicCompressionPipeline(CompressionPipeline):
    """Topological execution engine for arbitrary non-linear compression pipelines."""

    def __init__(self) -> None:
        self.nodes: dict[str, PipelineStageNode] = {}

    def add_stage(
        self,
        stage_id: str,
        method: CompressionMethod,
        dependencies: list[str] | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> "DirectedAcyclicCompressionPipeline":
        if stage_id in self.nodes:
            raise InvalidPipelineDAGError(
                f"Duplicate stage_id '{stage_id}' in compression pipeline DAG."
            )
        self.nodes[stage_id] = PipelineStageNode(
            stage_id=stage_id,
            method=method,
            dependencies=dependencies or [],
            parameters=parameters or {},
        )
        return self

    def get_topological_order(self) -> list[str]:
        """Compute execution order using Kahn's algorithm with cycle detection."""
        in_degree: dict[str, int] = dict.fromkeys(self.nodes, 0)
        adj: dict[str, list[str]] = defaultdict(list)

        for node_id, node in self.nodes.items():
            for dep in node.dependencies:
                if dep not in self.nodes:
                    raise InvalidPipelineDAGError(
                        f"Stage '{node_id}' depends on missing prerequisite stage '{dep}'"
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
            unresolved = [n for n, d in in_degree.items() if d > 0]
            raise InvalidPipelineDAGError(
                f"Cyclic dependency detected in compression pipeline DAG! Unresolved nodes: {unresolved}"
            )

        return topo_order

    def validate_dag(self, initial_metadata: ModelMetadata) -> bool:
        """Validate applicability against model topology."""
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
        """Execute all nodes in topological order."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        order = self.get_topological_order()

        logger.info(f"Executing Compression DAG with {len(order)} stages: {' -> '.join(order)}")

        model = model_adapter.load_for_compression(model_id, revision=revision)
        tokenizer = model_adapter.get_tokenizer(model_id, revision=revision)

        current_artifact: CompressionArtifact | None = None
        applied_methods: list[str] = []

        for idx, stage_id in enumerate(order):
            node = self.nodes[stage_id]
            stage_start = time.perf_counter()
            logger.info(
                f"Starting DAG stage [{idx + 1}/{len(order)}]: '{stage_id}' ({node.method.name})"
            )

            stage_out_dir = output_dir / f"stage_{idx}_{stage_id}"
            stage_out_dir.mkdir(parents=True, exist_ok=True)

            current_artifact = node.method.compress(
                model=model,
                tokenizer=tokenizer,
                output_dir=stage_out_dir,
                **node.parameters,
            )
            node.executed = True
            node.execution_time_sec = time.perf_counter() - stage_start
            node.output_artifact = current_artifact
            applied_methods.append(node.method.name)
            logger.info(f"Completed stage '{stage_id}' in {node.execution_time_sec:.2f}s")

        if current_artifact is None:
            baseline_dir = output_dir / "uncompressed_baseline"
            baseline_dir.mkdir(parents=True, exist_ok=True)
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(baseline_dir)
            if hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(baseline_dir)
            current_artifact = CompressionArtifact(
                output_path=baseline_dir,
                format="safetensors",
                compressed_size_bytes=0,
                applied_methods=["baseline"],
            )

        current_artifact.applied_methods = applied_methods
        return current_artifact
