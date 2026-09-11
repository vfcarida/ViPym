"""Topological DAG compression pipeline engine."""

from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from vipym.config.exceptions import InvalidPipelineDAGError
from vipym.core.logger import get_logger
from vipym.interfaces.compression import CompressionArtifact, CompressionMethod, CompressionPipeline
from vipym.interfaces.model import ModelAdapter, ModelMetadata
from vipym.observability.progress import PipelineProgressTracker
from vipym.pipelines.node import PipelineStageNode

logger = get_logger(__name__)


class DirectedAcyclicCompressionPipeline(CompressionPipeline):
    """Topological execution engine for arbitrary non-linear and branching compression pipelines."""

    def __init__(self) -> None:
        self.nodes: dict[str, PipelineStageNode] = {}
        self.stage_artifacts: dict[str, CompressionArtifact] = {}
        self.leaf_artifacts: dict[str, CompressionArtifact] = {}

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

    def get_branch_artifacts(self) -> dict[str, CompressionArtifact]:
        """Return output artifacts of all terminal leaf stages in the DAG."""
        return dict(self.leaf_artifacts)

    def execute(
        self,
        model_adapter: ModelAdapter,
        model_id: str,
        output_dir: Path,
        revision: str = "main",
    ) -> CompressionArtifact:
        """Execute all nodes in topological order with branch isolation and artifact reloading."""
        import copy

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        order = self.get_topological_order()

        logger.info(f"Executing Compression DAG with {len(order)} stages: {' -> '.join(order)}")

        # Determine out-degree of each stage to identify terminal leaf nodes
        dependents_count: dict[str, int] = dict.fromkeys(self.nodes, 0)
        for node in self.nodes.values():
            for dep in node.dependencies:
                dependents_count[dep] += 1
        leaf_stage_ids = {nid for nid, cnt in dependents_count.items() if cnt == 0}

        self.stage_artifacts.clear()
        self.leaf_artifacts.clear()
        stage_models: dict[str, Any] = {}

        tracker = PipelineProgressTracker(
            total_stages=len(order),
            pipeline_name="compression_dag",
            pipeline_id=f"dag_{model_id}",
        )

        current_artifact: CompressionArtifact | None = None
        applied_methods: list[str] = []

        for idx, stage_id in enumerate(order):
            node = self.nodes[stage_id]
            logger.info(
                f"Starting DAG stage [{idx + 1}/{len(order)}]: '{stage_id}' ({node.method.name})"
            )
            tracker.start_stage(stage_name=stage_id, stage_type=node.method.name)

            stage_out_dir = output_dir / f"stage_{idx}_{stage_id}"
            stage_out_dir.mkdir(parents=True, exist_ok=True)

            # Load model corresponding to parent dependency (or base model if root)
            if not node.dependencies:
                stage_model = model_adapter.load_for_compression(model_id, revision=revision)
                stage_tokenizer = model_adapter.get_tokenizer(model_id, revision=revision)
            else:
                parent_id = node.dependencies[0]
                parent_artifact = self.stage_artifacts[parent_id]
                parent_path = str(parent_artifact.output_path)

                stage_model = None
                try:
                    stage_model = model_adapter.load_for_compression(parent_path, revision=revision)
                except Exception as load_err:
                    logger.debug(
                        f"Adapter could not load from artifact path '{parent_path}' ({load_err}); falling back to parent clone."
                    )

                if stage_model is None:
                    if parent_id in stage_models:
                        try:
                            stage_model = copy.deepcopy(stage_models[parent_id])
                        except Exception:
                            stage_model = model_adapter.load_for_compression(
                                model_id, revision=revision
                            )
                    else:
                        stage_model = model_adapter.load_for_compression(
                            model_id, revision=revision
                        )

                try:
                    stage_tokenizer = model_adapter.get_tokenizer(parent_path, revision=revision)
                except Exception:
                    stage_tokenizer = model_adapter.get_tokenizer(model_id, revision=revision)

            # Build multi-parent context for merging nodes
            parent_artifacts = {
                dep: self.stage_artifacts[dep]
                for dep in node.dependencies
                if dep in self.stage_artifacts
            }
            parent_models_dict = {
                dep: stage_models[dep] for dep in node.dependencies if dep in stage_models
            }

            stage_kwargs = dict(node.parameters)
            if len(node.dependencies) > 1:
                stage_kwargs["parent_artifacts"] = parent_artifacts
                stage_kwargs["parent_models"] = parent_models_dict

            try:
                current_artifact = node.method.compress(
                    model=stage_model,
                    tokenizer=stage_tokenizer,
                    output_dir=stage_out_dir,
                    **stage_kwargs,
                )
                node.executed = True
                duration = tracker.complete_stage(
                    stage_name=stage_id,
                    metrics={"method": node.method.name},
                )
                node.execution_time_sec = duration
                node.output_artifact = current_artifact
                self.stage_artifacts[stage_id] = current_artifact
                stage_models[stage_id] = stage_model
                applied_methods.append(node.method.name)

                if stage_id in leaf_stage_ids:
                    self.leaf_artifacts[stage_id] = current_artifact

                from vipym.utils.resilience import safe_cuda_memory_cleanup

                safe_cuda_memory_cleanup()
                logger.info(f"Completed stage '{stage_id}' in {duration:.2f}s")
            except Exception as e:
                tracker.fail_stage(stage_name=stage_id, error=e)
                raise

        if current_artifact is None:
            baseline_dir = output_dir / "uncompressed_baseline"
            baseline_dir.mkdir(parents=True, exist_ok=True)
            base_m = model_adapter.load_for_compression(model_id, revision=revision)
            base_t = model_adapter.get_tokenizer(model_id, revision=revision)
            if hasattr(base_m, "save_pretrained"):
                base_m.save_pretrained(baseline_dir)
            if hasattr(base_t, "save_pretrained"):
                base_t.save_pretrained(baseline_dir)
            current_artifact = CompressionArtifact(
                output_path=baseline_dir,
                format="safetensors",
                compressed_size_bytes=0,
                applied_methods=["baseline"],
            )

        # Attach branch metadata to primary artifact
        if self.leaf_artifacts:
            current_artifact.metadata["branches"] = {
                k: str(v.output_path) for k, v in self.leaf_artifacts.items()
            }
        current_artifact.applied_methods = applied_methods
        return current_artifact


# Alias for cross-namespace consistency
DAGCompressionPipeline = DirectedAcyclicCompressionPipeline
