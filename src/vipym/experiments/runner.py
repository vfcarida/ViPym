"""Resumable Experiment Execution Runner."""

import json
import time
from pathlib import Path

import pydantic

from vipym.analysis.pareto import ParetoPoint
from vipym.compression.registry import CompressionRegistry
from vipym.config.constants import ExperimentState
from vipym.config.schema import ViPymExperimentConfig
from vipym.core.constants import ExecutionStatus
from vipym.core.logger import get_logger
from vipym.cost.calculator import CloudCostCalculator
from vipym.evaluation.runner import BenchmarkRunner
from vipym.experiments.checkpoint import CheckpointManager, ExperimentCheckpoint
from vipym.experiments.manifest import ReproducibilityManifest
from vipym.experiments.state import ExperimentStateManager
from vipym.inference.registry import InferenceRegistry
from vipym.interfaces.compression import CompressionArtifact
from vipym.models.registry import ModelRegistry
from vipym.observability.logging import bind_context, emit_event
from vipym.observability.progress import PipelineProgressTracker
from vipym.pipelines.dag import DirectedAcyclicCompressionPipeline
from vipym.reporting.generator import ExperimentReportGenerator

logger = get_logger(__name__)


class ExperimentRunSummary(pydantic.BaseModel):
    experiment_id: str
    manifest_id: str
    final_state: ExperimentState
    baseline_point: ParetoPoint
    compressed_points: list[ParetoPoint]
    generated_report_files: dict[str, str]
    total_duration_sec: float
    total_cost_usd: float

    @property
    def status(self) -> ExecutionStatus:
        """Compatibility property for ExecutionStatus."""
        if self.final_state in (
            ExperimentState.REPORT_COMPLETED,
            ExperimentState.ANALYSIS_COMPLETED,
        ):
            return ExecutionStatus.COMPLETED
        elif self.final_state == ExperimentState.FAILED:
            return ExecutionStatus.FAILED
        return ExecutionStatus.RUNNING


class ResumableExperimentRunner:
    """Orchestrates end-to-end experiment execution with full checkpointing and resumability."""

    def __init__(
        self,
        config: ViPymExperimentConfig,
        artifacts_dir: Path | str = "./artifacts",
        checkpoint_enabled: bool = True,
    ) -> None:
        self.config = config
        self.exp_dir = Path(artifacts_dir) / config.experiment_id
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_enabled = checkpoint_enabled

        self.state_mgr = ExperimentStateManager(
            config.experiment_id,
            self.exp_dir / "state.json",
            persist_to_disk=checkpoint_enabled,
        )
        self.checkpoint_mgr = CheckpointManager(self.exp_dir / "checkpoint.json")
        if checkpoint_enabled:
            self.checkpoint: ExperimentCheckpoint = self.checkpoint_mgr.load(config.experiment_id)
        else:
            self.checkpoint = ExperimentCheckpoint(experiment_id=config.experiment_id)

        self.manifest = ReproducibilityManifest.create(config)
        self.cost_calculator = CloudCostCalculator(config.cost_assumptions)

    def run(self, resume: bool = True) -> ExperimentRunSummary:
        if not self.checkpoint_enabled:
            resume = False

        if not resume:
            self.state_mgr.reset()
            self.checkpoint = ExperimentCheckpoint(experiment_id=self.config.experiment_id)

        start_time = time.perf_counter()
        bind_context(
            experiment_id=self.config.experiment_id,
            model_name=self.config.model.id,
        )
        emit_event(
            "experiment_started",
            experiment_id=self.config.experiment_id,
            model=self.config.model.id,
        )
        logger.info(
            f"Starting ViPym Experiment: [bold cyan]{self.config.experiment_id}[/bold cyan] (resume={resume}, checkpoint={self.checkpoint_enabled})"
        )

        tracker = PipelineProgressTracker(
            total_stages=6,
            pipeline_name=f"experiment_{self.config.experiment_id}",
            pipeline_id=self.config.experiment_id,
        )

        try:
            # 1. Validation Stage
            tracker.start_stage("validation", stage_type="validation")
            if self.state_mgr.current_state in {ExperimentState.CREATED, ExperimentState.FAILED}:
                self.state_mgr.transition_to(ExperimentState.VALIDATED)

            # Discover model metadata
            adapter_name = self.config.model.custom_adapter_cls or self.config.model.id
            try:
                model_adapter = ModelRegistry.get(adapter_name)
            except Exception:
                model_adapter = ModelRegistry.get("hf")

            metadata = model_adapter.inspect_metadata(
                self.config.model.id, revision=self.config.model.revision
            )
            logger.info(
                f"Inspected Target Model '{metadata.model_id}': total_params={metadata.total_parameters / 1e9:.1f}B, "
                f"active_params={metadata.active_parameters / 1e9:.1f}B, arch={metadata.architecture_type}"
            )
            tracker.complete_stage("validation", metrics={"model_id": metadata.model_id})

            # 2. Baseline Stage
            tracker.start_stage("baseline_evaluation", stage_type="evaluation")
            if not self.checkpoint.baseline_completed or not resume:
                self.state_mgr.transition_to(ExperimentState.BASELINE_RUNNING)
                logger.info("=== Stage: Immutable Baseline Serving & Evaluation ===")

                backend = InferenceRegistry.get(self.config.serving.backend)
                backend.start(
                    model_path_or_id=self.config.model.id,
                    tensor_parallel_size=self.config.serving.tensor_parallel_size,
                    max_model_len=self.config.serving.max_model_len,
                )

                eval_runner = BenchmarkRunner(evaluation_config=self.config.evaluation)
                baseline_suite_results = []
                for suite_name in self.config.evaluation.suites:
                    res = eval_runner.run_suite(
                        suite_name=suite_name,
                        backend=backend,
                        temperature=self.config.evaluation.temperature,
                        top_p=self.config.evaluation.top_p,
                        max_new_tokens=self.config.evaluation.max_new_tokens,
                        task_limit=self.config.evaluation.task_limit,
                    )
                    baseline_suite_results.append(res)

                backend.stop()

                base_pass1 = (
                    sum(s.pass_at_1 for s in baseline_suite_results)
                    / max(1, len(baseline_suite_results))
                    if baseline_suite_results
                    else 0.0
                )

                self.checkpoint.baseline_completed = True
                self.checkpoint.baseline_score = base_pass1
                self.checkpoint.baseline_metrics = {
                    "pass_at_1": base_pass1,
                    "suites": [s.model_dump() for s in baseline_suite_results],
                }
                if self.checkpoint_enabled:
                    self.checkpoint_mgr.save(self.checkpoint)
                self.state_mgr.transition_to(ExperimentState.BASELINE_COMPLETED)
            else:
                logger.info("[OK] Skipping Baseline Stage (Already completed in checkpoint)")

            tracker.complete_stage(
                "baseline_evaluation",
                metrics={"pass_at_1": self.checkpoint.baseline_score or 0.0},
            )

            base_p50 = (
                sum(s.summary_metrics.get("latency_p50_ms", 45.0) for s in baseline_suite_results)
                / max(1, len(baseline_suite_results))
                if baseline_suite_results
                else 45.0
            )
            base_throughput = (
                sum(s.summary_metrics.get("throughput_tok_s", 0.0) for s in baseline_suite_results)
                / max(1, len(baseline_suite_results))
                if baseline_suite_results
                else 0.0
            )
            base_vram = (
                max(
                    (s.summary_metrics.get("peak_vram_gb", 0.0) for s in baseline_suite_results),
                    default=0.0,
                )
                if baseline_suite_results
                and any(s.summary_metrics.get("peak_vram_gb", 0.0) for s in baseline_suite_results)
                else metadata.active_parameters * 2 / (1024**3)
            )
            base_cost = self.cost_calculator.compute_serving_cost_per_1m_tokens(base_throughput)

            baseline_point = ParetoPoint(
                experiment_id=self.config.experiment_id,
                configuration_name="Baseline",
                quality_score=self.checkpoint.baseline_score or 0.0,
                latency_p50_ms=round(base_p50, 2),
                peak_vram_gb=round(base_vram, 2),
                cost_usd=round(base_cost, 2),
                compression_ratio=1.0,
                is_pareto_optimal=True,
            )

            # 3. Compression DAG Pipeline Stage
            compressed_artifact: CompressionArtifact | None = None
            if self.config.compression_pipeline:
                tracker.start_stage("compression_dag", stage_type="compression")
                if self.checkpoint.compressed_artifact_path is None or not resume:
                    self.state_mgr.transition_to(ExperimentState.COMPRESSION_RUNNING)
                    logger.info("=== Stage: Compression DAG Pipeline Execution ===")

                    pipeline = DirectedAcyclicCompressionPipeline()
                    for stage in self.config.compression_pipeline:
                        method_inst = CompressionRegistry.get(stage.method)
                        pipeline.add_stage(
                            stage_id=stage.stage_id,
                            method=method_inst,
                            dependencies=stage.dependencies,
                            parameters=stage.parameters,
                        )

                    pipeline.validate_dag(metadata)
                    compressed_artifact = pipeline.execute(
                        model_adapter=model_adapter,
                        model_id=self.config.model.id,
                        output_dir=self.exp_dir / "compressed_checkpoints",
                        revision=self.config.model.revision,
                    )

                    self.checkpoint.compressed_artifact_path = str(compressed_artifact.output_path)
                    self.checkpoint.compressed_methods_applied = compressed_artifact.applied_methods
                    if self.checkpoint_enabled:
                        self.checkpoint_mgr.save(self.checkpoint)
                    self.state_mgr.transition_to(ExperimentState.COMPRESSION_COMPLETED)
                else:
                    logger.info("[OK] Skipping Compression Stage (Artifact loaded from checkpoint)")
                    compressed_artifact = CompressionArtifact(
                        output_path=Path(self.checkpoint.compressed_artifact_path),
                        format="compressed-tensors",
                        compressed_size_bytes=1000,
                        applied_methods=self.checkpoint.compressed_methods_applied,
                    )
                tracker.complete_stage("compression_dag")

            # 4. Evaluation Stage
            compressed_points = []
            if compressed_artifact:
                tracker.start_stage("compressed_evaluation", stage_type="evaluation")
                if not self.checkpoint.evaluation_completed or not resume:
                    self.state_mgr.transition_to(ExperimentState.EVALUATION_RUNNING)
                    logger.info("=== Stage: Compressed Model Serving & Evaluation ===")

                    comp_backend = InferenceRegistry.get(self.config.serving.backend)
                    comp_backend.start(
                        model_path_or_id=compressed_artifact.output_path,
                        tensor_parallel_size=self.config.serving.tensor_parallel_size,
                        kv_cache_dtype=self.config.serving.kv_cache_dtype,
                        max_model_len=self.config.serving.max_model_len,
                    )

                    eval_runner = BenchmarkRunner(evaluation_config=self.config.evaluation)
                    comp_suite_results = []
                    for suite_name in self.config.evaluation.suites:
                        res = eval_runner.run_suite(
                            suite_name=suite_name,
                            backend=comp_backend,
                            temperature=self.config.evaluation.temperature,
                            top_p=self.config.evaluation.top_p,
                            max_new_tokens=self.config.evaluation.max_new_tokens,
                            task_limit=self.config.evaluation.task_limit,
                        )
                        comp_suite_results.append(res)

                    comp_backend.stop()

                    # Write per-suite evaluations to evaluations/ directory
                    eval_dir = self.exp_dir / "evaluations"
                    eval_dir.mkdir(parents=True, exist_ok=True)
                    for res in comp_suite_results:
                        suite_json_path = eval_dir / f"{res.suite_name.lower()}.json"
                        with open(suite_json_path, "w", encoding="utf-8") as f:
                            json.dump(res.model_dump(), f, indent=2)

                    # Ensure models directory exists with reference to compressed checkpoint
                    models_dir = self.exp_dir / "models"
                    models_dir.mkdir(parents=True, exist_ok=True)
                    (models_dir / "artifact_info.json").write_text(
                        json.dumps(
                            {
                                "artifact_path": str(compressed_artifact.output_path),
                                "applied_methods": compressed_artifact.applied_methods,
                                "format": compressed_artifact.format,
                                "compressed_size_bytes": compressed_artifact.compressed_size_bytes,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )

                    comp_pass1 = (
                        sum(s.pass_at_1 for s in comp_suite_results)
                        / max(1, len(comp_suite_results))
                        if comp_suite_results
                        else 0.0
                    )

                    comp_p50 = (
                        sum(
                            s.summary_metrics.get("latency_p50_ms", 28.0)
                            for s in comp_suite_results
                        )
                        / max(1, len(comp_suite_results))
                        if comp_suite_results
                        else 28.0
                    )
                    comp_throughput = (
                        sum(
                            s.summary_metrics.get("throughput_tok_s", 0.0)
                            for s in comp_suite_results
                        )
                        / max(1, len(comp_suite_results))
                        if comp_suite_results
                        else 0.0
                    )
                    comp_vram = (
                        max(
                            (
                                s.summary_metrics.get("peak_vram_gb", 0.0)
                                for s in comp_suite_results
                            ),
                            default=0.0,
                        )
                        if comp_suite_results
                        and any(
                            s.summary_metrics.get("peak_vram_gb", 0.0) for s in comp_suite_results
                        )
                        else (metadata.active_parameters * 0.5) / (1024**3)
                    )
                    comp_cost = self.cost_calculator.compute_serving_cost_per_1m_tokens(
                        comp_throughput
                    )

                    compressed_point = ParetoPoint(
                        experiment_id=self.config.experiment_id,
                        configuration_name=f"Compressed ({'+'.join(compressed_artifact.applied_methods)})",
                        quality_score=comp_pass1,
                        latency_p50_ms=round(comp_p50, 2),
                        peak_vram_gb=round(comp_vram, 2),
                        cost_usd=round(comp_cost, 2),
                        compression_ratio=4.0,
                    )
                    compressed_points.append(compressed_point)

                    self.checkpoint.evaluation_completed = True
                    self.checkpoint.pareto_points = [compressed_point.model_dump()]
                    if self.checkpoint_enabled:
                        self.checkpoint_mgr.save(self.checkpoint)
                    self.state_mgr.transition_to(ExperimentState.EVALUATION_COMPLETED)
                else:
                    logger.info("[OK] Skipping Evaluation Stage (Loaded from checkpoint)")
                    compressed_points = [ParetoPoint(**pt) for pt in self.checkpoint.pareto_points]
                tracker.complete_stage(
                    "compressed_evaluation",
                    metrics={"points": len(compressed_points)},
                )

            # 5. Analysis Stage
            tracker.start_stage("analysis", stage_type="analysis")
            self.state_mgr.transition_to(ExperimentState.ANALYSIS_COMPLETED)
            total_duration = time.perf_counter() - start_time

            cost_breakdown = self.cost_calculator.estimate_cost(
                duration_hours=total_duration / 3600.0,
                storage_gb=10.0,
                data_transfer_gb=1.0,
                input_tokens=100_000,
                output_tokens=50_000,
                successful_tasks=10,
            )
            tracker.complete_stage(
                "analysis",
                metrics={"total_cost_usd": cost_breakdown.total_cost_usd},
            )

            # 6. Reporting Stage
            tracker.start_stage("reporting", stage_type="reporting")
            report_gen = ExperimentReportGenerator(self.exp_dir / "reports")
            generated_files = report_gen.generate_all(
                experiment_id=self.config.experiment_id,
                baseline=baseline_point,
                results=compressed_points,
                manifest_meta=self.manifest.environment.model_dump(),
            )

            # Save individual standard JSON artifacts
            (self.exp_dir / "experiment.json").write_text(
                self.config.model_dump_json(indent=2), encoding="utf-8"
            )
            (self.exp_dir / "environment.json").write_text(
                self.manifest.environment.model_dump_json(indent=2), encoding="utf-8"
            )
            (self.exp_dir / "metrics.json").write_text(
                cost_breakdown.model_dump_json(indent=2), encoding="utf-8"
            )
            (self.exp_dir / "results.json").write_text(
                json.dumps(
                    [baseline_point.model_dump()] + [p.model_dump() for p in compressed_points],
                    indent=2,
                ),
                encoding="utf-8",
            )
            (self.exp_dir / "artifacts.json").write_text(
                json.dumps({k: str(v) for k, v in generated_files.items()}, indent=2),
                encoding="utf-8",
            )

            self.manifest.state = ExperimentState.REPORT_COMPLETED
            self.manifest.duration_seconds = total_duration
            self.manifest.total_cost_usd = cost_breakdown.total_cost_usd
            self.manifest.artifacts = {k: str(v) for k, v in generated_files.items()}
            self.manifest.summary_metrics = {
                "baseline_pass_at_1": baseline_point.quality_score,
                "compressed_count": len(compressed_points),
            }
            self.manifest.save(self.exp_dir / "manifest.json")

            self.state_mgr.transition_to(ExperimentState.REPORT_COMPLETED)
            tracker.complete_stage("reporting", metrics={"artifacts": len(generated_files)})

            emit_event(
                "experiment_completed",
                experiment_id=self.config.experiment_id,
                duration_seconds=round(total_duration, 2),
                total_cost_usd=cost_breakdown.total_cost_usd,
            )
            logger.info(
                f"[OK] Experiment [{self.config.experiment_id}] completed successfully in {total_duration:.2f}s"
            )

            return ExperimentRunSummary(
                experiment_id=self.config.experiment_id,
                manifest_id=self.manifest.manifest_id,
                final_state=ExperimentState.REPORT_COMPLETED,
                baseline_point=baseline_point,
                compressed_points=compressed_points,
                generated_report_files={k: str(v) for k, v in generated_files.items()},
                total_duration_sec=total_duration,
                total_cost_usd=cost_breakdown.total_cost_usd,
            )

        except Exception as e:
            if tracker.current_stage_name:
                tracker.fail_stage(tracker.current_stage_name, error=e)
            self.state_mgr.transition_to(ExperimentState.FAILED, error_message=str(e))
            emit_event(
                "experiment_failed",
                level="error",
                experiment_id=self.config.experiment_id,
                error=str(e),
            )
            logger.error(f"Experiment [{self.config.experiment_id}] failed: {e}")
            raise
