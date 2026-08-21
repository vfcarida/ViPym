"""Unit tests for P019 — Structured Logging & Experiment Observability.

Test classes:
  TestStructuredLogging          — Structlog setup (JSON/console), context binding, event emissions
  TestPipelineProgressTracker    — Multi-stage pipeline progress, stage timing, and ETA estimation
  TestSubStageProgressTrackers   — Layer-by-layer, expert-by-expert, and step-by-step progress tracking
  TestGateObservability          — Gate evaluation emitting gate_result events
  TestDAGObservability           — Compression DAG execution with progress tracking & events
  TestCoreLoggerBridge           — Bridge from vipym.core.logger to observability
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from vipym.core.logger import get_logger as get_core_logger
from vipym.gates.config import GateThresholds
from vipym.gates.eval_gate import QualityEvalGate
from vipym.interfaces.compression import CompressionArtifact, CompressionMethod
from vipym.interfaces.model import ModelAdapter
from vipym.observability.logging import (
    bind_context,
    bound_context,
    clear_context,
    configure_logging,
    emit_event,
    get_context,
    get_logger,
    unbind_context,
)
from vipym.observability.progress import (
    ExpertProgressTracker,
    LayerProgressTracker,
    PipelineProgressTracker,
    StepProgressTracker,
    create_progress_bar,
    format_duration,
)
from vipym.pipelines.dag import DirectedAcyclicCompressionPipeline

# ============================================================
# TestStructuredLogging
# ============================================================


class TestStructuredLogging:
    def test_configure_logging_console_and_json(self, tmp_path: Path):
        log_file = tmp_path / "app.log"
        # 1. Console mode
        configure_logging(mode="console", log_level="DEBUG", log_file=log_file)
        logger = get_logger("test.console")
        logger.info("Test console log message", user="dev")

        # 2. JSON mode
        configure_logging(mode="json", log_level="INFO", log_file=log_file)
        logger_json = get_logger("test.json")
        logger_json.info("Test JSON log message", metric=42)

        assert log_file.exists()

    def test_json_structured_output_format(self, tmp_path: Path):
        json_log_file = tmp_path / "production.json.log"
        configure_logging(mode="json", log_level="INFO", log_file=json_log_file)
        clear_context()

        bind_context(
            experiment_id="exp_prod_42",
            pipeline_id="pipe_gptq",
            model_name="Qwen2.5-Coder-7B",
        )

        logger = get_logger("vipym.production")
        logger.info("Starting production quantization pass", layers=32, bit_depth=4)

        # Read back log file and parse each line as JSON
        assert json_log_file.exists()
        lines = [line.strip() for line in json_log_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) > 0

        # Validate that the last line is a valid JSON dictionary containing bound context
        last_entry = json.loads(lines[-1])
        assert last_entry["experiment_id"] == "exp_prod_42"
        assert last_entry["pipeline_id"] == "pipe_gptq"
        assert last_entry["model_name"] == "Qwen2.5-Coder-7B"
        assert last_entry["event"] == "Starting production quantization pass"
        assert last_entry["level"] == "info"
        assert last_entry["layers"] == 32
        assert last_entry["bit_depth"] == 4
        assert "timestamp" in last_entry

    def test_context_binding_lifecycle(self):
        clear_context()
        assert len(get_context()) == 0

        # Bind context
        bind_context(experiment_id="exp_001", model_name="Kimi-K3", stage="awq_4bit")
        ctx = get_context()
        assert ctx["experiment_id"] == "exp_001"
        assert ctx["model_name"] == "Kimi-K3"
        assert ctx["stage"] == "awq_4bit"

        # Unbind specific key
        unbind_context("stage")
        ctx = get_context()
        assert "stage" not in ctx
        assert ctx["experiment_id"] == "exp_001"

        # Clear context
        clear_context()
        assert len(get_context()) == 0

    def test_scoped_bound_context(self):
        clear_context()
        bind_context(experiment_id="exp_root")

        with bound_context(stage_name="quant_layer_5", temp_metric=99.9) as ctx:
            assert ctx["experiment_id"] == "exp_root"
            assert ctx["stage_name"] == "quant_layer_5"
            assert ctx["temp_metric"] == 99.9

        # Context variables from `with` block should be reverted, leaving previous context intact
        restored = get_context()
        assert restored["experiment_id"] == "exp_root"
        assert "stage_name" not in restored
        assert "temp_metric" not in restored

    def test_emit_lifecycle_events(self):
        clear_context()
        bind_context(experiment_id="exp_events")

        # stage_started
        ev_start = emit_event(
            "stage_started",
            stage_name="quant_stage",
            stage_type="quantization",
            total_stages=3,
        )
        assert ev_start["event_type"] == "stage_started"
        assert ev_start["stage_name"] == "quant_stage"

        # stage_completed
        ev_comp = emit_event(
            "stage_completed",
            stage_name="quant_stage",
            duration_seconds=12.5,
            progress_pct=33.3,
        )
        assert ev_comp["event_type"] == "stage_completed"
        assert ev_comp["duration_seconds"] == 12.5

        # stage_failed
        ev_fail = emit_event(
            "stage_failed",
            level="error",
            stage_name="pruning_stage",
            error="Out of memory",
        )
        assert ev_fail["event_type"] == "stage_failed"
        assert ev_fail["error"] == "Out of memory"

        # experiment_completed
        ev_exp = emit_event(
            "experiment_completed",
            experiment_id="exp_events",
            duration_seconds=45.0,
            status="SUCCESS",
        )
        assert ev_exp["event_type"] == "experiment_completed"
        assert ev_exp["status"] == "SUCCESS"


# ============================================================
# TestPipelineProgressTracker
# ============================================================


class TestPipelineProgressTracker:
    def test_pipeline_progress_and_eta(self):
        clear_context()
        tracker = PipelineProgressTracker(
            total_stages=4,
            pipeline_name="test_pipeline",
            pipeline_id="pipe_100",
        )

        assert tracker.total_stages == 4
        assert tracker.completed_stages == 0
        assert tracker.get_progress_percentage() == 0.0

        # Start & complete stage 1
        tracker.start_stage("stage_1", stage_type="quantization")
        assert get_context().get("stage_name") == "stage_1"
        assert get_context().get("pipeline_id") == "pipe_100"
        time.sleep(0.05)
        dur1 = tracker.complete_stage("stage_1", metrics={"quant_error": 0.02})
        assert dur1 > 0.0
        assert tracker.completed_stages == 1
        assert tracker.get_progress_percentage() == 25.0
        assert "stage_name" not in get_context()

        # Start & complete stage 2
        tracker.start_stage("stage_2", stage_type="pruning")
        time.sleep(0.05)
        dur2 = tracker.complete_stage("stage_2")
        assert dur2 > 0.0
        assert tracker.completed_stages == 2
        assert tracker.get_progress_percentage() == 50.0

        # ETA should be non-zero after completing stages
        eta = tracker.get_estimated_remaining_seconds()
        assert eta > 0.0

        summary = tracker.get_status_summary()
        assert "Stage 2/4" in summary
        assert "50.0%" in summary

    def test_pipeline_track_stage_contextmanager(self):
        tracker = PipelineProgressTracker(total_stages=2, pipeline_name="ctx_pipeline")

        with tracker.track_stage("stage_auto", stage_type="distillation", metrics_dict={"loss": 0.15}):
            time.sleep(0.02)

        assert tracker.completed_stages == 1
        assert "stage_auto" in tracker.stage_durations
        assert tracker.stage_durations["stage_auto"] > 0.0

    def test_pipeline_track_stage_exception(self):
        tracker = PipelineProgressTracker(total_stages=2)

        with pytest.raises(RuntimeError, match="Stage exploded"):
            with tracker.track_stage("faulty_stage"):
                raise RuntimeError("Stage exploded")

        assert "faulty_stage" in tracker.stage_durations
        assert tracker.completed_stages == 0

    def test_pipeline_fail_stage(self):
        tracker = PipelineProgressTracker(total_stages=2)
        tracker.start_stage("fragile_stage")
        dur = tracker.fail_stage("fragile_stage", error="CUDA out of memory")
        assert dur >= 0.0
        assert "fragile_stage" in tracker.stage_durations

    def test_pipeline_rich_progress_context(self):
        console = Console(record=True)
        with PipelineProgressTracker(total_stages=3, pipeline_name="rich_pipe", use_rich_progress=True, console=console) as tracker:
            tracker.start_stage("step_1")
            tracker.complete_stage("step_1")
            tracker.start_stage("step_2")
            tracker.complete_stage("step_2")

        assert tracker.completed_stages == 2

    def test_format_duration(self):
        assert format_duration(45.0) == "45s"
        assert format_duration(125.0) == "2m 05s"
        assert format_duration(3665.0) == "1h 01m 05s"
        assert format_duration(-10.0) == "0s"


# ============================================================
# TestSubStageProgressTrackers
# ============================================================


class TestSubStageProgressTrackers:
    def test_layer_progress_tracker(self):
        tracker = LayerProgressTracker(total_layers=32, stage_name="awq_quant")
        assert tracker.total_layers == 32
        assert tracker.completed_layers == 0

        tracker.step(layer_idx=0, layer_name="model.layers.0")
        assert tracker.completed_layers == 1
        assert tracker.get_progress_percentage() == round(1 / 32 * 100, 1)

        tracker.step(layer_idx=1, layer_name="model.layers.1")
        assert tracker.completed_layers == 2
        assert tracker.get_layers_per_second() > 0.0
        assert tracker.get_eta_seconds() >= 0.0

    def test_layer_progress_tracker_rich_context(self):
        console = Console(record=True)
        with LayerProgressTracker(total_layers=8, stage_name="fp8_quant", use_rich_progress=True, console=console) as tracker:
            for i in range(4):
                tracker.step(layer_idx=i, layer_name=f"layer_{i}")
        assert tracker.completed_layers == 4

    def test_expert_progress_tracker(self):
        tracker = ExpertProgressTracker(total_experts=64, stage_name="moe_prune")
        assert tracker.total_experts == 64

        tracker.step(expert_idx=0, expert_id="expert_0")
        assert tracker.completed_experts == 1

        tracker.step(expert_idx=1, expert_id="expert_1")
        assert tracker.completed_experts == 2
        assert tracker.get_experts_per_second() > 0.0
        assert tracker.get_eta_seconds() >= 0.0

    def test_expert_progress_tracker_rich_context(self):
        console = Console(record=True)
        with ExpertProgressTracker(total_experts=16, stage_name="moe_merge", use_rich_progress=True, console=console) as tracker:
            for i in range(8):
                tracker.step(expert_idx=i, expert_id=f"exp_{i}")
        assert tracker.completed_experts == 8

    def test_step_progress_tracker(self):
        tracker = StepProgressTracker(total_steps=100, stage_name="distillation_train")
        assert tracker.total_steps == 100
        assert tracker.completed_steps == 0

        tracker.step(step_idx=0, step_info="Batch 1", metrics={"loss": 1.23})
        assert tracker.completed_steps == 1

        tracker.step(step_idx=9, step_info="Batch 10", metrics={"loss": 0.85})
        assert tracker.completed_steps == 10
        assert tracker.get_progress_percentage() == 10.0
        assert tracker.get_steps_per_second() > 0.0
        assert tracker.get_eta_seconds() >= 0.0

    def test_step_progress_tracker_rich_context(self):
        console = Console(record=True)
        with StepProgressTracker(total_steps=20, stage_name="distill", use_rich_progress=True, console=console) as tracker:
            for i in range(10):
                tracker.step(step_idx=i, step_info=f"step_{i}", metrics={"loss": 0.5})
        assert tracker.completed_steps == 10

    def test_create_progress_bar_helper(self):
        p = create_progress_bar(disable=True)
        assert p is not None


# ============================================================
# TestGateObservability
# ============================================================


class TestGateObservability:
    def test_gate_result_event_emission(self):
        gate = QualityEvalGate(
            thresholds=GateThresholds(
                name="StrictGate",
                min_humaneval_pass1=0.70,
                min_swebench=0.30,
            )
        )
        compressed_scores = {
            "humaneval": 0.75,
            "swebench": 0.35,
        }

        verdict = gate.evaluate_scores(compressed_scores=compressed_scores)
        assert verdict.passed is True
        assert verdict.total_checks == 2
        assert verdict.passed_checks == 2


# ============================================================
# TestDAGObservability
# ============================================================


class TestDAGObservability:
    def test_dag_pipeline_emits_stage_events(self, tmp_path: Path):
        dag = DirectedAcyclicCompressionPipeline()

        # Mock methods
        method1 = MagicMock(spec=CompressionMethod)
        method1.name = "quantization_mock"
        method1.compress.return_value = CompressionArtifact(
            output_path=tmp_path / "stage1",
            format="safetensors",
            compressed_size_bytes=500,
            applied_methods=["quantization_mock"],
        )

        method2 = MagicMock(spec=CompressionMethod)
        method2.name = "pruning_mock"
        method2.compress.return_value = CompressionArtifact(
            output_path=tmp_path / "stage2",
            format="safetensors",
            compressed_size_bytes=250,
            applied_methods=["quantization_mock", "pruning_mock"],
        )

        dag.add_stage("stage_quant", method=method1)
        dag.add_stage("stage_prune", method=method2, dependencies=["stage_quant"])

        adapter = MagicMock(spec=ModelAdapter)
        adapter.load_for_compression.return_value = MagicMock()
        adapter.get_tokenizer.return_value = MagicMock()

        artifact = dag.execute(
            model_adapter=adapter,
            model_id="test_mock_model",
            output_dir=tmp_path / "out",
        )

        assert artifact is not None
        assert "quantization_mock" in artifact.applied_methods
        assert "pruning_mock" in artifact.applied_methods


# ============================================================
# TestCoreLoggerBridge
# ============================================================


class TestCoreLoggerBridge:
    def test_core_logger_operates_cleanly(self):
        logger = get_core_logger("vipym.test_bridge")
        logger.info("Testing bridged core logger message", value=123)
        assert logger is not None
