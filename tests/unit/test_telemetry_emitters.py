"""Unit tests for Enterprise Telemetry Emitters (WandB, MLflow, Composite, NoOp)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from vipym.telemetry.emitters import (
    CompositeTelemetryEmitter,
    MLflowTelemetryEmitter,
    NoOpTelemetryEmitter,
    WandbTelemetryEmitter,
    get_telemetry_emitter,
)


def test_noop_telemetry_emitter() -> None:
    """Verify NoOpTelemetryEmitter records metrics in-memory without side-effects."""
    emitter = NoOpTelemetryEmitter()
    assert emitter.is_active is False

    emitter.start_run(run_name="test-run", config={"lr": 0.001})
    assert emitter.is_active is True

    emitter.log_metrics({"loss": 0.25}, step=1)
    emitter.log_metrics({"loss": 0.15}, step=2)
    assert len(emitter.recorded_metrics) == 2

    emitter.log_artifact("checkpoint.pt", artifact_name="best_model")
    emitter.end_run(status="FINISHED")
    assert emitter.is_active is False


def test_wandb_telemetry_emitter_fallback() -> None:
    """Verify WandbTelemetryEmitter falls back safely to in-memory mode when wandb is missing."""
    with patch.dict(sys.modules, {"wandb": None}):
        emitter = WandbTelemetryEmitter(project="test-proj", offline=True)
        emitter.start_run(run_name="fallback-run", config={"bits": 4})
        assert emitter.is_active is True

        emitter.log_metrics({"pass_at_1": 0.65}, step=1)
        assert len(emitter.in_memory_metrics) == 1
        assert emitter.in_memory_metrics[0]["metrics"]["pass_at_1"] == 0.65

        emitter.log_artifact("weights.safetensors")
        emitter.end_run()
        assert emitter.is_active is False


def test_wandb_telemetry_emitter_active_mock(tmp_path: Path) -> None:
    """Verify WandbTelemetryEmitter properly delegates to wandb module when available."""
    mock_wandb = MagicMock()
    mock_run = MagicMock()
    mock_wandb.init.return_value = mock_run

    with patch.dict(sys.modules, {"wandb": mock_wandb}):
        emitter = WandbTelemetryEmitter(project="prod-compress", entity="vipym-team")
        emitter.start_run(run_name="awq-exp-01", config={"scheme": "W4A16"}, tags=["awq", "llama3"])

        mock_wandb.init.assert_called_once()
        init_kwargs = mock_wandb.init.call_args[1]
        assert init_kwargs["project"] == "prod-compress"
        assert init_kwargs["name"] == "awq-exp-01"
        assert "vipym" in init_kwargs["tags"]

        emitter.log_metrics({"latency_p50_ms": 15.2}, step=10)
        mock_wandb.log.assert_called_once_with({"latency_p50_ms": 15.2}, step=10)

        dummy_file = tmp_path / "model.safetensors"
        dummy_file.write_bytes(b"data")
        emitter.log_artifact(dummy_file, artifact_name="compressed-model")
        mock_wandb.log_artifact.assert_called_once()

        emitter.end_run(status="FINISHED")
        mock_wandb.finish.assert_called_once_with(exit_code=0)


def test_mlflow_telemetry_emitter_fallback() -> None:
    """Verify MLflowTelemetryEmitter falls back safely to in-memory mode when mlflow is missing."""
    with patch.dict(sys.modules, {"mlflow": None}):
        emitter = MLflowTelemetryEmitter(experiment_name="test-exp")
        emitter.start_run(run_name="mlflow-fallback", config={"nested": {"param": 10}})
        assert emitter.is_active is True

        emitter.log_metrics({"vram_gb": 4.5})
        assert len(emitter.in_memory_metrics) == 1
        assert emitter.in_memory_metrics[0]["metrics"]["vram_gb"] == 4.5

        emitter.end_run()
        assert emitter.is_active is False


def test_mlflow_telemetry_emitter_active_mock(tmp_path: Path) -> None:
    """Verify MLflowTelemetryEmitter properly delegates to mlflow module when available."""
    mock_mlflow = MagicMock()
    mock_run = MagicMock()
    mock_mlflow.start_run.return_value = mock_run

    with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
        emitter = MLflowTelemetryEmitter(
            tracking_uri="http://localhost:5000", experiment_name="sweeps"
        )
        emitter.start_run(
            run_name="gptq-point-0",
            config={"bits": 4, "group_size": 128, "model": {"name": "llama"}},
            tags=["gptq", "fp16"],
        )

        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
        mock_mlflow.set_experiment.assert_called_once_with("sweeps")
        mock_mlflow.log_params.assert_called_once()
        params = mock_mlflow.log_params.call_args[0][0]
        assert params["bits"] == 4
        assert params["model.name"] == "llama"

        emitter.log_metrics({"pass_at_1": 0.72}, step=5)
        mock_mlflow.log_metrics.assert_called_once_with({"pass_at_1": 0.72}, step=5)

        dummy_file = tmp_path / "report.md"
        dummy_file.write_text("# Report", encoding="utf-8")
        emitter.log_artifact(dummy_file, artifact_name="final_report")
        mock_mlflow.log_artifact.assert_called_once()

        emitter.end_run(status="FINISHED")
        mock_mlflow.end_run.assert_called_once_with(status="FINISHED")


def test_composite_telemetry_emitter() -> None:
    """Verify CompositeTelemetryEmitter broadcasts all calls across multiple child emitters."""
    mock1 = MagicMock()
    mock2 = MagicMock()

    composite = CompositeTelemetryEmitter([mock1, mock2])
    composite.start_run("test-run", {"k": "v"})
    mock1.start_run.assert_called_once_with("test-run", {"k": "v"}, None)
    mock2.start_run.assert_called_once_with("test-run", {"k": "v"}, None)

    composite.log_metrics({"metric": 1.0}, step=1)
    mock1.log_metrics.assert_called_once_with({"metric": 1.0}, 1)
    mock2.log_metrics.assert_called_once_with({"metric": 1.0}, 1)

    composite.log_artifact("file.txt", "artifact", "model")
    mock1.log_artifact.assert_called_once_with("file.txt", "artifact", "model")
    mock2.log_artifact.assert_called_once_with("file.txt", "artifact", "model")

    composite.end_run("FINISHED")
    mock1.end_run.assert_called_once_with("FINISHED")
    mock2.end_run.assert_called_once_with("FINISHED")


def test_get_telemetry_emitter_factory() -> None:
    """Verify get_telemetry_emitter returns expected classes based on provider arguments."""
    assert isinstance(get_telemetry_emitter("noop"), NoOpTelemetryEmitter)
    assert isinstance(get_telemetry_emitter("wandb"), WandbTelemetryEmitter)
    assert isinstance(get_telemetry_emitter("mlflow"), MLflowTelemetryEmitter)
    assert isinstance(get_telemetry_emitter("composite"), CompositeTelemetryEmitter)
