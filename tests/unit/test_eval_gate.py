"""Unit tests for ViPym Evaluation Gate and CLI Quality Gating."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from vipym.cli.main import app
from vipym.interfaces.evaluation import EvaluationSuiteResult

runner = CliRunner()


def test_evaluate_basic(tmp_path: Path) -> None:
    """Verify standard evaluate command execution without gating."""
    mock_res = EvaluationSuiteResult(
        suite_name="humaneval",
        benchmark_version="1.0.0",
        pass_at_1=0.65,
        compile_rate=0.92,
        unit_test_pass_rate=0.65,
        total_tasks=10,
        passed_tasks=6,
        task_results=[],
    )

    with (
        patch("vipym.evaluation.runner.BenchmarkRunner") as mock_runner_cls,
        patch("vipym.inference.registry.InferenceRegistry.get") as mock_get,
    ):
        mock_runner_inst = MagicMock()
        mock_runner_inst.run_suite.return_value = mock_res
        mock_runner_cls.return_value = mock_runner_inst

        mock_backend = MagicMock()
        mock_get.return_value = mock_backend

        result = runner.invoke(
            app,
            [
                "evaluate",
                "--model",
                "dummy/model",
                "--suite",
                "humaneval",
                "--limit",
                "10",
            ],
        )

        assert result.exit_code == 0
        assert "Pass@1 = 65.0%" in result.stdout
        mock_backend.start.assert_called_once_with("dummy/model")
        mock_backend.stop.assert_called_once()
        mock_runner_inst.run_suite.assert_called_once_with("humaneval", mock_backend, task_limit=10)


def test_evaluate_gate_passed(tmp_path: Path) -> None:
    """Verify evaluate gate passes and produces valid JSON when metrics meet thresholds."""
    mock_res = EvaluationSuiteResult(
        suite_name="humaneval",
        benchmark_version="1.0.0",
        pass_at_1=0.75,
        compile_rate=0.95,
        unit_test_pass_rate=0.75,
        total_tasks=20,
        passed_tasks=15,
        task_results=[],
    )
    out_json = tmp_path / "gate_out.json"

    with (
        patch("vipym.evaluation.runner.BenchmarkRunner") as mock_runner_cls,
        patch("vipym.inference.registry.InferenceRegistry.get") as mock_get,
    ):
        mock_runner_inst = MagicMock()
        mock_runner_inst.run_suite.return_value = mock_res
        mock_runner_cls.return_value = mock_runner_inst

        mock_backend = MagicMock()
        mock_get.return_value = mock_backend

        result = runner.invoke(
            app,
            [
                "evaluate",
                "--model",
                "dummy/model",
                "--gate",
                "--min-pass1",
                "0.70",
                "--min-compile-rate",
                "0.90",
                "--output-json",
                str(out_json),
            ],
        )

        assert result.exit_code == 0
        assert "GATE PASSED" in result.stdout
        assert out_json.exists()

        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert data["gate_passed"] is True
        assert data["pass_at_1"] == 0.75
        assert data["compile_rate"] == 0.95
        assert len(data["violations"]) == 0


def test_evaluate_gate_failed_pass1(tmp_path: Path) -> None:
    """Verify evaluate gate exits with code 1 when Pass@1 falls below threshold."""
    mock_res = EvaluationSuiteResult(
        suite_name="humaneval",
        benchmark_version="1.0.0",
        pass_at_1=0.55,
        compile_rate=0.92,
        unit_test_pass_rate=0.55,
        total_tasks=20,
        passed_tasks=11,
        task_results=[],
    )
    out_json = tmp_path / "gate_fail.json"

    with (
        patch("vipym.evaluation.runner.BenchmarkRunner") as mock_runner_cls,
        patch("vipym.inference.registry.InferenceRegistry.get") as mock_get,
    ):
        mock_runner_inst = MagicMock()
        mock_runner_inst.run_suite.return_value = mock_res
        mock_runner_cls.return_value = mock_runner_inst

        mock_backend = MagicMock()
        mock_get.return_value = mock_backend

        result = runner.invoke(
            app,
            [
                "evaluate",
                "--model",
                "dummy/model",
                "--gate",
                "--min-pass1",
                "0.60",
                "--output-json",
                str(out_json),
            ],
        )

        assert result.exit_code != 0
        assert "GATE FAILED" in result.stdout
        assert "Pass@1" in result.stdout
        assert out_json.exists()

        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert data["gate_passed"] is False
        assert len(data["violations"]) > 0


def test_evaluate_backend_fallback() -> None:
    """Verify that evaluate command falls back to 'hf' if requested backend fails to load."""
    mock_res = EvaluationSuiteResult(
        suite_name="humaneval",
        benchmark_version="1.0.0",
        pass_at_1=0.80,
        compile_rate=1.0,
        unit_test_pass_rate=0.80,
        total_tasks=5,
        passed_tasks=4,
        task_results=[],
    )

    hf_backend = MagicMock()

    def mock_registry_get(name: str):
        if name == "vllm":
            raise RuntimeError("vLLM not installed")
        return hf_backend

    with (
        patch("vipym.evaluation.runner.BenchmarkRunner") as mock_runner_cls,
        patch(
            "vipym.inference.registry.InferenceRegistry.get",
            side_effect=mock_registry_get,
        ),
    ):
        mock_runner_inst = MagicMock()
        mock_runner_inst.run_suite.return_value = mock_res
        mock_runner_cls.return_value = mock_runner_inst

        result = runner.invoke(
            app,
            [
                "evaluate",
                "--model",
                "dummy/model",
                "--backend",
                "vllm",
            ],
        )

        assert result.exit_code == 0
        hf_backend.start.assert_called_once_with("dummy/model")
