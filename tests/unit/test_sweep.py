"""Unit tests for Multi-Experiment Grid Sweep & Pareto Exploration Engine (vipym sweep)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vipym.cli.main import app
from vipym.experiments.sweep import SweepGridConfig, SweepResult, SweepRunner


@pytest.fixture
def sweep_yaml_file(tmp_path: Path) -> Path:
    cfg_content = """
sweep_id: "test-grid-sweep"
base_model: "moonshotai/Kimi-K3"
grid:
  quantization: ["awq", "gptq"]
  bits: [4, 8]
  kv_cache: ["fp8_e4m3", "fp16"]
evaluation:
  suites: ["humaneval"]
  limit: 5
"""
    file_path = tmp_path / "sweep_config.yaml"
    file_path.write_text(cfg_content.strip(), encoding="utf-8")
    return file_path


def test_sweep_config_loading_and_expansion(sweep_yaml_file: Path):
    """Verify loading from YAML and Cartesian product expansion."""
    cfg = SweepGridConfig.from_yaml(sweep_yaml_file)
    assert cfg.sweep_id == "test-grid-sweep"
    assert cfg.model_id == "moonshotai/Kimi-K3"
    assert len(cfg.grid) == 3

    runner = SweepRunner(config=cfg, artifacts_dir=sweep_yaml_file.parent / "artifacts")
    points = runner.expand_grid()
    # 2 methods * 2 bits * 2 kv_cache = 8 total points
    assert len(points) == 8
    assert points[0]["point_id"].startswith("pt_000_")
    assert "quantization" in points[0]["parameters"]
    assert "bits" in points[0]["parameters"]
    assert "kv_cache" in points[0]["parameters"]


def test_sweep_runner_execution_and_pareto_front(sweep_yaml_file: Path):
    """Verify end-to-end execution, checkpointing, and Pareto optimization."""
    cfg = SweepGridConfig.from_yaml(sweep_yaml_file)
    artifacts_dir = sweep_yaml_file.parent / "artifacts"
    runner = SweepRunner(config=cfg, artifacts_dir=artifacts_dir)

    result = runner.run(resume=False)
    assert isinstance(result, SweepResult)
    assert result.total_points == 8
    assert result.completed_points == 8
    assert result.failed_points == 0
    assert len(result.all_points) == 8
    assert len(result.pareto_optimal_points) > 0
    assert Path(result.report_file).exists()

    # Verify Pareto optimal points are truly non-dominated
    for p in result.pareto_optimal_points:
        assert p.is_pareto_optimal is True

    # Test state checkpointing
    state_file = runner.sweep_dir / "state.json"
    assert state_file.exists()
    state_data = json.loads(state_file.read_text(encoding="utf-8"))
    assert len(state_data["executed_ids"]) == 8

    # Test resumption: running with resume=True should skip all 8 completed points immediately
    runner2 = SweepRunner(config=cfg, artifacts_dir=artifacts_dir)
    res2 = runner2.run(resume=True)
    assert res2.completed_points == 8


def test_cli_sweep_command(sweep_yaml_file: Path):
    """Verify Typer CLI 'vipym sweep' command."""
    cli_runner = CliRunner()
    artifacts_dir = sweep_yaml_file.parent / "cli_artifacts"

    res = cli_runner.invoke(
        app,
        [
            "sweep",
            "--config",
            str(sweep_yaml_file),
            "--output",
            str(artifacts_dir),
            "--no-resume",
        ],
    )
    assert res.exit_code == 0
    assert "Grid Sweep Completed" in res.stdout
    assert "Pareto Frontier Optimums" in res.stdout
    assert "Detailed Sweep Report" in res.stdout
