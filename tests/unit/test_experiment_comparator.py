"""Unit tests for ExperimentComparator and vipym compare CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vipym.analysis.comparator import ExperimentComparator
from vipym.cli.main import app

runner = CliRunner()


@pytest.fixture
def sample_experiment_dirs(tmp_path: Path) -> list[Path]:
    """Create two sample experiment output directories."""
    exp1 = tmp_path / "exp_awq"
    exp1.mkdir(parents=True)
    (exp1 / "manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp_awq",
                "model": {"id": "openai-community/gpt2"},
            }
        ),
        encoding="utf-8",
    )
    (exp1 / "evaluations").mkdir()
    (exp1 / "evaluations" / "humaneval.json").write_text(
        json.dumps({"pass_at_1": 0.82}),
        encoding="utf-8",
    )

    exp2 = tmp_path / "exp_wanda_gptq"
    exp2.mkdir(parents=True)
    (exp2 / "manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp_wanda_gptq",
                "model": {"id": "openai-community/gpt2"},
            }
        ),
        encoding="utf-8",
    )
    (exp2 / "evaluations").mkdir()
    (exp2 / "evaluations" / "humaneval.json").write_text(
        json.dumps({"pass_at_1": 0.79}),
        encoding="utf-8",
    )

    return [exp1, exp2]


class TestExperimentComparator:
    def test_comparator_loads_experiments(self, sample_experiment_dirs: list[Path], tmp_path: Path):
        """Verify comparator extracts metrics and generates summary matrix."""
        comp = ExperimentComparator(sample_experiment_dirs)
        assert len(comp.summaries) == 2
        assert comp.summaries[0].experiment_id == "exp_awq"
        assert comp.summaries[1].experiment_id == "exp_wanda_gptq"

        table = comp.format_rich_table()
        assert table is not None

        out_html = tmp_path / "diff.html"
        generated = comp.generate_html_report(out_html)
        assert generated.exists()
        content = generated.read_text(encoding="utf-8")
        assert "exp_awq" in content
        assert "exp_wanda_gptq" in content

    def test_cli_compare_command(self, sample_experiment_dirs: list[Path], tmp_path: Path):
        """Verify `vipym compare` CLI command execution."""
        out_html = tmp_path / "cli_diff.html"
        res = runner.invoke(
            app,
            [
                "compare",
                str(sample_experiment_dirs[0]),
                str(sample_experiment_dirs[1]),
                "--output",
                str(out_html),
            ],
        )
        assert res.exit_code == 0
        assert "ViPym Cross-Experiment Comparison Matrix" in res.stdout
        assert out_html.exists()

    def test_comparator_parses_results_json(self, tmp_path: Path):
        """Verify comparator loads empirical Pareto metrics directly from results.json."""
        exp_dir = tmp_path / "exp_with_results"
        exp_dir.mkdir(parents=True)
        (exp_dir / "manifest.json").write_text(
            json.dumps({"experiment_id": "exp_results", "model": {"id": "test/model"}}),
            encoding="utf-8",
        )
        results = [
            {
                "experiment_id": "exp_results",
                "configuration_name": "Baseline (FP16)",
                "quality_score": 0.80,
                "latency_p50_ms": 40.0,
                "peak_vram_gb": 16.0,
                "cost_usd": 1.20,
                "compression_ratio": 1.0,
            },
            {
                "experiment_id": "exp_results",
                "configuration_name": "Compressed (AWQ+GPTQ)",
                "quality_score": 0.78,
                "latency_p50_ms": 15.0,
                "peak_vram_gb": 4.5,
                "cost_usd": 0.35,
                "compression_ratio": 3.85,
            },
        ]
        (exp_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")

        comp = ExperimentComparator([exp_dir])
        assert len(comp.summaries) == 1
        s = comp.summaries[0]
        assert s.experiment_id == "exp_results"
        assert s.baseline_pass_at_1 == 0.80
        assert s.best_compressed_name == "Compressed (AWQ+GPTQ)"
        assert s.best_compressed_pass_at_1 == 0.78
        assert s.compression_ratio == 3.85
        assert s.latency_p50_ms == 15.0
        assert s.cost_per_1m_tokens == 0.35
        assert s.quality_retention_pct == pytest.approx(97.5, rel=1e-2)
