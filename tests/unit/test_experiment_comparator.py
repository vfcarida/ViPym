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
