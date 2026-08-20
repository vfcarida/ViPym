"""Unit tests for P016 — Quality Regression Guard (Eval Gate), Relative Retention, and CLI Decisions.

Test classes:
  TestGateConfig          — YAML schema parsing, defaults, custom threshold mappings
  TestQualityEvalGate     — Relative scoring vs teacher baseline, max drop guard, latency thresholds, pass/fail
  TestGateMarkdownReport  — GitHub-style Markdown table formatting and failure messages
  TestGateCLI             — CLI invocation, exit codes (0 on PASS, 1 on FAIL, 2 on ERROR)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from vipym.cli.main import app
from vipym.gates.config import GateThresholds, GatesConfig
from vipym.gates.eval_gate import GateCheckResult, GateVerdict, QualityEvalGate


# ============================================================
# TestGateConfig
# ============================================================


class TestGateConfig:
    def test_default_gate_thresholds(self):
        th = GateThresholds()
        assert th.name == "se_production"
        assert th.min_se_composite == 0.65
        assert th.min_humaneval_pass1 == 0.70
        assert th.min_aider_edit == 0.70
        assert th.min_bigcodebench == 0.50
        assert th.min_swebench == 0.35
        assert th.max_latency_p95_ms == 5000.0
        assert th.max_quality_drop_any_suite == 0.50

    def test_from_yaml_and_dict(self, tmp_path: Path):
        yaml_content = """
gates:
  se_strict:
    min_se_composite: 0.85
    min_humaneval_pass1: 0.80
    min_aider_edit: 0.80
    min_bigcodebench: 0.60
    max_latency_p95_ms: 3000.0
    max_quality_drop_any_suite: 0.25
"""
        cfg_file = tmp_path / "gates.yaml"
        cfg_file.write_text(yaml_content, encoding="utf-8")

        gates_cfg = GatesConfig.from_yaml(cfg_file)
        gate = gates_cfg.get_gate("se_strict")

        assert gate.name == "se_strict"
        assert gate.min_se_composite == 0.85
        assert gate.min_humaneval_pass1 == 0.80
        assert gate.max_latency_p95_ms == 3000.0
        assert gate.max_quality_drop_any_suite == 0.25


# ============================================================
# TestQualityEvalGate
# ============================================================


class TestQualityEvalGate:
    @pytest.fixture
    def teacher_scores(self) -> dict[str, float]:
        return {
            "se_composite": 0.90,
            "humaneval": 0.85,
            "aider_edit": 0.80,
            "bigcodebench": 0.65,
            "swebench": 0.45,
            "testgeneval": 0.78,
            "crqbench": 0.60,
        }

    def test_gate_passes_when_all_above_threshold(self, teacher_scores):
        gate = QualityEvalGate(GateThresholds(min_se_composite=0.85))

        # Compressed model retains ~95% across all suites
        compressed_scores = {
            "se_composite": 0.86,  # 0.86 / 0.90 = 95.6% >= 85%
            "humaneval": 0.82,  # 0.82 / 0.85 = 96.5% >= 70%
            "aider_edit": 0.78,
            "bigcodebench": 0.62,
            "swebench": 0.42,
            "testgeneval": 0.74,
            "crqbench": 0.58,
        }
        telemetry = {"latency_p95_ms": 1500.0}

        verdict = gate.evaluate_scores(
            compressed_scores=compressed_scores,
            teacher_scores=teacher_scores,
            telemetry=telemetry,
        )

        assert isinstance(verdict, GateVerdict)
        assert verdict.passed is True
        assert verdict.exit_code == 0
        assert len(verdict.failed_checks) == 0
        assert "PASS" in verdict.markdown_report

    def test_gate_fails_when_se_composite_below_threshold(self, teacher_scores):
        gate = QualityEvalGate(GateThresholds(min_se_composite=0.85))

        # Compressed model composite drops to 0.50 (55.6% retention < 85%)
        compressed_scores = {
            "se_composite": 0.50,
            "humaneval": 0.80,
            "aider_edit": 0.75,
            "bigcodebench": 0.60,
            "swebench": 0.40,
        }

        verdict = gate.evaluate_scores(
            compressed_scores=compressed_scores,
            teacher_scores=teacher_scores,
        )

        assert verdict.passed is False
        assert verdict.exit_code == 1
        assert len(verdict.failed_checks) >= 1
        assert any(c.metric_name == "se_composite" for c in verdict.failed_checks)

    def test_gate_fails_when_single_suite_drops_too_much(self, teacher_scores):
        # Even if composite is acceptable, no individual suite should drop >50%
        gate = QualityEvalGate(GateThresholds(max_quality_drop_any_suite=0.50))

        compressed_scores = {
            "se_composite": 0.80,
            "humaneval": 0.80,
            "aider_edit": 0.75,
            "bigcodebench": 0.60,
            "swebench": 0.10,  # Dropped from 0.45 to 0.10 (77.8% drop > 50%)
        }

        verdict = gate.evaluate_scores(
            compressed_scores=compressed_scores,
            teacher_scores=teacher_scores,
        )

        assert verdict.passed is False
        assert verdict.exit_code == 1
        assert any("swebench_drop" in c.metric_name for c in verdict.failed_checks)

    def test_gate_fails_when_p95_latency_exceeds_threshold(self, teacher_scores):
        gate = QualityEvalGate(GateThresholds(max_latency_p95_ms=2000.0))

        compressed_scores = {
            "se_composite": 0.88,
            "humaneval": 0.82,
        }
        telemetry = {"latency_p95_ms": 3500.0}  # Exceeds 2000ms

        verdict = gate.evaluate_scores(
            compressed_scores=compressed_scores,
            teacher_scores=teacher_scores,
            telemetry=telemetry,
        )

        assert verdict.passed is False
        assert any(c.metric_name == "latency_p95_ms" for c in verdict.failed_checks)

    def test_gate_without_teacher_uses_absolute_scores(self):
        gate = QualityEvalGate(
            GateThresholds(
                min_humaneval_pass1=0.70,
                use_relative_scoring=False,
            )
        )

        compressed_scores = {"humaneval": 0.75}
        verdict = gate.evaluate_scores(compressed_scores=compressed_scores)
        assert verdict.passed is True

        compressed_low = {"humaneval": 0.60}
        verdict_low = gate.evaluate_scores(compressed_scores=compressed_low)
        assert verdict_low.passed is False


# ============================================================
# TestGateMarkdownReport
# ============================================================


class TestGateMarkdownReport:
    def test_markdown_table_formatting(self):
        gate = QualityEvalGate()
        compressed = {"se_composite": 0.85, "humaneval": 0.80}
        teacher = {"se_composite": 0.90, "humaneval": 0.85}

        verdict = gate.evaluate_scores(compressed, teacher)
        md = verdict.markdown_report

        assert "# ViPym Quality Evaluation Gate Report" in md
        assert "| Check | Metric | Required | Actual | Teacher Baseline |" in md
        assert "PASS" in md


# ============================================================
# TestGateCLI
# ============================================================


class TestGateCLI:
    def test_cli_gate_run_pass(self, tmp_path: Path):
        cfg_file = tmp_path / "gates.yaml"
        cfg_file.write_text("gates:\n  se_production:\n    min_se_composite: 0.65\n", encoding="utf-8")

        runner = CliRunner()
        res = runner.invoke(
            app,
            [
                "gate",
                "run",
                "--config",
                str(cfg_file),
                "--model",
                "models/Kimi-K3-AWQ-4bit",
                "--teacher",
                "models/Kimi-K3-Base",
            ],
        )

        assert res.exit_code == 0
        assert "Running Quality Gate" in res.output
        assert "PASS" in res.output

    def test_cli_gate_run_fail(self, tmp_path: Path):
        cfg_file = tmp_path / "strict_gates.yaml"
        # Require impossible 99.9% composite retention
        cfg_file.write_text("gates:\n  se_production:\n    min_se_composite: 0.999\n", encoding="utf-8")

        runner = CliRunner()
        res = runner.invoke(
            app,
            [
                "gate",
                "run",
                "--config",
                str(cfg_file),
                "--model",
                "models/Kimi-K3-AWQ-4bit",
                "--teacher",
                "models/Kimi-K3-Base",
            ],
        )

        assert res.exit_code == 1
        assert "FAIL" in res.output
