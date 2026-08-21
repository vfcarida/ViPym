"""Quality Regression Guard and Automated Evaluation Gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from vipym.core.logger import get_logger
from vipym.gates.config import GateThresholds
from vipym.observability.logging import emit_event

logger = get_logger(__name__)


@dataclass
class GateCheckResult:
    """Detailed evaluation result for an individual threshold check."""

    name: str
    metric_name: str
    operator: str
    required_threshold: float
    actual_value: float
    teacher_baseline_value: float | None = None
    relative_retention: float | None = None
    passed: bool = True
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GateVerdict:
    """Overall evaluation gate decision with detailed check results and Markdown report."""

    gate_name: str
    passed: bool
    exit_code: int  # 0 = PASS, 1 = FAIL, 2 = ERROR
    total_checks: int
    passed_checks: int
    failed_checks: list[GateCheckResult]
    checks: list[GateCheckResult]
    markdown_report: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "passed": self.passed,
            "exit_code": self.exit_code,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": [c.to_dict() for c in self.failed_checks],
            "checks": [c.to_dict() for c in self.checks],
            "markdown_report": self.markdown_report,
        }


class QualityEvalGate:
    """Automated gate that checks post-compression quality against relative teacher baselines."""

    def __init__(self, thresholds: GateThresholds | None = None) -> None:
        self.thresholds = thresholds or GateThresholds()

    def evaluate_scores(
        self,
        compressed_scores: dict[str, float],
        teacher_scores: dict[str, float] | None = None,
        telemetry: dict[str, Any] | None = None,
    ) -> GateVerdict:
        """Run all configured quality and latency checks against the compressed model results."""
        checks: list[GateCheckResult] = []
        th = self.thresholds
        use_rel = th.use_relative_scoring and (teacher_scores is not None and len(teacher_scores) > 0)

        # Helper mapping of metrics to threshold attributes
        suite_mappings: list[tuple[str, str, float]] = [
            ("se_composite", "SE Composite Score", th.min_se_composite),
            ("humaneval", "HumanEval pass@1", th.min_humaneval_pass1),
            ("human_eval", "HumanEval pass@1", th.min_humaneval_pass1),
            ("aider_edit", "Aider Edit Benchmark", th.min_aider_edit),
            ("aider", "Aider Edit Benchmark", th.min_aider_edit),
            ("bigcodebench", "BigCodeBench", th.min_bigcodebench),
            ("bcb", "BigCodeBench", th.min_bigcodebench),
            ("swebench", "SWE-bench Verified", th.min_swebench),
            ("swe_bench", "SWE-bench Verified", th.min_swebench),
            ("testgeneval", "TestGenEval Line/Branch Coverage", th.min_testgeneval_coverage),
            ("crqbench", "CRQBench Code Review Precision", th.min_crqbench_precision),
        ]

        evaluated_suites: set[str] = set()

        for key, display_name, min_thresh in suite_mappings:
            if key in compressed_scores and display_name not in evaluated_suites:
                actual = float(compressed_scores[key])
                teacher_val = float(teacher_scores.get(key, 0.0)) if teacher_scores else None

                if use_rel and teacher_val is not None and teacher_val > 0:
                    retention = actual / teacher_val
                    passed = retention >= min_thresh
                    err = None if passed else f"Relative retention {retention*100:.1f}% is below required {min_thresh*100:.1f}%"
                else:
                    retention = None
                    passed = actual >= min_thresh
                    err = None if passed else f"Score {actual:.3f} is below required threshold {min_thresh:.3f}"

                checks.append(
                    GateCheckResult(
                        name=f"Quality: {display_name}",
                        metric_name=key,
                        operator=">=",
                        required_threshold=min_thresh,
                        actual_value=actual,
                        teacher_baseline_value=teacher_val,
                        relative_retention=retention,
                        passed=passed,
                        error_message=err,
                    )
                )
                evaluated_suites.add(display_name)

        # 2. Check maximum allowable quality drop on ANY single benchmark suite
        if use_rel and teacher_scores:
            min_allowable_retention = 1.0 - th.max_quality_drop_any_suite  # e.g., 1.0 - 0.50 = 0.50
            for suite_key, teacher_val in teacher_scores.items():
                if suite_key in compressed_scores and teacher_val > 0:
                    actual = float(compressed_scores[suite_key])
                    retention = actual / teacher_val
                    drop = 1.0 - retention
                    passed = retention >= min_allowable_retention
                    err = (
                        None
                        if passed
                        else f"Suite '{suite_key}' suffered {drop*100:.1f}% drop (exceeds max allowable {th.max_quality_drop_any_suite*100:.0f}%)"
                    )

                    checks.append(
                        GateCheckResult(
                            name=f"Max Drop: {suite_key}",
                            metric_name=f"{suite_key}_drop",
                            operator="<=",
                            required_threshold=th.max_quality_drop_any_suite,
                            actual_value=drop,
                            teacher_baseline_value=teacher_val,
                            relative_retention=retention,
                            passed=passed,
                            error_message=err,
                        )
                    )

        # 3. Check custom suite thresholds
        for suite_k, target_val in th.suite_thresholds.items():
            if suite_k in compressed_scores:
                actual = float(compressed_scores[suite_k])
                passed = actual >= target_val
                checks.append(
                    GateCheckResult(
                        name=f"Custom: {suite_k}",
                        metric_name=suite_k,
                        operator=">=",
                        required_threshold=target_val,
                        actual_value=actual,
                        passed=passed,
                        error_message=None if passed else f"Custom metric {actual:.3f} < {target_val:.3f}",
                    )
                )

        # 4. Check Latency P95
        if telemetry and "latency_p95_ms" in telemetry:
            p95_ms = float(telemetry["latency_p95_ms"])
            passed = p95_ms <= th.max_latency_p95_ms
            checks.append(
                GateCheckResult(
                    name="Performance: Latency p95",
                    metric_name="latency_p95_ms",
                    operator="<=",
                    required_threshold=th.max_latency_p95_ms,
                    actual_value=p95_ms,
                    passed=passed,
                    error_message=None if passed else f"p95 latency {p95_ms:.1f}ms exceeds max {th.max_latency_p95_ms:.1f}ms",
                )
            )

        failed = [c for c in checks if not c.passed]
        overall_passed = len(failed) == 0 and len(checks) > 0
        exit_code = 0 if overall_passed else (1 if checks else 2)

        verdict = GateVerdict(
            gate_name=th.name,
            passed=overall_passed,
            exit_code=exit_code,
            total_checks=len(checks),
            passed_checks=len(checks) - len(failed),
            failed_checks=failed,
            checks=checks,
            markdown_report="",
        )
        verdict.markdown_report = self.generate_markdown_report(verdict)
        emit_event(
            "gate_result",
            gate_name=verdict.gate_name,
            passed=verdict.passed,
            exit_code=verdict.exit_code,
            total_checks=verdict.total_checks,
            passed_checks=verdict.passed_checks,
            failed_count=len(verdict.failed_checks),
        )
        return verdict

    def generate_markdown_report(self, verdict: GateVerdict) -> str:
        """Generate a formatted GitHub-style Markdown report table showing scores vs thresholds."""
        status_icon = "🟢 **PASS**" if verdict.passed else "🔴 **FAIL**"
        lines = [
            f"# ViPym Quality Evaluation Gate Report: `{verdict.gate_name}`",
            "",
            f"**Overall Verdict**: {status_icon} ({verdict.passed_checks}/{verdict.total_checks} checks passed)",
            "",
            "| Check | Metric | Required | Actual | Teacher Baseline | Relative Retention | Status |",
            "|---|---|---|---|---|---|:---:|",
        ]

        for c in verdict.checks:
            icon = "✅ PASS" if c.passed else "❌ FAIL"
            teacher_str = f"{c.teacher_baseline_value:.3f}" if c.teacher_baseline_value is not None else "N/A"
            retention_str = f"{c.relative_retention*100:.1f}%" if c.relative_retention is not None else "N/A"
            req_str = f"{c.operator} {c.required_threshold*100:.0f}%" if "Drop" in c.name or c.relative_retention is not None else f"{c.operator} {c.required_threshold}"
            actual_str = f"{c.actual_value:.3f}"

            lines.append(
                f"| {c.name} | `{c.metric_name}` | {req_str} | {actual_str} | {teacher_str} | {retention_str} | {icon} |"
            )

        if verdict.failed_checks:
            lines.extend([
                "",
                "### ⚠️ Failed Check Details",
                "",
            ])
            for fc in verdict.failed_checks:
                lines.append(f"- **{fc.name}**: {fc.error_message}")

        return "\n".join(lines)
