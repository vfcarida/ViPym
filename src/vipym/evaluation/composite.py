"""Software Engineering (SE) Composite Score Calculator.

Computes the weighted composite SE score across all 5 core evaluation dimensions:
  SE_Score = 0.30 * swebench + 0.25 * aider_edit + 0.20 * bigcodebench + 0.15 * testgen + 0.10 * code_review

Validates quality thresholds for 'Production Ready' compressed models:
- SE_Composite >= 0.65 (or >= 65% of uncompressed teacher baseline)
- No single suite score drops below 50% of baseline
- TestGenEval coverage >= 60%
- Code Review precision >= 40%
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vipym.core.logger import get_logger
from vipym.interfaces.evaluation import EvaluationSuiteResult

logger = get_logger(__name__)

# Canonical suite weights
DEFAULT_SE_WEIGHTS: dict[str, float] = {
    "swebench": 0.30,
    "aider_edit": 0.25,
    "bigcodebench": 0.20,
    "testgen": 0.15,
    "code_review": 0.10,
}

# Alias mapping
SUITE_ALIASES: dict[str, str] = {
    "swebench_lite": "swebench",
    "swebench_verified": "swebench",
    "swebench_full": "swebench",
    "aider_bench": "aider_edit",
    "aider": "aider_edit",
    "aider_polyglot": "aider_edit",
    "bigcodebench_full": "bigcodebench",
    "bigcodebench_hard": "bigcodebench",
    "bigcodebench_lite": "bigcodebench",
    "testgeneval": "testgen",
    "crqbench": "code_review",
}

CATEGORY_MAP: dict[str, str] = {
    "swebench": "Repository Bug Fixing",
    "aider_edit": "Code Editing & Refactoring",
    "bigcodebench": "Practical Library Coding",
    "testgen": "Unit Test Generation",
    "code_review": "Code Review & Defect Detection",
}


@dataclass
class SECompositeReport:
    """Comprehensive evaluation report summarizing composite SE score and production readiness."""

    composite_score: float
    relative_score: float | None = None
    category_scores: dict[str, float] = field(default_factory=dict)
    suite_scores: dict[str, float] = field(default_factory=dict)
    is_production_ready: bool = False
    readiness_reasons: list[str] = field(default_factory=list)
    thresholds_passed: dict[str, bool] = field(default_factory=dict)


class SECompositeCalculator:
    """Calculates weighted SE composite quality score and validates production-readiness gates."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        min_composite_threshold: float = 0.65,
        min_individual_ratio: float = 0.50,
        min_testgen_coverage: float = 0.60,
        min_review_precision: float = 0.40,
    ) -> None:
        self.weights = weights or dict(DEFAULT_SE_WEIGHTS)
        self.min_composite_threshold = min_composite_threshold
        self.min_individual_ratio = min_individual_ratio
        self.min_testgen_coverage = min_testgen_coverage
        self.min_review_precision = min_review_precision

    def compute(
        self,
        suite_results: dict[str, float | EvaluationSuiteResult | dict[str, Any]],
        baseline_results: dict[str, float] | None = None,
    ) -> SECompositeReport:
        """Compute the weighted composite score and evaluate against production readiness criteria."""
        normalized_scores: dict[str, float] = {}
        metadata_map: dict[str, dict[str, Any]] = {}

        # 1. Normalize suite names and extract primary metric score
        for raw_name, result in suite_results.items():
            canonical_name = SUITE_ALIASES.get(raw_name.lower(), raw_name.lower())
            score = 0.0

            if isinstance(result, (int, float)):
                score = float(result)
            elif isinstance(result, EvaluationSuiteResult):
                score = result.pass_at_1
                metadata_map[canonical_name] = result.summary_metrics
            elif isinstance(result, dict):
                score = float(result.get("pass_at_1", result.get("score", 0.0)))
                metadata_map[canonical_name] = result

            normalized_scores[canonical_name] = score

        # 2. Re-weight available suites
        available_weights: dict[str, float] = {}
        for suite, w in self.weights.items():
            if suite in normalized_scores:
                available_weights[suite] = w

        total_weight = sum(available_weights.values())
        if total_weight > 0:
            reweighted = {k: v / total_weight for k, v in available_weights.items()}
        else:
            reweighted = {}

        # 3. Compute weighted composite score
        composite_score = sum(normalized_scores[k] * reweighted[k] for k in reweighted)

        # 4. Compute category breakdown
        category_scores: dict[str, float] = {}
        for canonical, score in normalized_scores.items():
            cat_name = CATEGORY_MAP.get(canonical, canonical)
            category_scores[cat_name] = score

        # 5. Production Readiness Evaluation
        readiness_reasons: list[str] = []
        thresholds_passed: dict[str, bool] = {}

        # Gate 1: Composite score threshold
        relative_score: float | None = None
        if baseline_results:
            base_calc = self.__class__(weights=self.weights)
            base_report = base_calc.compute(baseline_results)
            if base_report.composite_score > 0:
                relative_score = composite_score / base_report.composite_score

        eval_score = relative_score if relative_score is not None else composite_score
        gate1_passed = eval_score >= self.min_composite_threshold
        thresholds_passed["composite_threshold"] = gate1_passed
        if not gate1_passed:
            readiness_reasons.append(
                f"SE Composite score {eval_score:.1%} is below minimum threshold {self.min_composite_threshold:.1%}"
            )

        # Gate 2: No single suite drops below 50% of baseline (or absolute 0.20)
        gate2_passed = True
        if baseline_results:
            for suite, score in normalized_scores.items():
                base_score = baseline_results.get(suite, 0.0)
                if isinstance(base_score, (int, float)) and base_score > 0:
                    ratio = score / base_score
                    if ratio < self.min_individual_ratio:
                        gate2_passed = False
                        readiness_reasons.append(
                            f"Suite '{suite}' dropped to {ratio:.1%} of baseline (< {self.min_individual_ratio:.1%})"
                        )
        else:
            for suite, score in normalized_scores.items():
                if score < 0.20:
                    gate2_passed = False
                    readiness_reasons.append(
                        f"Suite '{suite}' score {score:.1%} is below minimum floor (20.0%)"
                    )
        thresholds_passed["no_suite_drop"] = gate2_passed

        # Gate 3: TestGen coverage threshold
        testgen_meta = metadata_map.get("testgen", {})
        testgen_cov = testgen_meta.get("line_coverage", normalized_scores.get("testgen", 0.0))
        if "testgen" in normalized_scores:
            gate3_passed = testgen_cov >= self.min_testgen_coverage
            thresholds_passed["testgen_coverage"] = gate3_passed
            if not gate3_passed:
                readiness_reasons.append(
                    f"TestGenEval coverage {testgen_cov:.1%} is below required {self.min_testgen_coverage:.1%}"
                )
        else:
            thresholds_passed["testgen_coverage"] = True

        # Gate 4: Code review precision threshold
        review_meta = metadata_map.get("code_review", {})
        review_prec = review_meta.get("precision", normalized_scores.get("code_review", 0.0))
        if "code_review" in normalized_scores:
            gate4_passed = review_prec >= self.min_review_precision
            thresholds_passed["review_precision"] = gate4_passed
            if not gate4_passed:
                readiness_reasons.append(
                    f"Code Review precision {review_prec:.1%} is below required {self.min_review_precision:.1%}"
                )
        else:
            thresholds_passed["review_precision"] = True

        is_production_ready = all(thresholds_passed.values())
        if is_production_ready:
            readiness_reasons.append(
                "Model passed all SE benchmark quality thresholds for production readiness."
            )

        return SECompositeReport(
            composite_score=composite_score,
            relative_score=relative_score,
            category_scores=category_scores,
            suite_scores=normalized_scores,
            is_production_ready=is_production_ready,
            readiness_reasons=readiness_reasons,
            thresholds_passed=thresholds_passed,
        )


def compute_se_composite_score(
    suite_results: dict[str, float | EvaluationSuiteResult | dict[str, Any]],
    baseline_results: dict[str, float] | None = None,
) -> SECompositeReport:
    """Convenience helper to compute composite SE score."""
    calculator = SECompositeCalculator()
    return calculator.compute(suite_results, baseline_results=baseline_results)
