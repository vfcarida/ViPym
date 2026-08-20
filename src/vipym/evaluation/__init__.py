"""Evaluation package with auto-registration."""

from vipym.evaluation import agents as agents
from vipym.evaluation import suites as suites
from vipym.evaluation.composite import (
    SECompositeCalculator,
    SECompositeReport,
    compute_se_composite_score,
)
from vipym.evaluation.contamination import ContaminationAuditor, ContaminationReport
from vipym.evaluation.registry import EvaluationRegistry
from vipym.evaluation.runner import BenchmarkRunner
from vipym.evaluation.scoring import (
    calculate_pass_at_k_metrics,
    compute_pass_at_k,
)

__all__ = [
    "BenchmarkRunner",
    "ContaminationAuditor",
    "ContaminationReport",
    "EvaluationRegistry",
    "SECompositeCalculator",
    "SECompositeReport",
    "agents",
    "calculate_pass_at_k_metrics",
    "compute_pass_at_k",
    "compute_se_composite_score",
    "suites",
]
