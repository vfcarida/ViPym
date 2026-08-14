"""Evaluation package with auto-registration."""

import vipym.evaluation.suites
from vipym.evaluation.contamination import ContaminationAuditor, ContaminationReport
from vipym.evaluation.registry import EvaluationRegistry
from vipym.evaluation.runner import BenchmarkRunner

__all__ = [
    "BenchmarkRunner",
    "ContaminationAuditor",
    "ContaminationReport",
    "EvaluationRegistry",
]
