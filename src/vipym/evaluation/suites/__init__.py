"""Benchmark suites."""

from vipym.evaluation.suites.humaneval import HumanEvalSuite
from vipym.evaluation.suites.mbpp import LiveCodeBenchSuite, MBPPSuite, SWEBenchSuite

__all__ = [
    "HumanEvalSuite",
    "LiveCodeBenchSuite",
    "MBPPSuite",
    "SWEBenchSuite",
]
