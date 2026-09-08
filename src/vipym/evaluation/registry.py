"""Dynamic Evaluation Suite Registry."""

from __future__ import annotations

from typing import Any

from vipym.core.exceptions import BenchmarkEvaluationError
from vipym.interfaces.evaluation import EvaluationSuite


class EvaluationRegistry:
    """Registry for discovering and instantiating benchmark evaluation suites."""

    _registry: dict[str, type[EvaluationSuite]] = {}

    @classmethod
    def register(cls, name: str, suite_cls: type[EvaluationSuite] | None = None) -> Any:
        """Register an evaluation suite directly or as a decorator."""

        def decorator(subclass: type[EvaluationSuite]) -> type[EvaluationSuite]:
            cls._registry[name.lower()] = subclass
            return subclass

        if suite_cls is not None:
            return decorator(suite_cls)
        return decorator

    @classmethod
    def get(cls, name: str) -> EvaluationSuite:
        key = name.lower()
        if key not in cls._registry:
            raise BenchmarkEvaluationError(
                f"Evaluation suite '{name}' not found in registry. "
                f"Available suites: {list(cls._registry.keys())}"
            )
        return cls._registry[key]()

    @classmethod
    def list_suites(cls) -> dict[str, type[EvaluationSuite]]:
        return dict(cls._registry)
