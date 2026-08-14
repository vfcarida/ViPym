"""Dynamic Evaluation Suite Registry."""

from typing import Dict, Type
from vipym.core.exceptions import BenchmarkEvaluationError
from vipym.interfaces.evaluation import EvaluationSuite


class EvaluationRegistry:
    """Registry for discovering and instantiating benchmark evaluation suites."""

    _registry: Dict[str, Type[EvaluationSuite]] = {}

    @classmethod
    def register(cls, name: str, suite_cls: Type[EvaluationSuite]) -> None:
        cls._registry[name.lower()] = suite_cls

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
    def list_suites(cls) -> Dict[str, Type[EvaluationSuite]]:
        return dict(cls._registry)
