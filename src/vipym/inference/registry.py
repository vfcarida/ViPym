"""Dynamic Inference Backend Registry."""

from __future__ import annotations

from typing import Any

from vipym.core.exceptions import InferenceRuntimeError
from vipym.interfaces.inference import InferenceBackend


class InferenceRegistry:
    """Registry for discovering and instantiating Inference Backends."""

    _registry: dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, backend_cls: Any = None) -> Any:
        """Register an inference backend directly or as a decorator."""

        def decorator(subclass: Any) -> Any:
            cls._registry[name.lower()] = subclass
            return subclass

        if backend_cls is not None:
            return decorator(backend_cls)
        return decorator

    @classmethod
    def get(cls, name: str) -> InferenceBackend:
        key = name.lower()
        if key not in cls._registry:
            raise InferenceRuntimeError(
                f"Inference backend '{name}' not found in registry. "
                f"Available backends: {list(cls._registry.keys())}"
            )
        entry = cls._registry[key]
        if isinstance(entry, InferenceBackend):
            return entry
        if callable(entry):
            return entry()
        raise InferenceRuntimeError(f"Registry entry for '{name}' is not callable or an instance.")

    @classmethod
    def get_class(cls, name: str) -> type[InferenceBackend]:
        key = name.lower()
        if key not in cls._registry:
            raise InferenceRuntimeError(
                f"Inference backend '{name}' not found in registry. "
                f"Available backends: {list(cls._registry.keys())}"
            )
        entry = cls._registry[key]
        if isinstance(entry, type):
            return entry
        return type(entry)

    @classmethod
    def list_backends(cls) -> dict[str, Any]:
        return dict(cls._registry)
