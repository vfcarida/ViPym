"""Dynamic Model Adapter Registry."""

from __future__ import annotations

from typing import Any

from vipym.core.exceptions import ModelAdapterError
from vipym.interfaces.model import ModelAdapter


class ModelRegistry:
    """Registry for discovering and instantiating Model Adapters."""

    _registry: dict[str, type[ModelAdapter]] = {}

    @classmethod
    def register(cls, name: str, adapter_cls: type[ModelAdapter] | None = None) -> Any:
        """Register a model adapter class directly or as a decorator."""

        def decorator(subclass: type[ModelAdapter]) -> type[ModelAdapter]:
            cls._registry[name.lower()] = subclass
            return subclass

        if adapter_cls is not None:
            return decorator(adapter_cls)
        return decorator

    @classmethod
    def get(cls, name: str) -> ModelAdapter:
        """Instantiate a registered model adapter."""
        key = name.lower()
        if key not in cls._registry:
            raise ModelAdapterError(
                f"Model adapter '{name}' not found in registry. "
                f"Available adapters: {list(cls._registry.keys())}"
            )
        return cls._registry[key]()

    @classmethod
    def list_adapters(cls) -> dict[str, type[ModelAdapter]]:
        """List all registered adapters."""
        return dict(cls._registry)
