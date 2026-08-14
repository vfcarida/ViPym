"""Dynamic Model Adapter Registry."""

from typing import Dict, Type
from vipym.core.exceptions import ModelAdapterError
from vipym.interfaces.model import ModelAdapter


class ModelRegistry:
    """Registry for discovering and instantiating Model Adapters."""

    _registry: Dict[str, Type[ModelAdapter]] = {}

    @classmethod
    def register(cls, name: str, adapter_cls: Type[ModelAdapter]) -> None:
        """Register a model adapter class."""
        cls._registry[name.lower()] = adapter_cls

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
    def list_adapters(cls) -> Dict[str, Type[ModelAdapter]]:
        """List all registered adapters."""
        return dict(cls._registry)
