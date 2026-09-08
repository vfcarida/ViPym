"""Dynamic Compression Plugin Registry."""

from __future__ import annotations

from typing import Any

from vipym.core.exceptions import CompressionPipelineError
from vipym.interfaces.compression import CompressionMethod


class CompressionRegistry:
    """Registry for discovering and instantiating Compression Methods."""

    _registry: dict[str, type[CompressionMethod]] = {}

    @classmethod
    def register(cls, name: str, method_cls: type[CompressionMethod] | None = None) -> Any:
        """Register a compression method class directly or as a decorator."""

        def decorator(subclass: type[CompressionMethod]) -> type[CompressionMethod]:
            cls._registry[name.lower()] = subclass
            return subclass

        if method_cls is not None:
            return decorator(method_cls)
        return decorator

    @classmethod
    def get(cls, name: str) -> CompressionMethod:
        key = name.lower()
        if key not in cls._registry:
            raise CompressionPipelineError(
                f"Compression method '{name}' not found in registry. "
                f"Available methods: {list(cls._registry.keys())}"
            )
        return cls._registry[key]()

    @classmethod
    def list_methods(cls) -> dict[str, type[CompressionMethod]]:
        return dict(cls._registry)
