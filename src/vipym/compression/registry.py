"""Dynamic Compression Plugin Registry."""

from typing import Dict, Type
from vipym.core.exceptions import CompressionPipelineError
from vipym.interfaces.compression import CompressionMethod


class CompressionRegistry:
    """Registry for discovering and instantiating Compression Methods."""

    _registry: Dict[str, Type[CompressionMethod]] = {}

    @classmethod
    def register(cls, name: str, method_cls: Type[CompressionMethod]) -> None:
        cls._registry[name.lower()] = method_cls

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
    def list_methods(cls) -> Dict[str, Type[CompressionMethod]]:
        return dict(cls._registry)
