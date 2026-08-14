"""Dynamic Inference Backend Registry."""

from vipym.core.exceptions import InferenceRuntimeError
from vipym.interfaces.inference import InferenceBackend


class InferenceRegistry:
    """Registry for discovering and instantiating Inference Backends."""

    _registry: dict[str, type[InferenceBackend]] = {}

    @classmethod
    def register(cls, name: str, backend_cls: type[InferenceBackend]) -> None:
        cls._registry[name.lower()] = backend_cls

    @classmethod
    def get(cls, name: str) -> InferenceBackend:
        key = name.lower()
        if key not in cls._registry:
            raise InferenceRuntimeError(
                f"Inference backend '{name}' not found in registry. "
                f"Available backends: {list(cls._registry.keys())}"
            )
        return cls._registry[key]()

    @classmethod
    def list_backends(cls) -> dict[str, type[InferenceBackend]]:
        return dict(cls._registry)
