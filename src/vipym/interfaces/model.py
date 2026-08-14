"""Interfaces for Model Adapters and architecture metadata."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import pydantic
import torch
import torch.nn as nn

from vipym.core.constants import ComputeArchitecture, SupportedDtype


@dataclass(frozen=True)
class PluginCapability:
    """Explicit capability declaration required by all plugins."""
    supported_architectures: Set[ComputeArchitecture]
    supported_dtypes: Set[SupportedDtype]
    supports_moe: bool = False
    requires_training: bool = False
    requires_calibration: bool = False
    supported_runtimes: Set[str] = field(default_factory=lambda: {"vllm", "sglang", "hf"})
    min_gpu_memory_gb: float = 0.0
    recommended_gpu_count: int = 1


class ModelMetadata(pydantic.BaseModel):
    """Inspected metadata of a foundational LLM."""
    model_id: str
    revision: str
    total_parameters: int
    active_parameters: int
    architecture_type: ComputeArchitecture
    native_dtypes: List[SupportedDtype]
    context_window: int
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: Optional[int] = None
    num_experts: Optional[int] = None
    num_selected_experts: Optional[int] = None
    has_custom_kernels: bool = False
    raw_config: Dict[str, Any] = field(default_factory=dict)


class ModelAdapter(ABC):
    """Abstract interface for wrapping and introspecting LLM architectures."""

    @abstractmethod
    def get_capabilities(self) -> PluginCapability:
        """Return supported capabilities."""
        pass

    @abstractmethod
    def inspect_metadata(self, model_id_or_path: str, revision: str = "main") -> ModelMetadata:
        """Inspect model metadata without full weight loading."""
        pass

    @abstractmethod
    def load_for_compression(
        self,
        model_id_or_path: str,
        revision: str = "main",
        device_map: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs: Any,
    ) -> nn.Module:
        """Load PyTorch model in memory for offline compression."""
        pass

    @abstractmethod
    def get_tokenizer(self, model_id_or_path: str, revision: str = "main") -> Any:
        """Obtain immutable tokenizer."""
        pass
