"""Interfaces for Compression Algorithms and DAG Pipelines."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pydantic
import torch.nn as nn

from vipym.interfaces.model import ModelAdapter, ModelMetadata, PluginCapability


class CompressionArtifact(pydantic.BaseModel):
    """Artifact resulting from executing a compression method."""

    output_path: Path
    format: str  # e.g., "compressed-tensors", "safetensors", "awq", "gptq"
    compressed_size_bytes: int
    applied_methods: list[str]
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)


class CompressionMethod(ABC):
    """Abstract interface for a discrete compression algorithm."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique method identifier."""
        pass

    @abstractmethod
    def get_capabilities(self) -> PluginCapability:
        """Return algorithm capability manifest."""
        pass

    @abstractmethod
    def validate_applicability(self, model_metadata: ModelMetadata) -> None:
        """Verify algorithm applicability against model topology."""
        pass

    @abstractmethod
    def compress(
        self,
        model: nn.Module,
        tokenizer: Any,
        calibration_data: Any | None = None,
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> CompressionArtifact:
        """Execute the compression algorithm and save output artifacts."""
        pass


class CompressionPipeline(ABC):
    """Abstract interface for multi-stage compression DAG orchestrators."""

    @abstractmethod
    def add_stage(
        self,
        stage_id: str,
        method: CompressionMethod,
        dependencies: list[str] | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> "CompressionPipeline":
        """Add a stage node to the compression DAG."""
        pass

    @abstractmethod
    def validate_dag(self, initial_metadata: ModelMetadata) -> bool:
        """Validate topological acyclicity and dtype compatibility across edges."""
        pass

    @abstractmethod
    def execute(
        self,
        model_adapter: ModelAdapter,
        model_id: str,
        output_dir: Path,
        revision: str = "main",
    ) -> CompressionArtifact:
        """Execute the DAG in topological order."""
        pass
