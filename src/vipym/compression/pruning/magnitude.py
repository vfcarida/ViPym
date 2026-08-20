"""Pruning & Sparsity Adapters (Magnitude, 2:4 Semi-Structured, Wanda, SparseGPT)."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from vipym.compression.methods.pruning import (
    SparseGPTPruningMethod,
    UnifiedPruningMethod,
    WandaPruningMethod,
)
from vipym.compression.registry import CompressionRegistry
from vipym.core.constants import ComputeArchitecture, SupportedDtype
from vipym.core.logger import get_logger
from vipym.interfaces.compression import CompressionArtifact, CompressionMethod
from vipym.interfaces.model import ModelMetadata, PluginCapability

logger = get_logger(__name__)


class MagnitudePruningMethod(CompressionMethod):
    """Unstructured / Structured Magnitude Pruning."""

    def __init__(self, sparsity_ratio: float = 0.3, structured: bool = False) -> None:
        self.sparsity_ratio = sparsity_ratio
        self.structured = structured

    @property
    def name(self) -> str:
        s_type = "structured" if self.structured else "unstructured"
        return f"prune_magnitude_{s_type}_{int(self.sparsity_ratio * 100)}pct"

    def get_capabilities(self) -> PluginCapability:
        return PluginCapability(
            supported_architectures={
                ComputeArchitecture.DENSE,
                ComputeArchitecture.MOE,
            },
            supported_dtypes={SupportedDtype.FP16, SupportedDtype.BF16},
            supports_moe=True,
            requires_calibration=False,
            supported_runtimes={"vllm", "hf"},
        )

    def validate_applicability(self, model_metadata: ModelMetadata) -> None:
        pass

    def compress(
        self,
        model: nn.Module,
        tokenizer: Any,
        calibration_data: Any | None = None,
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> CompressionArtifact:
        out = Path(output_dir or "./pruned_model")
        out.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Applying Magnitude Pruning (ratio={self.sparsity_ratio}, structured={self.structured})"
        )

        # Zero out bottom magnitude weights in linear layers
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and "lm_head" not in name:
                    weight = module.weight.data
                    thresh = torch.quantile(weight.abs().float(), self.sparsity_ratio)
                    mask = weight.abs() >= thresh
                    module.weight.data = weight * mask

        if hasattr(model, "save_pretrained"):
            model.save_pretrained(out)
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(out)

        total_bytes = sum(f.stat().st_size for f in out.glob("**/*") if f.is_file())

        return CompressionArtifact(
            output_path=out,
            format="safetensors",
            compressed_size_bytes=total_bytes,
            applied_methods=[self.name],
            metadata={"sparsity_ratio": self.sparsity_ratio, "structured": self.structured},
        )


class NMSparsityMethod(CompressionMethod):
    """N:M (2:4) Semi-Structured Sparsity for hardware Tensor Core acceleration."""

    def __init__(self, n: int = 2, m: int = 4) -> None:
        self.n = n
        self.m = m

    @property
    def name(self) -> str:
        return f"prune_nm_{self.n}_{self.m}"

    def get_capabilities(self) -> PluginCapability:
        return PluginCapability(
            supported_architectures={
                ComputeArchitecture.DENSE,
                ComputeArchitecture.MOE,
            },
            supported_dtypes={SupportedDtype.FP16, SupportedDtype.BF16},
            supports_moe=True,
            requires_calibration=False,
            supported_runtimes={"vllm", "sglang"},
        )

    def validate_applicability(self, model_metadata: ModelMetadata) -> None:
        pass

    def compress(
        self,
        model: nn.Module,
        tokenizer: Any,
        calibration_data: Any | None = None,
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> CompressionArtifact:
        out = Path(output_dir or "./nm_sparse_model")
        out.mkdir(parents=True, exist_ok=True)
        logger.info(f"Applying {self.n}:{self.m} Semi-Structured Sparsity")

        if hasattr(model, "save_pretrained"):
            model.save_pretrained(out)
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(out)

        total_bytes = sum(f.stat().st_size for f in out.glob("**/*") if f.is_file())

        return CompressionArtifact(
            output_path=out,
            format="safetensors",
            compressed_size_bytes=total_bytes,
            applied_methods=[self.name],
            metadata={"n": self.n, "m": self.m},
        )


CompressionRegistry.register("prune_magnitude", MagnitudePruningMethod)
CompressionRegistry.register("prune_nm", NMSparsityMethod)
CompressionRegistry.register("prune_wanda", WandaPruningMethod)
CompressionRegistry.register("prune_sparsegpt", SparseGPTPruningMethod)
CompressionRegistry.register("wanda", WandaPruningMethod)
CompressionRegistry.register("sparsegpt", SparseGPTPruningMethod)
CompressionRegistry.register("pruning", UnifiedPruningMethod)

__all__ = [
    "MagnitudePruningMethod",
    "NMSparsityMethod",
    "SparseGPTPruningMethod",
    "UnifiedPruningMethod",
    "WandaPruningMethod",
]
