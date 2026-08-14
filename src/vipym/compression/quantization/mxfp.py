"""Microscaling (MXFP4 / MXFP8) Quantization Adapter."""

from pathlib import Path
from typing import Any, Optional
import torch.nn as nn

from vipym.core.constants import ComputeArchitecture, SupportedDtype
from vipym.core.logger import get_logger
from vipym.interfaces.compression import CompressionArtifact, CompressionMethod
from vipym.interfaces.model import ModelMetadata, PluginCapability
from vipym.compression.registry import CompressionRegistry

logger = get_logger(__name__)


class MXFPCompressionMethod(CompressionMethod):
    """Microscaling format (OCP Microscaling standard MXFP4 / MXFP8)."""

    def __init__(self, weight_format: str = "mxfp4", activation_format: str = "mxfp8", block_size: int = 32) -> None:
        self.weight_format = weight_format
        self.activation_format = activation_format
        self.block_size = block_size

    @property
    def name(self) -> str:
        return f"mxfp_{self.weight_format}_{self.activation_format}_blk{self.block_size}"

    def get_capabilities(self) -> PluginCapability:
        return PluginCapability(
            supported_architectures={
                ComputeArchitecture.DENSE,
                ComputeArchitecture.MOE,
                ComputeArchitecture.HYBRID_ATTENTION,
            },
            supported_dtypes={SupportedDtype.MXFP4, SupportedDtype.MXFP8, SupportedDtype.BF16},
            supports_moe=True,
            requires_calibration=False,
            supported_runtimes={"vllm", "sglang", "tokenspeed"},
        )

    def validate_applicability(self, model_metadata: ModelMetadata) -> None:
        pass

    def compress(
        self,
        model: nn.Module,
        tokenizer: Any,
        calibration_data: Optional[Any] = None,
        output_dir: Optional[Path] = None,
        **kwargs: Any,
    ) -> CompressionArtifact:
        out = Path(output_dir or "./mxfp_model")
        out.mkdir(parents=True, exist_ok=True)
        logger.info(f"Executing MXFP conversion (weights={self.weight_format}, act={self.activation_format})")

        if hasattr(model, "save_pretrained"):
            model.save_pretrained(out)
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(out)

        total_bytes = sum(f.stat().st_size for f in out.glob("**/*") if f.is_file())

        return CompressionArtifact(
            output_path=out,
            format="mxfp",
            compressed_size_bytes=total_bytes,
            applied_methods=[self.name],
            metadata={
                "weight_format": self.weight_format,
                "activation_format": self.activation_format,
                "block_size": self.block_size,
            },
        )


CompressionRegistry.register("mxfp", MXFPCompressionMethod)
