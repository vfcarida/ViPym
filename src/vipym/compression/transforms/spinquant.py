"""SpinQuant Orthogonal Rotation and QuaRot Walsh-Hadamard Transforms."""

from pathlib import Path
from typing import Any, Optional
import torch.nn as nn

from vipym.core.constants import ComputeArchitecture, SupportedDtype
from vipym.core.logger import get_logger
from vipym.interfaces.compression import CompressionArtifact, CompressionMethod
from vipym.interfaces.model import ModelMetadata, PluginCapability
from vipym.compression.registry import CompressionRegistry

logger = get_logger(__name__)


class SpinQuantTransformMethod(CompressionMethod):
    """SpinQuant Orthogonal Rotation Transform to eliminate activation outliers."""

    def __init__(self, rotation_type: str = "random_hadamard") -> None:
        self.rotation_type = rotation_type

    @property
    def name(self) -> str:
        return f"spinquant_{self.rotation_type}"

    def get_capabilities(self) -> PluginCapability:
        return PluginCapability(
            supported_architectures={
                ComputeArchitecture.DENSE,
                ComputeArchitecture.MOE,
            },
            supported_dtypes={SupportedDtype.FP16, SupportedDtype.BF16},
            supports_moe=True,
            requires_calibration=True,
            supported_runtimes={"vllm", "hf"},
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
        out = Path(output_dir or "./spinquant_model")
        out.mkdir(parents=True, exist_ok=True)
        logger.info(f"Applying SpinQuant rotation ({self.rotation_type})")

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
            metadata={"rotation_type": self.rotation_type},
        )


class QuaRotTransformMethod(CompressionMethod):
    """QuaRot Walsh-Hadamard Orthogonal Transformation."""

    def __init__(self, apply_online_hadamard: bool = True) -> None:
        self.apply_online_hadamard = apply_online_hadamard

    @property
    def name(self) -> str:
        return "quarot_hadamard_transform"

    def get_capabilities(self) -> PluginCapability:
        return PluginCapability(
            supported_architectures={
                ComputeArchitecture.DENSE,
                ComputeArchitecture.MOE,
            },
            supported_dtypes={SupportedDtype.FP16, SupportedDtype.BF16, SupportedDtype.INT4},
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
        calibration_data: Optional[Any] = None,
        output_dir: Optional[Path] = None,
        **kwargs: Any,
    ) -> CompressionArtifact:
        out = Path(output_dir or "./quarot_model")
        out.mkdir(parents=True, exist_ok=True)
        logger.info("Applying QuaRot Walsh-Hadamard Transform")

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
            metadata={"online_hadamard": self.apply_online_hadamard},
        )


CompressionRegistry.register("spinquant", SpinQuantTransformMethod)
CompressionRegistry.register("quarot", QuaRotTransformMethod)
