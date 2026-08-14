"""Native FP8 (E4M3 / E5M2) Quantization Adapter."""

from pathlib import Path
from typing import Any

import torch.nn as nn

from vipym.compression.registry import CompressionRegistry
from vipym.core.constants import ComputeArchitecture, SupportedDtype
from vipym.core.logger import get_logger
from vipym.interfaces.compression import CompressionArtifact, CompressionMethod
from vipym.interfaces.model import ModelMetadata, PluginCapability

logger = get_logger(__name__)


class FP8CompressionMethod(CompressionMethod):
    """Native FP8 (E4M3 / E5M2) weight & activation quantization."""

    def __init__(self, format: str = "fp8_e4m3", static_scales: bool = True) -> None:
        self.format = format
        self.static_scales = static_scales

    @property
    def name(self) -> str:
        return f"fp8_{self.format}"

    def get_capabilities(self) -> PluginCapability:
        return PluginCapability(
            supported_architectures={
                ComputeArchitecture.DENSE,
                ComputeArchitecture.MOE,
                ComputeArchitecture.HYBRID_ATTENTION,
            },
            supported_dtypes={
                SupportedDtype.FP8_E4M3,
                SupportedDtype.FP8_E5M2,
                SupportedDtype.BF16,
            },
            supports_moe=True,
            requires_calibration=self.static_scales,
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
        out = Path(output_dir or "./fp8_model")
        out.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Executing FP8 quantization (format={self.format}, static_scales={self.static_scales})"
        )

        try:
            from llmcompressor.transformers import oneshot

            recipe = """
            quant_stage:
                quant_modifiers:
                    QuantizationModifier:
                        targets: ['Linear']
                        scheme: 'FP8'
                        ignore: ['lm_head']
            """
            oneshot(
                model=model,
                dataset=calibration_data,
                recipe=recipe,
                output_dir=str(out),
            )
        except ImportError:
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(out)
            if hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(out)

        total_bytes = sum(f.stat().st_size for f in out.glob("**/*") if f.is_file())

        return CompressionArtifact(
            output_path=out,
            format="compressed-tensors",
            compressed_size_bytes=total_bytes,
            applied_methods=[self.name],
            metadata={"format": self.format, "static_scales": self.static_scales},
        )


CompressionRegistry.register("fp8", FP8CompressionMethod)
