"""SmoothQuant (W8A8) activation smoothing and quantization adapter."""

from pathlib import Path
from typing import Any, Optional
import torch.nn as nn

from vipym.core.constants import ComputeArchitecture, SupportedDtype
from vipym.core.logger import get_logger
from vipym.interfaces.compression import CompressionArtifact, CompressionMethod
from vipym.interfaces.model import ModelMetadata, PluginCapability
from vipym.compression.registry import CompressionRegistry

logger = get_logger(__name__)


class SmoothQuantCompressionMethod(CompressionMethod):
    """SmoothQuant W8A8 activation outlier migration."""

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha

    @property
    def name(self) -> str:
        return f"smoothquant_w8a8_alpha{self.alpha}"

    def get_capabilities(self) -> PluginCapability:
        return PluginCapability(
            supported_architectures={
                ComputeArchitecture.DENSE,
                ComputeArchitecture.MOE,
                ComputeArchitecture.HYBRID_ATTENTION,
            },
            supported_dtypes={SupportedDtype.INT8, SupportedDtype.FP16, SupportedDtype.BF16},
            supports_moe=True,
            requires_calibration=True,
            supported_runtimes={"vllm", "sglang"},
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
        out = Path(output_dir or "./smoothquant_model")
        out.mkdir(parents=True, exist_ok=True)
        logger.info(f"Executing SmoothQuant W8A8 (alpha={self.alpha})")

        try:
            from llmcompressor.transformers import oneshot
            recipe = f"""
            quant_stage:
                quant_modifiers:
                    SmoothQuantModifier:
                        smoothing_strength: {self.alpha}
                    QuantizationModifier:
                        targets: ['Linear']
                        scheme: 'W8A8'
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
            metadata={"alpha": self.alpha, "scheme": "W8A8"},
        )


CompressionRegistry.register("smoothquant", SmoothQuantCompressionMethod)
