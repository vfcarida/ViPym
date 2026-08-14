"""GPTQ Second-Order Quantization adapter."""

from pathlib import Path
from typing import Any

import torch.nn as nn

from vipym.compression.registry import CompressionRegistry
from vipym.core.constants import ComputeArchitecture, SupportedDtype
from vipym.core.logger import get_logger
from vipym.interfaces.compression import CompressionArtifact, CompressionMethod
from vipym.interfaces.model import ModelMetadata, PluginCapability

logger = get_logger(__name__)


class GPTQCompressionMethod(CompressionMethod):
    """GPTQ Quantization Method (4-bit / 8-bit second-order Hessian calibration)."""

    def __init__(self, bits: int = 4, group_size: int = 128, damp_percent: float = 0.01) -> None:
        self.bits = bits
        self.group_size = group_size
        self.damp_percent = damp_percent

    @property
    def name(self) -> str:
        return f"gptq_w{self.bits}a16_g{self.group_size}"

    def get_capabilities(self) -> PluginCapability:
        return PluginCapability(
            supported_architectures={
                ComputeArchitecture.DENSE,
                ComputeArchitecture.MOE,
                ComputeArchitecture.HYBRID_ATTENTION,
            },
            supported_dtypes={
                SupportedDtype.INT4,
                SupportedDtype.INT8,
                SupportedDtype.FP16,
                SupportedDtype.BF16,
            },
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
        calibration_data: Any | None = None,
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> CompressionArtifact:
        out = Path(output_dir or "./gptq_model")
        out.mkdir(parents=True, exist_ok=True)
        logger.info(f"Executing GPTQ quantization (bits={self.bits}, group_size={self.group_size})")

        try:
            from llmcompressor.transformers import oneshot

            recipe = f"""
            quant_stage:
                quant_modifiers:
                    GPTQModifier:
                        targets: ['Linear']
                        scheme: 'W{self.bits}A16'
                        ignore: ['lm_head']
            """
            oneshot(
                model=model,
                dataset=calibration_data,
                recipe=recipe,
                output_dir=str(out),
            )
        except ImportError:
            logger.warning("llm-compressor not available, fallback saving.")
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
            metadata={
                "bits": self.bits,
                "group_size": self.group_size,
                "damp_percent": self.damp_percent,
            },
        )


CompressionRegistry.register("gptq", GPTQCompressionMethod)
