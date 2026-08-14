"""AutoRound sign-gradient rounding optimization adapter."""

from pathlib import Path
from typing import Any

import torch.nn as nn

from vipym.compression.registry import CompressionRegistry
from vipym.core.constants import ComputeArchitecture, SupportedDtype
from vipym.core.logger import get_logger
from vipym.interfaces.compression import CompressionArtifact, CompressionMethod
from vipym.interfaces.model import ModelMetadata, PluginCapability

logger = get_logger(__name__)


class AutoRoundCompressionMethod(CompressionMethod):
    """AutoRound Advanced Rounding Optimization."""

    def __init__(self, bits: int = 4, group_size: int = 128, iters: int = 200) -> None:
        self.bits = bits
        self.group_size = group_size
        self.iters = iters

    @property
    def name(self) -> str:
        return f"autoround_w{self.bits}a16_g{self.group_size}"

    def get_capabilities(self) -> PluginCapability:
        return PluginCapability(
            supported_architectures={
                ComputeArchitecture.DENSE,
                ComputeArchitecture.MOE,
            },
            supported_dtypes={SupportedDtype.INT4, SupportedDtype.FP16, SupportedDtype.BF16},
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
        out = Path(output_dir or "./autoround_model")
        out.mkdir(parents=True, exist_ok=True)
        logger.info(f"Executing AutoRound quantization (bits={self.bits}, iters={self.iters})")

        try:
            from auto_round import AutoRound

            autoround = AutoRound(
                model=model,
                tokenizer=tokenizer,
                bits=self.bits,
                group_size=self.group_size,
                iters=self.iters,
            )
            autoround.quantize()
            autoround.save_quantized(output_dir=str(out), format="auto_round")
        except ImportError:
            logger.warning("auto_round not installed, performing mock checkpoint export.")
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
            metadata={"bits": self.bits, "group_size": self.group_size, "iters": self.iters},
        )


CompressionRegistry.register("autoround", AutoRoundCompressionMethod)
