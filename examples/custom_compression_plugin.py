"""Example demonstrating how to write and register a custom compression algorithm plugin in ViPym."""

from pathlib import Path
from typing import Any

import torch.nn as nn

from vipym.compression.registry import CompressionRegistry
from vipym.config.constants import ComputeArchitecture, SupportedDtype
from vipym.interfaces.compression import CompressionArtifact, CompressionMethod
from vipym.interfaces.model import ModelMetadata, PluginCapability


class CustomQuantizationPlugin(CompressionMethod):
    """Custom Quantization Algorithm (e.g. Novel Weight-Only Int3 format)."""

    def __init__(self, target_bits: int = 3) -> None:
        self.target_bits = target_bits

    @property
    def name(self) -> str:
        return f"custom_int{self.target_bits}_quantizer"

    def get_capabilities(self) -> PluginCapability:
        return PluginCapability(
            supported_architectures={ComputeArchitecture.DENSE, ComputeArchitecture.MOE},
            supported_dtypes={SupportedDtype.INT4, SupportedDtype.FP16, SupportedDtype.BF16},
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
        out = Path(output_dir or "./custom_compressed_weights")
        out.mkdir(parents=True, exist_ok=True)
        print(f"Applying custom {self.target_bits}-bit quantization logic...")

        if hasattr(model, "save_pretrained"):
            model.save_pretrained(out)
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(out)

        return CompressionArtifact(
            output_path=out,
            format="custom-safetensors",
            compressed_size_bytes=1000000,
            applied_methods=[self.name],
            metadata={"bits": self.target_bits},
        )


# Register the plugin into ViPym's dynamic registry
CompressionRegistry.register("custom_int3", CustomQuantizationPlugin)
print(f"Registered custom plugin: {CompressionRegistry.list_methods().keys()}")
