"""KV-Cache Quantization Adapters (FP8 and INT4)."""

from pathlib import Path
from typing import Any

import torch.nn as nn

from vipym.compression.registry import CompressionRegistry
from vipym.core.constants import ComputeArchitecture, SupportedDtype
from vipym.core.logger import get_logger
from vipym.interfaces.compression import CompressionArtifact, CompressionMethod
from vipym.interfaces.model import ModelMetadata, PluginCapability

logger = get_logger(__name__)


class KVCacheQuantizationMethod(CompressionMethod):
    """KV-Cache Quantization for serving runtimes (e.g. vLLM kv_cache_dtype='fp8')."""

    def __init__(self, kv_dtype: str = "fp8_e4m3") -> None:
        self.kv_dtype = kv_dtype

    @property
    def name(self) -> str:
        return f"kv_cache_{self.kv_dtype}"

    def get_capabilities(self) -> PluginCapability:
        return PluginCapability(
            supported_architectures={
                ComputeArchitecture.DENSE,
                ComputeArchitecture.MOE,
                ComputeArchitecture.HYBRID_ATTENTION,
            },
            supported_dtypes={SupportedDtype.FP8_E4M3, SupportedDtype.INT4, SupportedDtype.BF16},
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
        out = Path(output_dir or "./kv_cache_config")
        out.mkdir(parents=True, exist_ok=True)
        logger.info(f"Configuring KV-Cache Quantization ({self.kv_dtype})")

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
            metadata={"kv_cache_dtype": self.kv_dtype},
        )


CompressionRegistry.register("kv_cache_fp8", lambda: KVCacheQuantizationMethod("fp8_e4m3"))
CompressionRegistry.register("kv_cache_int4", lambda: KVCacheQuantizationMethod("int4"))
