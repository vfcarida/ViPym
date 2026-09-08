"""KV-Cache Quantization Adapters (FP8 and INT4) with Empirical Scale Calibration.

Calculates layer-wise Key and Value activation scales (E4M3 / E5M2 / INT4) across
attention layers using forward hooks, exporting runtime configurations compatible
with vLLM, SGLang, and TensorRT-LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from vipym.compression.registry import CompressionRegistry
from vipym.core.constants import ComputeArchitecture, SupportedDtype
from vipym.core.logger import get_logger
from vipym.interfaces.compression import CompressionArtifact, CompressionMethod
from vipym.interfaces.model import ModelMetadata, PluginCapability

logger = get_logger(__name__)

# Max representable values per floating point / integer format
_FP8_E4M3_MAX = 448.0
_FP8_E5M2_MAX = 57344.0
_INT4_MAX = 7.0


def _get_fp8_max(kv_dtype: str) -> float:
    """Return max absolute representable range for target KV format."""
    dtype_str = kv_dtype.lower()
    if "e5m2" in dtype_str:
        return _FP8_E5M2_MAX
    if "int4" in dtype_str:
        return _INT4_MAX
    return _FP8_E4M3_MAX


def _collect_kv_activation_scales(
    model: nn.Module,
    calibration_data: Any,
    tokenizer: Any,
    device: torch.device,
    kv_dtype: str,
    scaling_granularity: str = "per_tensor",
) -> dict[str, float]:
    """Profile Key and Value projection activations across calibration samples via forward hooks."""
    fp8_max = _get_fp8_max(kv_dtype)
    max_act_map: dict[str, float] = {}
    hooks = []

    def _make_hook(layer_name: str):
        def _hook(module: nn.Module, inputs: Any, outputs: Any) -> None:
            tensor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            if isinstance(tensor, torch.Tensor):
                current_max = tensor.detach().float().abs().max().item()
                if layer_name not in max_act_map or current_max > max_act_map[layer_name]:
                    max_act_map[layer_name] = max(current_max, 1e-6)

        return _hook

    # Locate Key/Value projections across attention modules
    targeted_layers: list[str] = []
    for name, module in model.named_modules():
        name_lower = name.lower()
        is_kv_layer = any(
            pattern in name_lower
            for pattern in ["k_proj", "v_proj", "qkv_proj", ".k.", ".v.", "c_attn", "key", "value"]
        ) and isinstance(module, (nn.Linear, nn.Module))

        if is_kv_layer and hasattr(module, "register_forward_hook"):
            hooks.append(module.register_forward_hook(_make_hook(name)))
            targeted_layers.append(name)

    if not hooks:
        # Fallback: Hook linear layers inside attention blocks
        for name, module in model.named_modules():
            if ("attn" in name.lower() or "attention" in name.lower()) and isinstance(
                module, nn.Linear
            ):
                hooks.append(module.register_forward_hook(_make_hook(name)))
                targeted_layers.append(name)

    logger.info(
        f"Registered {len(hooks)} forward hooks for KV-cache calibration profiling across attention layers."
    )

    try:
        model.eval()
        with torch.no_grad():
            if isinstance(calibration_data, torch.Tensor):
                model(calibration_data.to(device))
            elif isinstance(calibration_data, (list, tuple)):
                for sample in calibration_data:
                    if isinstance(sample, dict):
                        inp = {
                            k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in sample.items()
                        }
                        model(**inp)
                    elif isinstance(sample, torch.Tensor):
                        model(sample.to(device))
                    elif isinstance(sample, str) and tokenizer is not None and callable(tokenizer):
                        tokens = tokenizer(
                            sample, truncation=True, max_length=512, return_tensors="pt"
                        )
                        inp = {k: v.to(device) for k, v in tokens.items()}
                        model(**inp)
                    elif isinstance(sample, list) and all(isinstance(x, int) for x in sample):
                        # Token IDs sequence
                        input_tensor = torch.tensor([sample], dtype=torch.long, device=device)
                        model(input_tensor)
    finally:
        for hook in hooks:
            hook.remove()

    # Compute scale factor per hooked layer: scale = max_activation / FP8_MAX
    scales: dict[str, float] = {}
    for layer_name, max_val in max_act_map.items():
        scale = max_val / fp8_max
        scales[f"{layer_name}.scale"] = round(max(scale, 1e-6), 6)

    return scales


class KVCacheQuantizationMethod(CompressionMethod):
    """Production KV-Cache Quantization with empirical scale factor profiling (vLLM / SGLang)."""

    def __init__(
        self,
        kv_dtype: str = "fp8_e4m3",
        scaling_granularity: str = "per_tensor",
        device: str | None = None,
    ) -> None:
        self.kv_dtype = kv_dtype.lower()
        self.scaling_granularity = scaling_granularity.lower()
        self.device = device

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
            supported_dtypes={
                SupportedDtype.FP8_E4M3,
                SupportedDtype.FP8_E5M2,
                SupportedDtype.INT4,
                SupportedDtype.BF16,
            },
            supports_moe=True,
            requires_calibration=True,
            supported_runtimes={"vllm", "sglang", "tensorrt-llm"},
        )

    def validate_applicability(self, model_metadata: ModelMetadata) -> None:
        """Validate model compatibility with KV-cache quantization."""
        if model_metadata.architecture_type not in {
            ComputeArchitecture.DENSE,
            ComputeArchitecture.MOE,
            ComputeArchitecture.HYBRID_ATTENTION,
        }:
            logger.warning(
                f"Model '{model_metadata.model_id}' architecture '{model_metadata.architecture_type}' "
                f"is not standard causal attention; KV-cache quantization may not be applicable."
            )

    def compress(
        self,
        model: nn.Module,
        tokenizer: Any,
        calibration_data: Any | None = None,
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> CompressionArtifact:
        """Calibrate Key/Value activation scales and export vLLM/SGLang runtime artifacts."""
        out = Path(output_dir or "./kv_cache_config")
        out.mkdir(parents=True, exist_ok=True)
        device = (
            torch.device(self.device)
            if self.device
            else next(model.parameters()).device
            if list(model.parameters())
            else torch.device("cpu")
        )

        logger.info(
            f"Configuring KV-Cache Quantization: format={self.kv_dtype}, granularity={self.scaling_granularity}"
        )

        # 1. Profile empirical activation scales via forward hooks if calibration data is available
        scales: dict[str, float] = {}
        if calibration_data is not None:
            logger.info(
                "Executing forward hooks on attention layers to collect empirical KV scales..."
            )
            scales = _collect_kv_activation_scales(
                model=model,
                calibration_data=calibration_data,
                tokenizer=tokenizer,
                device=device,
                kv_dtype=self.kv_dtype,
                scaling_granularity=self.scaling_granularity,
            )
            logger.info(f"Successfully computed {len(scales)} empirical KV scale factors.")
        else:
            # Deterministic fallback: safe normalized scale based on standard head dimension
            logger.warning(
                "No calibration data provided for KV-cache quantization. Generating normalized default scales."
            )
            fp8_max = _get_fp8_max(self.kv_dtype)
            default_scale = round(16.0 / fp8_max, 6)
            for name, module in model.named_modules():
                if any(
                    k in name.lower() for k in ["k_proj", "v_proj", "key", "value"]
                ) and isinstance(module, nn.Linear):
                    scales[f"{name}.scale"] = default_scale

        # 2. Export kv_cache_scales.json for vLLM and SGLang
        scales_path = out / "kv_cache_scales.json"
        scales_payload = {
            "kv_cache_dtype": self.kv_dtype,
            "scaling_granularity": self.scaling_granularity,
            "scales_count": len(scales),
            "scales": scales,
        }
        scales_path.write_text(json.dumps(scales_payload, indent=2), encoding="utf-8")
        logger.info(f"Saved KV-cache scales to {scales_path.resolve()}")

        # 3. Export serving_config.json for runtime configuration injection
        serving_cfg_path = out / "serving_config.json"
        serving_payload = {
            "kv_cache_dtype": "fp8" if "fp8" in self.kv_dtype else self.kv_dtype,
            "kv_cache_scheme": self.kv_dtype,
            "kv_cache_scales_file": "kv_cache_scales.json",
            "calculate_kv_scales": False,
            "scaling_granularity": self.scaling_granularity,
        }
        serving_cfg_path.write_text(json.dumps(serving_payload, indent=2), encoding="utf-8")

        # 4. Save model weights and tokenizer if present
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(out)
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(out)

        # 5. Inject quantization_config into config.json if present
        config_json_path = out / "config.json"
        if config_json_path.exists():
            try:
                cfg_data = json.loads(config_json_path.read_text(encoding="utf-8"))
                quant_cfg = cfg_data.get("quantization_config", {})
                quant_cfg["kv_cache_dtype"] = self.kv_dtype
                quant_cfg["kv_cache_scales_file"] = "kv_cache_scales.json"
                cfg_data["quantization_config"] = quant_cfg
                config_json_path.write_text(json.dumps(cfg_data, indent=2), encoding="utf-8")
            except Exception as e:
                logger.debug(f"Could not update config.json: {e}")

        total_bytes = sum(f.stat().st_size for f in out.glob("**/*") if f.is_file())
        reduction_factor = "2.0x" if "fp8" in self.kv_dtype else "4.0x"

        return CompressionArtifact(
            output_path=out,
            format="safetensors",
            compressed_size_bytes=total_bytes,
            applied_methods=[self.name],
            metadata={
                "kv_cache_dtype": self.kv_dtype,
                "scaling_granularity": self.scaling_granularity,
                "scales_file": "kv_cache_scales.json",
                "num_layers_scaled": len(scales),
                "estimated_kv_cache_reduction": reduction_factor,
            },
        )


# Register in CompressionRegistry (both direct and factory lambda)
CompressionRegistry.register("kv_cache_fp8", KVCacheQuantizationMethod)
CompressionRegistry.register("kv_cache_fp8_e4m3", lambda: KVCacheQuantizationMethod("fp8_e4m3"))
CompressionRegistry.register("kv_cache_fp8_e5m2", lambda: KVCacheQuantizationMethod("fp8_e5m2"))
CompressionRegistry.register("kv_cache_int4", lambda: KVCacheQuantizationMethod("int4"))
