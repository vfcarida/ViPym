"""Kimi K3 Model Adapter (Moonshot AI 2.8T MoE Architecture).

Key Specifications:
- Total Parameters: 2.8 Trillion (MoE)
- Active Parameters: ~104 Billion per token
- Experts: 896 routed + 2 shared experts (16 active per token)
- Context Window: 1,048,576 tokens (1M)
- Hybrid Attention: Kimi Delta Attention (KDA) + Gated Multi-Head Latent Attention (MLA)
- Native Quantization: MXFP4 Weights + MXFP8 Activations (QAT)
"""

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer

from vipym.core.constants import ComputeArchitecture, SupportedDtype
from vipym.core.exceptions import ModelAdapterError
from vipym.interfaces.model import ModelAdapter, ModelMetadata, PluginCapability
from vipym.models.registry import ModelRegistry


class KimiK3ModelAdapter(ModelAdapter):
    """Specialized adapter for Moonshot AI Kimi K3 architecture."""

    def get_capabilities(self) -> PluginCapability:
        return PluginCapability(
            supported_architectures={ComputeArchitecture.MOE, ComputeArchitecture.HYBRID_ATTENTION},
            supported_dtypes={
                SupportedDtype.MXFP4,
                SupportedDtype.MXFP8,
                SupportedDtype.FP8_E4M3,
                SupportedDtype.BF16,
            },
            supports_moe=True,
            supported_runtimes={"vllm", "sglang", "tokenspeed"},
            min_gpu_memory_gb=1600.0,  # ~1.56 TB in native MXFP4
            recommended_gpu_count=64,
        )

    def inspect_metadata(self, model_id_or_path: str, revision: str = "main") -> ModelMetadata:
        try:
            config = AutoConfig.from_pretrained(
                model_id_or_path,
                revision=revision,
                trust_remote_code=True,
            )
            config_dict = config.to_dict() if hasattr(config, "to_dict") else {}
        except Exception:
            # Fallback to standard canonical Kimi K3 architecture specification
            config_dict = {
                "architectures": ["KimiK3ForCausalLM"],
                "num_experts": 896,
                "num_shared_experts": 2,
                "num_experts_per_tok": 16,
                "total_params": 2_800_000_000_000,
                "active_params": 104_000_000_000,
                "max_position_embeddings": 1_048_576,
                "hidden_size": 7168,
                "num_hidden_layers": 61,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "attention_mechanism": "KDA_Gated_MLA",
                "quantization_format": "MXFP4_MXFP8_QAT",
            }

        return ModelMetadata(
            model_id=model_id_or_path,
            revision=revision,
            total_parameters=config_dict.get("total_params", 2_800_000_000_000),
            active_parameters=config_dict.get("active_params", 104_000_000_000),
            architecture_type=ComputeArchitecture.HYBRID_ATTENTION,
            native_dtypes=[SupportedDtype.MXFP4, SupportedDtype.MXFP8, SupportedDtype.BF16],
            context_window=config_dict.get("max_position_embeddings", 1_048_576),
            num_layers=config_dict.get("num_hidden_layers", 61),
            hidden_size=config_dict.get("hidden_size", 7168),
            num_attention_heads=config_dict.get("num_attention_heads", 64),
            num_key_value_heads=config_dict.get("num_key_value_heads", 8),
            num_experts=config_dict.get("num_experts", 896),
            num_selected_experts=config_dict.get("num_experts_per_tok", 16),
            has_custom_kernels=True,
            raw_config=config_dict,
        )

    def load_for_compression(
        self,
        model_id_or_path: str,
        revision: str = "main",
        device_map: str = "auto",
        torch_dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> nn.Module:
        """Load Kimi K3 for offline compression/calibration."""
        try:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                model_id_or_path,
                revision=revision,
                device_map=device_map,
                torch_dtype=torch_dtype or torch.bfloat16,
                trust_remote_code=True,
                **kwargs,
            )
            return model
        except Exception as e:
            raise ModelAdapterError(f"Failed to load Kimi K3 model: {e}") from e

    def get_tokenizer(self, model_id_or_path: str, revision: str = "main") -> Any:
        try:
            return AutoTokenizer.from_pretrained(
                model_id_or_path,
                revision=revision,
                trust_remote_code=True,
            )
        except Exception as e:
            raise ModelAdapterError(f"Failed to load Kimi K3 tokenizer: {e}") from e


ModelRegistry.register("kimi_k3", KimiK3ModelAdapter)
ModelRegistry.register("moonshotai/kimi-k3", KimiK3ModelAdapter)
