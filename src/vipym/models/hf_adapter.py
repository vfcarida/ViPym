"""Hugging Face Model Adapter implementation."""

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from vipym.core.constants import ComputeArchitecture, SupportedDtype
from vipym.core.exceptions import ModelAdapterError
from vipym.interfaces.model import ModelAdapter, ModelMetadata, PluginCapability
from vipym.models.registry import ModelRegistry


class HuggingFaceModelAdapter(ModelAdapter):
    """General adapter for standard Hugging Face AutoModel architectures."""

    def get_capabilities(self) -> PluginCapability:
        return PluginCapability(
            supported_architectures={ComputeArchitecture.DENSE, ComputeArchitecture.MOE},
            supported_dtypes={
                SupportedDtype.FP32,
                SupportedDtype.FP16,
                SupportedDtype.BF16,
                SupportedDtype.FP8_E4M3,
            },
            supports_moe=True,
            supported_runtimes={"vllm", "sglang", "hf"},
        )

    def inspect_metadata(self, model_id_or_path: str, revision: str = "main") -> ModelMetadata:
        try:
            config = AutoConfig.from_pretrained(
                model_id_or_path,
                revision=revision,
                trust_remote_code=True,
            )
        except Exception as e:
            raise ModelAdapterError(
                f"Failed to inspect HuggingFace config for '{model_id_or_path}': {e}"
            ) from e

        config_dict = config.to_dict() if hasattr(config, "to_dict") else {}

        # Estimate parameter count from config
        hidden_size = getattr(config, "hidden_size", 4096)
        num_layers = getattr(config, "num_hidden_layers", 32)
        num_heads = getattr(config, "num_attention_heads", 32)
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        vocab_size = getattr(config, "vocab_size", 32000)
        max_pos = getattr(config, "max_position_embeddings", 4096)
        num_experts = getattr(
            config, "num_local_experts", getattr(config, "n_routed_experts", None)
        )
        num_selected = getattr(
            config, "num_experts_per_tok", getattr(config, "num_active_experts", None)
        )

        is_moe = num_experts is not None and num_experts > 1
        arch_type = ComputeArchitecture.MOE if is_moe else ComputeArchitecture.DENSE

        # Rough calculation of parameter size if not directly specified
        total_params = getattr(config, "total_params", None)
        if total_params is None:
            # Estimate standard transformer dense or MoE
            emb_params = vocab_size * hidden_size
            layer_params = 4 * hidden_size * hidden_size + 3 * hidden_size * (hidden_size * 4)
            if is_moe and num_experts:
                layer_params = 4 * hidden_size * hidden_size + num_experts * (
                    3 * hidden_size * (hidden_size * 4) // 2
                )
            total_params = emb_params + num_layers * layer_params

        active_params = total_params
        if is_moe and num_experts and num_selected:
            active_params = emb_params + num_layers * (
                4 * hidden_size * hidden_size
                + num_selected * (3 * hidden_size * (hidden_size * 4) // 2)
            )

        return ModelMetadata(
            model_id=model_id_or_path,
            revision=revision,
            total_parameters=int(total_params),
            active_parameters=int(active_params),
            architecture_type=arch_type,
            native_dtypes=[SupportedDtype.BF16, SupportedDtype.FP16],
            context_window=int(max_pos),
            num_layers=int(num_layers),
            hidden_size=int(hidden_size),
            num_attention_heads=int(num_heads),
            num_key_value_heads=int(num_kv_heads) if num_kv_heads else None,
            num_experts=int(num_experts) if num_experts else None,
            num_selected_experts=int(num_selected) if num_selected else None,
            raw_config=config_dict,
        )

    def load_for_compression(
        self,
        model_id_or_path: str,
        revision: str = "main",
        device_map: str | None = "auto",
        torch_dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> nn.Module:
        try:
            is_cuda = torch.cuda.is_available()
            dtype = torch_dtype or (
                torch.bfloat16
                if (is_cuda and torch.cuda.is_bf16_supported())
                else (torch.float16 if is_cuda else torch.float32)
            )
            dev_map = device_map if is_cuda else None
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id_or_path,
                    revision=revision,
                    device_map=dev_map,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    **kwargs,
                )
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id_or_path,
                    revision=revision,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    **kwargs,
                )
            return model
        except Exception as e:
            raise ModelAdapterError(
                f"Failed to load model for compression '{model_id_or_path}': {e}"
            ) from e

    def get_tokenizer(self, model_id_or_path: str, revision: str = "main") -> Any:
        try:
            return AutoTokenizer.from_pretrained(
                model_id_or_path,
                revision=revision,
                trust_remote_code=True,
            )
        except Exception as e:
            raise ModelAdapterError(
                f"Failed to load tokenizer for '{model_id_or_path}': {e}"
            ) from e


ModelRegistry.register("hf", HuggingFaceModelAdapter)
ModelRegistry.register("huggingface", HuggingFaceModelAdapter)
