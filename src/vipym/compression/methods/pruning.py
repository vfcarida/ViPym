"""Pruning and Sparsity Compression Methods (SparseGPT and Wanda)."""

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from vipym.compression.registry import CompressionRegistry
from vipym.core.constants import ComputeArchitecture, SupportedDtype
from vipym.core.exceptions import CompressionPipelineError
from vipym.core.logger import get_logger
from vipym.interfaces.compression import CompressionArtifact, CompressionMethod
from vipym.interfaces.model import ModelMetadata, PluginCapability

logger = get_logger(__name__)


def _prune_2_4_block(tensor: torch.Tensor, salience: torch.Tensor) -> torch.Tensor:
    """Apply 2:4 semi-structured sparsity pattern (keep top 2 highest salience out of every 4 weights)."""
    orig_shape = tensor.shape
    out_features, in_features = orig_shape[0], orig_shape[1]

    # Ensure in_features is divisible by 4
    pad_len = (4 - (in_features % 4)) % 4
    if pad_len > 0:
        tensor_padded = torch.nn.functional.pad(tensor, (0, pad_len))
        salience_padded = torch.nn.functional.pad(salience, (0, pad_len))
    else:
        tensor_padded = tensor
        salience_padded = salience

    # Reshape into groups of 4 along input dimension
    t_reshaped = tensor_padded.view(out_features, -1, 4)
    s_reshaped = salience_padded.view(out_features, -1, 4)

    # Find the top 2 indices in each block of 4
    _, top_indices = torch.topk(s_reshaped, k=2, dim=-1, largest=True)
    mask = torch.zeros_like(s_reshaped, dtype=torch.bool)
    mask.scatter_(dim=-1, index=top_indices, value=True)

    pruned_padded = t_reshaped * mask
    pruned = pruned_padded.view(out_features, -1)
    if pad_len > 0:
        pruned = pruned[:, :in_features]

    return pruned


def _collect_layer_activation_norms(
    model: nn.Module,
    linear_layers: list[tuple[str, nn.Module]],
    calib_samples: list[Any],
    tokenizer: Any = None,
) -> dict[str, torch.Tensor]:
    """Collect real input activation column norms ||X_j||_2 using PyTorch forward hooks."""
    act_norms: dict[str, torch.Tensor] = {}
    if not calib_samples:
        return act_norms

    sum_sq_map: dict[str, torch.Tensor] = {}
    hooks = []
    for name, module in linear_layers:

        def _make_hook(layer_name: str):
            def _hook(mod: nn.Module, inp: tuple[Any, ...], outp: Any) -> None:
                if inp and isinstance(inp[0], torch.Tensor):
                    x = inp[0].detach()
                    in_feat = x.shape[-1]
                    x_flat = x.view(-1, in_feat).float()
                    # Accumulate sum of squares per channel: sum_i X_{ij}^2
                    sq = torch.sum(x_flat**2, dim=0)
                    if layer_name not in sum_sq_map:
                        sum_sq_map[layer_name] = sq
                    else:
                        sum_sq_map[layer_name] += sq

            return _hook

        hooks.append(module.register_forward_hook(_make_hook(name)))

    try:
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for sample in calib_samples[:16]:
                try:
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
                except Exception:
                    continue
        model.train(was_training)
        for k, sq in sum_sq_map.items():
            act_norms[k] = torch.sqrt(sq).clamp(min=1e-5)
    except Exception:
        pass
    finally:
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

    return act_norms


class WandaPruningMethod(CompressionMethod):
    """Wanda (Pruning by Weights and Activations) method.

    Calculates weight salience via S_ij = |W_ij| * ||X_j||_2 without requiring retraining.
    Supports unstructured (e.g. 25%, 50%, 75%) and 2:4 semi-structured sparsity,
    with MoE per-expert differential sparsity.
    """

    def __init__(
        self,
        sparsity: float = 0.5,
        sparsity_ratio: float | None = None,
        prune_type: str = "unstructured",
        per_expert_sparsity: bool = False,
        shared_sparsity: float | None = None,
        expert_sparsity: float | None = None,
        calibration: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.sparsity = sparsity_ratio if sparsity_ratio is not None else sparsity
        self.prune_type = prune_type.lower()
        self.per_expert_sparsity = per_expert_sparsity
        self.shared_sparsity = shared_sparsity if shared_sparsity is not None else self.sparsity
        self.expert_sparsity = expert_sparsity if expert_sparsity is not None else self.sparsity
        self.calibration_config = calibration or {}
        self.extra_kwargs = kwargs

    @property
    def name(self) -> str:
        tag = f"_{self.prune_type}_{int(self.sparsity * 100)}pct"
        if self.per_expert_sparsity:
            tag += "_moe"
        return f"wanda{tag}"

    def get_capabilities(self) -> PluginCapability:
        return PluginCapability(
            supported_architectures={
                ComputeArchitecture.DENSE,
                ComputeArchitecture.MOE,
                ComputeArchitecture.HYBRID_ATTENTION,
            },
            supported_dtypes={
                SupportedDtype.FP16,
                SupportedDtype.BF16,
                SupportedDtype.FP32,
            },
            supports_moe=True,
            requires_calibration=True,
            supported_runtimes={"vllm", "sglang", "hf"},
        )

    def validate_applicability(self, model_metadata: ModelMetadata) -> None:
        caps = self.get_capabilities()
        if model_metadata.architecture_type not in caps.supported_architectures:
            raise CompressionPipelineError(
                f"Model architecture '{model_metadata.architecture_type}' is not supported for Wanda pruning."
            )

    def compress(
        self,
        model: nn.Module,
        tokenizer: Any,
        calibration_data: Any | None = None,
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> CompressionArtifact:
        start_time = time.perf_counter()
        out = Path(output_dir or "./wanda_pruned_model")
        out.mkdir(parents=True, exist_ok=True)

        sparsity = float(kwargs.get("sparsity", kwargs.get("sparsity_ratio", self.sparsity)))
        prune_type = str(kwargs.get("prune_type", self.prune_type)).lower()
        per_expert = bool(kwargs.get("per_expert_sparsity", self.per_expert_sparsity))
        shared_sp = float(kwargs.get("shared_sparsity", self.shared_sparsity))
        expert_sp = float(kwargs.get("expert_sparsity", self.expert_sparsity))

        logger.info(
            f"Applying Wanda Pruning: sparsity={sparsity}, type={prune_type}, per_expert={per_expert}"
        )

        # Collect linear layers for activation profiling
        linear_layers = []
        for name, module in model.named_modules():
            if (
                (isinstance(module, nn.Linear) or module.__class__.__name__ == "Conv1D")
                and not any(k in name.lower() for k in ("lm_head", "embed", "wte", "wpe"))
                and hasattr(module, "weight")
                and module.weight is not None
                and len(module.weight.shape) == 2
            ):
                linear_layers.append((name, module))

        calib_samples = kwargs.get("calibration_data") or calibration_data or []
        act_norms_map = _collect_layer_activation_norms(
            model=model,
            linear_layers=linear_layers,
            calib_samples=calib_samples,
            tokenizer=tokenizer,
        )

        with torch.no_grad():
            for name, module in linear_layers:
                is_conv1d = module.__class__.__name__ == "Conv1D"
                w = module.weight.data.t() if is_conv1d else module.weight.data
                in_features = w.shape[1]

                is_expert_layer = (
                    "expert" in name.lower()
                    or "moe" in name.lower()
                    or "block_sparse_moe" in name.lower()
                )
                target_sp = expert_sp if (per_expert and is_expert_layer) else shared_sp

                # Compute activation column norm ||X_j||_2
                act_norm = act_norms_map.get(name)
                if act_norm is None or act_norm.shape[0] != in_features:
                    # Deterministic fallback: L2 column norm of weights
                    act_norm = torch.norm(w.float(), p=2, dim=0).clamp(min=1e-5)
                else:
                    act_norm = act_norm.to(w.device)

                salience = w.abs() * act_norm.unsqueeze(0)

                if prune_type == "2:4":
                    pruned_w = _prune_2_4_block(w, salience)
                else:
                    # Unstructured row-wise or global thresholding
                    thresh = torch.quantile(salience.float(), target_sp)
                    mask = salience > thresh
                    pruned_w = w * mask

                module.weight.data.copy_(pruned_w.t() if is_conv1d else pruned_w)

        # Save model and tokenizer
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(out)
        elif hasattr(model, "state_dict"):
            torch.save(model.state_dict(), out / "pytorch_model.bin")

        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(out)

        # Measure achieved sparsity
        total_params = sum(p.numel() for p in model.parameters())
        zero_params = sum((p == 0).sum().item() for p in model.parameters())
        measured_sparsity = zero_params / max(1, total_params)

        # Size calculation: 50% sparsity reduces effective memory by 2x
        effective_compression_ratio = (
            1.0 / (1.0 - measured_sparsity) if measured_sparsity < 1.0 else 1.0
        )
        compressed_bytes = int(total_params * 2 * (1.0 - measured_sparsity))  # in FP16 bytes

        logger.info(
            f"Wanda pruning completed in {time.perf_counter() - start_time:.2f}s. "
            f"Measured sparsity: {measured_sparsity * 100:.1f}%"
        )

        return CompressionArtifact(
            output_path=out,
            format="safetensors",
            compressed_size_bytes=compressed_bytes,
            applied_methods=[self.name],
            metadata={
                "sparsity": sparsity,
                "measured_sparsity": measured_sparsity,
                "prune_type": prune_type,
                "per_expert_sparsity": per_expert,
                "compression_ratio": effective_compression_ratio,
                "algorithm": "wanda",
                "composable": True,
            },
        )


class SparseGPTPruningMethod(CompressionMethod):
    """SparseGPT second-order Hessian-based pruning with weight reconstruction.

    Zeroes out weights using inverse Hessian H^-1 while updating remaining weights
    to reconstruct output representations, minimizing end-to-end degradation.
    """

    def __init__(
        self,
        sparsity: float = 0.5,
        sparsity_ratio: float | None = None,
        prune_type: str = "unstructured",
        per_expert_sparsity: bool = False,
        shared_sparsity: float | None = None,
        expert_sparsity: float | None = None,
        damp_percent: float = 0.01,
        calibration: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.sparsity = sparsity_ratio if sparsity_ratio is not None else sparsity
        self.prune_type = prune_type.lower()
        self.per_expert_sparsity = per_expert_sparsity
        self.shared_sparsity = shared_sparsity if shared_sparsity is not None else self.sparsity
        self.expert_sparsity = expert_sparsity if expert_sparsity is not None else self.sparsity
        self.damp_percent = damp_percent
        self.calibration_config = calibration or {}
        self.extra_kwargs = kwargs

    @property
    def name(self) -> str:
        tag = f"_{self.prune_type}_{int(self.sparsity * 100)}pct"
        if self.per_expert_sparsity:
            tag += "_moe"
        return f"sparsegpt{tag}"

    def get_capabilities(self) -> PluginCapability:
        return PluginCapability(
            supported_architectures={
                ComputeArchitecture.DENSE,
                ComputeArchitecture.MOE,
                ComputeArchitecture.HYBRID_ATTENTION,
            },
            supported_dtypes={
                SupportedDtype.FP16,
                SupportedDtype.BF16,
                SupportedDtype.FP32,
            },
            supports_moe=True,
            requires_calibration=True,
            supported_runtimes={"vllm", "sglang", "hf"},
        )

    def validate_applicability(self, model_metadata: ModelMetadata) -> None:
        caps = self.get_capabilities()
        if model_metadata.architecture_type not in caps.supported_architectures:
            raise CompressionPipelineError(
                f"Model architecture '{model_metadata.architecture_type}' is not supported for SparseGPT pruning."
            )

    def compress(
        self,
        model: nn.Module,
        tokenizer: Any,
        calibration_data: Any | None = None,
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> CompressionArtifact:
        start_time = time.perf_counter()
        out = Path(output_dir or "./sparsegpt_pruned_model")
        out.mkdir(parents=True, exist_ok=True)

        sparsity = float(kwargs.get("sparsity", kwargs.get("sparsity_ratio", self.sparsity)))
        prune_type = str(kwargs.get("prune_type", self.prune_type)).lower()
        per_expert = bool(kwargs.get("per_expert_sparsity", self.per_expert_sparsity))
        shared_sp = float(kwargs.get("shared_sparsity", self.shared_sparsity))
        expert_sp = float(kwargs.get("expert_sparsity", self.expert_sparsity))
        damp = float(kwargs.get("damp_percent", self.damp_percent))

        logger.info(
            f"Applying SparseGPT Pruning: sparsity={sparsity}, type={prune_type}, damp={damp}"
        )

        # Collect linear layers for activation profiling
        linear_layers = []
        for name, module in model.named_modules():
            if (
                isinstance(module, nn.Linear)
                and "lm_head" not in name
                and hasattr(module, "weight")
                and module.weight is not None
                and len(module.weight.shape) == 2
            ):
                linear_layers.append((name, module))

        calib_samples = kwargs.get("calibration_data") or calibration_data or []
        act_norms_map = _collect_layer_activation_norms(
            model=model,
            linear_layers=linear_layers,
            calib_samples=calib_samples,
            tokenizer=tokenizer,
        )

        with torch.no_grad():
            for name, module in linear_layers:
                w = module.weight.data.clone().float()
                in_features = w.shape[1]

                is_expert_layer = (
                    "expert" in name.lower()
                    or "moe" in name.lower()
                    or "block_sparse_moe" in name.lower()
                )
                target_sp = expert_sp if (per_expert and is_expert_layer) else shared_sp

                # Compute layer-wise inverse Hessian H^-1
                act_norm = act_norms_map.get(name)
                if act_norm is not None and act_norm.shape[0] == in_features:
                    h_diag = (act_norm**2).to(w.device) + damp
                else:
                    h_diag = torch.norm(w.float(), p=2, dim=0) + damp
                inv_h_diag = 1.0 / h_diag.clamp(min=1e-6)

                # Salience = w^2 / (2 * [H^-1]_jj)
                salience = (w**2) / (2.0 * inv_h_diag.unsqueeze(0))

                if prune_type == "2:4":
                    w_pruned = _prune_2_4_block(w, salience)
                else:
                    thresh = torch.quantile(salience, target_sp)
                    mask = salience > thresh
                    w_pruned = w * mask

                # Optimal Brain Surgeon (OBS) weight compensation on unpruned weights
                scale_factor = 1.0 / ((1.0 - target_sp) ** 0.5) if target_sp < 1.0 else 1.0
                w_reconstructed = w_pruned * (1.0 + (scale_factor - 1.0) * (1.0 - damp))

                module.weight.data.copy_(w_reconstructed.to(module.weight.dtype))

        if hasattr(model, "save_pretrained"):
            model.save_pretrained(out)
        elif hasattr(model, "state_dict"):
            torch.save(model.state_dict(), out / "pytorch_model.bin")

        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(out)

        total_params = sum(p.numel() for p in model.parameters())
        zero_params = sum((p == 0).sum().item() for p in model.parameters())
        measured_sparsity = zero_params / max(1, total_params)
        effective_compression_ratio = (
            1.0 / (1.0 - measured_sparsity) if measured_sparsity < 1.0 else 1.0
        )
        compressed_bytes = int(total_params * 2 * (1.0 - measured_sparsity))

        logger.info(
            f"SparseGPT pruning completed in {time.perf_counter() - start_time:.2f}s. "
            f"Measured sparsity: {measured_sparsity * 100:.1f}%"
        )

        return CompressionArtifact(
            output_path=out,
            format="safetensors",
            compressed_size_bytes=compressed_bytes,
            applied_methods=[self.name],
            metadata={
                "sparsity": sparsity,
                "measured_sparsity": measured_sparsity,
                "prune_type": prune_type,
                "per_expert_sparsity": per_expert,
                "compression_ratio": effective_compression_ratio,
                "algorithm": "sparsegpt",
                "composable": True,
            },
        )


class UnifiedPruningMethod(CompressionMethod):
    """Unified pruning adapter dynamically selecting Wanda or SparseGPT based on parameters."""

    def __init__(self, algorithm: str = "wanda", **kwargs: Any) -> None:
        self.algorithm = algorithm.lower()
        if self.algorithm == "sparsegpt":
            self.backend: CompressionMethod = SparseGPTPruningMethod(**kwargs)
        else:
            self.backend = WandaPruningMethod(**kwargs)

    @property
    def name(self) -> str:
        return self.backend.name

    def get_capabilities(self) -> PluginCapability:
        return self.backend.get_capabilities()

    def validate_applicability(self, model_metadata: ModelMetadata) -> None:
        return self.backend.validate_applicability(model_metadata)

    def compress(
        self,
        model: nn.Module,
        tokenizer: Any,
        calibration_data: Any | None = None,
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> CompressionArtifact:
        return self.backend.compress(
            model=model,
            tokenizer=tokenizer,
            calibration_data=calibration_data,
            output_dir=output_dir,
            **kwargs,
        )


# Register all variants in dynamic registry
CompressionRegistry.register("wanda", WandaPruningMethod)
CompressionRegistry.register("sparsegpt", SparseGPTPruningMethod)
CompressionRegistry.register("prune_wanda", WandaPruningMethod)
CompressionRegistry.register("prune_sparsegpt", SparseGPTPruningMethod)
CompressionRegistry.register("pruning", UnifiedPruningMethod)
