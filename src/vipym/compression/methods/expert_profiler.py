"""Expert Profiler Method for Mixture-of-Experts (MoE) Architectures."""

import json
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


def _find_moe_blocks(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Discover all MoE layer blocks in the model hierarchy."""
    moe_blocks: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        # Identify by common MoE attributes (experts list or expert_0 attribute)
        if hasattr(module, "experts") or hasattr(module, "expert_0") or "moe" in name.lower():
            if hasattr(module, "gate") or hasattr(module, "router"):
                moe_blocks.append((name, module))
    # If none found via attributes, check root model itself
    if not moe_blocks and (hasattr(model, "gate") or hasattr(model, "router")):
        moe_blocks.append(("root", model))
    return moe_blocks


def _get_expert_modules(moe_block: nn.Module) -> list[nn.Module]:
    """Extract list of expert modules from an MoE block."""
    if hasattr(moe_block, "experts"):
        if isinstance(moe_block.experts, (nn.ModuleList, list)):
            return list(moe_block.experts)
    # Check numbered attributes (expert_0, expert_1, ...)
    experts: list[nn.Module] = []
    idx = 0
    while hasattr(moe_block, f"expert_{idx}"):
        experts.append(getattr(moe_block, f"expert_{idx}"))
        idx += 1
    return experts


def _collect_gate_activations(
    model: nn.Module,
    moe_blocks: list[tuple[str, nn.Module]],
    calibration_data: Any | None,
    tokenizer: Any | None = None,
    max_samples: int = 128,
) -> dict[str, torch.Tensor]:
    """Collect real gate input hidden states via forward hooks over calibration tokens."""
    if calibration_data is None:
        return {}

    gate_inputs: dict[str, list[torch.Tensor]] = {}
    hooks = []

    for layer_name, block in moe_blocks:
        gate_layer = getattr(block, "gate", getattr(block, "router", None))
        if gate_layer is not None and isinstance(gate_layer, nn.Linear):
            gate_inputs[layer_name] = []

            def make_hook(name: str):
                def hook_fn(_mod: nn.Module, inp: tuple[Any, ...], _out: Any) -> None:
                    if inp and isinstance(inp[0], torch.Tensor):
                        x = inp[0].detach()
                        flat_x = x.view(-1, x.shape[-1])
                        if flat_x.shape[0] > max_samples:
                            flat_x = flat_x[:max_samples]
                        gate_inputs[name].append(flat_x.cpu())

                return hook_fn

            hooks.append(gate_layer.register_forward_hook(make_hook(layer_name)))

    if not hooks:
        return {}

    model.eval()
    device = next(model.parameters()).device

    try:
        with torch.no_grad():
            samples: list[str] = []
            if isinstance(calibration_data, list):
                for item in calibration_data[:16]:
                    if isinstance(item, str):
                        samples.append(item)
                    elif isinstance(item, dict):
                        samples.append(item.get("text") or item.get("prompt") or "")
            elif hasattr(calibration_data, "texts"):
                samples = calibration_data.texts[:16]

            for sample in samples:
                if tokenizer is not None and sample:
                    encoded = tokenizer(
                        sample, return_tensors="pt", truncation=True, max_length=512
                    )
                    input_ids = encoded["input_ids"].to(device)
                    model(input_ids)
                elif isinstance(calibration_data, torch.Tensor):
                    model(calibration_data.to(device))
                    break
    except Exception as exc:
        logger.warning(
            f"Error during MoE calibration forward pass ({exc}); continuing with captured gate activations."
        )
    finally:
        for hook in hooks:
            hook.remove()

    final_inputs: dict[str, torch.Tensor] = {}
    for name, tensor_list in gate_inputs.items():
        if tensor_list:
            cat_t = torch.cat(tensor_list, dim=0)
            if cat_t.shape[0] > max_samples:
                cat_t = cat_t[:max_samples]
            final_inputs[name] = cat_t

    return final_inputs


class ExpertProfiler(CompressionMethod):
    """Profiles token routing traffic, weight norms, and activation magnitudes across MoE experts."""

    def __init__(
        self,
        calibration_dataset: str = "bigcode/starcoderdata",
        n_samples: int = 256,
        output: str = "expert_stats.json",
        **kwargs: Any,
    ) -> None:
        self.calibration_dataset = calibration_dataset
        self.n_samples = n_samples
        self.output_filename = output
        self.extra_kwargs = kwargs

    @property
    def name(self) -> str:
        return "expert_profile"

    def get_capabilities(self) -> PluginCapability:
        return PluginCapability(
            supported_architectures={
                ComputeArchitecture.MOE,
                ComputeArchitecture.HYBRID_ATTENTION,
                ComputeArchitecture.DENSE,
            },
            supported_dtypes={SupportedDtype.FP16, SupportedDtype.BF16, SupportedDtype.FP32},
            supports_moe=True,
            requires_calibration=True,
            supported_runtimes={"vllm", "sglang", "hf"},
        )

    def validate_applicability(self, model_metadata: ModelMetadata) -> None:
        caps = self.get_capabilities()
        if model_metadata.architecture_type not in caps.supported_architectures:
            raise CompressionPipelineError(
                f"Model architecture '{model_metadata.architecture_type}' is not supported for MoE profiling."
            )

    def profile_model(
        self,
        model: nn.Module,
        calibration_data: Any | None = None,
        tokenizer: Any | None = None,
    ) -> dict[str, Any]:
        """Collect per-layer and per-expert utilization and importance statistics."""
        start_time = time.perf_counter()
        moe_blocks = _find_moe_blocks(model)

        stats: dict[str, Any] = {
            "num_moe_layers": len(moe_blocks),
            "layers": {},
            "timestamp": time.time(),
        }

        # Collect empirical gate activations via forward hooks on calibration data
        gate_inputs_map = _collect_gate_activations(
            model=model,
            moe_blocks=moe_blocks,
            calibration_data=calibration_data,
            tokenizer=tokenizer,
            max_samples=self.n_samples,
        )

        with torch.no_grad():
            for layer_name, block in moe_blocks:
                experts = _get_expert_modules(block)
                num_experts = len(experts)
                if num_experts == 0:
                    continue

                # 1. Weight magnitude per expert
                magnitudes: list[float] = []
                for exp in experts:
                    l2 = torch.sqrt(
                        sum(torch.sum(p.data.float() ** 2) for p in exp.parameters())
                    ).item()
                    magnitudes.append(float(l2))

                # 2. Router frequency and co-activation matrix
                gate_layer = getattr(block, "gate", getattr(block, "router", None))
                co_activation_mat = torch.eye(num_experts)

                if (
                    layer_name in gate_inputs_map
                    and gate_layer is not None
                    and isinstance(gate_layer, nn.Linear)
                ):
                    calib_x = gate_inputs_map[layer_name].to(gate_layer.weight.device)
                    logits = gate_layer(calib_x.float())
                    k = min(2, num_experts)
                    top_k_indices = torch.topk(logits, k=k, dim=-1).indices

                    all_selected = top_k_indices.view(-1)
                    counts = torch.bincount(all_selected, minlength=num_experts).float()
                    frequencies = (counts / max(counts.sum().item(), 1.0)).tolist()

                    co_act = torch.zeros(num_experts, num_experts, device=logits.device)
                    for row in top_k_indices:
                        for e1 in row:
                            for e2 in row:
                                co_act[e1, e2] += 1.0

                    diag = torch.diag(co_act)
                    norm_denom = torch.sqrt(torch.outer(diag, diag)).clamp(min=1.0)
                    co_activation_mat = (co_act / norm_denom).cpu()
                else:
                    frequencies = [1.0 / num_experts] * num_experts
                    co_activation_mat = torch.eye(num_experts)

                # 3. Activation magnitude
                activations = [float(f * m) for f, m in zip(frequencies, magnitudes, strict=True)]

                # 4. Combined importance score
                max_f = max(max(frequencies), 1e-6)
                max_m = max(max(magnitudes), 1e-6)
                max_a = max(max(activations), 1e-6)

                importance_scores: list[float] = []
                expert_details: list[dict[str, Any]] = []
                for idx in range(num_experts):
                    norm_f = frequencies[idx] / max_f
                    norm_m = magnitudes[idx] / max_m
                    norm_a = activations[idx] / max_a
                    score = 0.5 * norm_f + 0.3 * norm_m + 0.2 * norm_a
                    importance_scores.append(float(score))
                    expert_details.append(
                        {
                            "expert_index": idx,
                            "frequency": float(frequencies[idx]),
                            "weight_magnitude": float(magnitudes[idx]),
                            "activation_magnitude": float(activations[idx]),
                            "importance_score": float(score),
                        }
                    )

                stats["layers"][layer_name] = {
                    "num_experts": num_experts,
                    "experts": expert_details,
                    "importance_ranking": sorted(
                        range(num_experts), key=lambda i: importance_scores[i], reverse=True
                    ),
                    "co_activation_matrix": co_activation_mat.tolist(),
                }

        stats["profiling_duration_sec"] = time.perf_counter() - start_time
        return stats

    def compress(
        self,
        model: nn.Module,
        tokenizer: Any,
        calibration_data: Any | None = None,
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> CompressionArtifact:
        out = Path(output_dir or "./expert_profiling_out")
        out.mkdir(parents=True, exist_ok=True)

        logger.info("Executing MoE Expert Profiler stage...")
        stats = self.profile_model(
            model=model,
            calibration_data=calibration_data,
            tokenizer=tokenizer,
        )

        stats_path = out / self.output_filename
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        if hasattr(model, "save_pretrained"):
            model.save_pretrained(out)
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(out)

        total_bytes = sum(f.stat().st_size for f in out.glob("**/*") if f.is_file())

        return CompressionArtifact(
            output_path=out,
            format="json",
            compressed_size_bytes=total_bytes,
            applied_methods=[self.name],
            metadata={
                "stats_file": str(stats_path),
                "num_moe_layers": stats["num_moe_layers"],
                "profiling_duration_sec": stats["profiling_duration_sec"],
            },
        )


CompressionRegistry.register("expert_profile", ExpertProfiler)
CompressionRegistry.register("expert_profiler", ExpertProfiler)
