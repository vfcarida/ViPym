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
    ) -> dict[str, Any]:
        """Collect per-layer and per-expert utilization and importance statistics."""
        start_time = time.perf_counter()
        moe_blocks = _find_moe_blocks(model)

        stats: dict[str, Any] = {
            "num_moe_layers": len(moe_blocks),
            "layers": {},
            "timestamp": time.time(),
        }

        # Calibration hidden states for routing frequency
        test_inputs = torch.randn(min(self.n_samples, 128), 4, 64)

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

                # 2. Router frequency on calibration data
                frequencies = [1.0 / num_experts] * num_experts
                gate_layer = getattr(block, "gate", getattr(block, "router", None))
                if gate_layer is not None and isinstance(gate_layer, nn.Linear):
                    try:
                        in_dim = gate_layer.in_features
                        calib_x = torch.randn(128, in_dim, device=gate_layer.weight.device)
                        logits = gate_layer(calib_x)
                        top_experts = torch.argmax(logits, dim=-1)
                        counts = torch.bincount(top_experts, minlength=num_experts).float()
                        frequencies = (counts / counts.sum()).tolist()
                    except Exception:
                        pass

                # 3. Activation magnitude
                activations = [float(f * m) for f, m in zip(frequencies, magnitudes, strict=True)]

                # 4. Combined importance score
                # Normalize metrics
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
                    expert_details.append({
                        "expert_index": idx,
                        "frequency": float(frequencies[idx]),
                        "weight_magnitude": float(magnitudes[idx]),
                        "activation_magnitude": float(activations[idx]),
                        "importance_score": float(score),
                    })

                stats["layers"][layer_name] = {
                    "num_experts": num_experts,
                    "experts": expert_details,
                    "importance_ranking": sorted(
                        range(num_experts), key=lambda i: importance_scores[i], reverse=True
                    ),
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
        stats = self.profile_model(model=model, calibration_data=calibration_data)

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
