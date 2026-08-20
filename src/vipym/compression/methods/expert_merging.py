"""MoE Expert Merging Compression Method."""

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from vipym.compression.methods.expert_profiler import _find_moe_blocks, _get_expert_modules
from vipym.compression.moe.router_utils import retrain_router
from vipym.compression.moe.similarity import (
    cluster_experts_by_similarity,
    compute_expert_similarity,
    merge_experts,
)
from vipym.compression.registry import CompressionRegistry
from vipym.core.constants import ComputeArchitecture, SupportedDtype
from vipym.core.exceptions import CompressionPipelineError
from vipym.core.logger import get_logger
from vipym.interfaces.compression import CompressionArtifact, CompressionMethod
from vipym.interfaces.model import ModelMetadata, PluginCapability

logger = get_logger(__name__)


class ExpertMergingMethod(CompressionMethod):
    """Combines functionally similar MoE experts via similarity clustering and parameter merging."""

    def __init__(
        self,
        similarity_threshold: float = 0.80,
        merge_strategy: str = "frequency_weighted",
        min_experts_per_layer: int = 1,
        target_num_experts: int | None = None,
        retrain_router: bool = True,
        router_lr: float = 1e-4,
        router_steps: int = 200,
        **kwargs: Any,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.merge_strategy = merge_strategy.lower()
        self.min_experts_per_layer = min_experts_per_layer
        self.target_num_experts = target_num_experts
        self.should_retrain_router = retrain_router
        self.router_lr = router_lr
        self.router_steps = router_steps
        self.extra_kwargs = kwargs

    @property
    def name(self) -> str:
        return f"expert_merge_{self.merge_strategy}_th{int(self.similarity_threshold * 100)}"

    def get_capabilities(self) -> PluginCapability:
        return PluginCapability(
            supported_architectures={
                ComputeArchitecture.MOE,
                ComputeArchitecture.HYBRID_ATTENTION,
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
                f"Model architecture '{model_metadata.architecture_type}' is not supported for MoE expert merging."
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
        out = Path(output_dir or "./expert_merged_model")
        out.mkdir(parents=True, exist_ok=True)

        sim_th = float(kwargs.get("similarity_threshold", self.similarity_threshold))
        strategy = str(kwargs.get("merge_strategy", self.merge_strategy)).lower()
        min_exp = int(kwargs.get("min_experts_per_layer", self.min_experts_per_layer))
        retrain = bool(kwargs.get("retrain_router", self.should_retrain_router))
        r_steps = int(kwargs.get("router_steps", self.router_steps))
        r_lr = float(kwargs.get("router_lr", self.router_lr))

        orig_params = sum(p.numel() for p in model.parameters())
        moe_blocks = _find_moe_blocks(model)
        logger.info(
            f"Applying MoE Expert Merging: {len(moe_blocks)} blocks, strategy={strategy}, threshold={sim_th}"
        )

        layer_merge_reports: dict[str, Any] = {}

        for layer_name, block in moe_blocks:
            experts = _get_expert_modules(block)
            num_experts = len(experts)
            if num_experts <= 1:
                continue

            gate_layer = getattr(block, "gate", getattr(block, "router", None))
            gate_weight = gate_layer.weight.data if (gate_layer is not None and isinstance(gate_layer, nn.Linear)) else None

            # Compute similarity matrix across experts in this layer
            sim_matrix = compute_expert_similarity(
                experts=experts,
                router_weight=gate_weight,
            )

            # Determine cluster target
            clusters = cluster_experts_by_similarity(
                similarity_matrix=sim_matrix,
                threshold=sim_th,
                min_clusters=min_exp,
            )

            if len(clusters) == num_experts and self.target_num_experts is not None and self.target_num_experts < num_experts:
                # Force merge to target_num_experts
                clusters = cluster_experts_by_similarity(
                    similarity_matrix=sim_matrix,
                    threshold=-1.0,  # force clustering by closest pairs
                    min_clusters=self.target_num_experts,
                )

            # Build merged expert list
            merged_expert_modules: list[nn.Module] = []
            for cluster in clusters:
                merged_exp = merge_experts(
                    expert_list=experts,
                    cluster_indices=cluster,
                    strategy=strategy,
                )
                merged_expert_modules.append(merged_exp)

            # Update block topology
            if hasattr(block, "experts") and isinstance(block.experts, nn.ModuleList):
                block.experts = nn.ModuleList(merged_expert_modules)
            else:
                # Clean old attributes and set new merged experts
                for idx in range(num_experts):
                    if hasattr(block, f"expert_{idx}"):
                        delattr(block, f"expert_{idx}")
                for idx, exp_mod in enumerate(merged_expert_modules):
                    setattr(block, f"expert_{idx}", exp_mod)

            # Update router weights to match new merged expert count
            if gate_layer is not None and isinstance(gate_layer, nn.Linear):
                new_num_experts = len(clusters)
                new_weight = torch.zeros(new_num_experts, gate_layer.in_features, device=gate_layer.weight.device)
                new_bias = torch.zeros(new_num_experts, device=gate_layer.weight.device) if gate_layer.bias is not None else None

                with torch.no_grad():
                    for new_idx, cluster in enumerate(clusters):
                        # Average router projection for merged cluster
                        new_weight[new_idx] = gate_layer.weight.data[cluster].mean(dim=0)
                        if new_bias is not None and gate_layer.bias is not None:
                            new_bias[new_idx] = gate_layer.bias.data[cluster].mean()

                    gate_layer.out_features = new_num_experts
                    gate_layer.weight = nn.Parameter(new_weight)
                    if new_bias is not None:
                        gate_layer.bias = nn.Parameter(new_bias)

                if retrain:
                    calib_tokens = torch.randn(128, gate_layer.in_features)
                    retrain_router(
                        router_layer=gate_layer,
                        calibration_hidden_states=calib_tokens,
                        retained_expert_indices=list(range(new_num_experts)),
                        steps=r_steps,
                        lr=r_lr,
                    )

            layer_merge_reports[layer_name] = {
                "num_experts_before": num_experts,
                "num_experts_after": len(clusters),
                "clusters": clusters,
                "merge_strategy": strategy,
            }

        # Measure parameter reduction
        new_params = sum(p.numel() for p in model.parameters())
        reduction_factor = orig_params / max(1, new_params)

        # Save model and tokenizer
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(out)
        elif hasattr(model, "state_dict"):
            torch.save(model.state_dict(), out / "pytorch_model.bin")

        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(out)

        report_file = out / "expert_merging_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "strategy": strategy,
                    "original_params": orig_params,
                    "merged_params": new_params,
                    "param_reduction_factor": reduction_factor,
                    "layers": layer_merge_reports,
                },
                f,
                indent=2,
            )

        total_bytes = sum(f.stat().st_size for f in out.glob("**/*") if f.is_file())
        logger.info(
            f"Expert merging finished: params {orig_params:,} -> {new_params:,} "
            f"({(1.0 - new_params / orig_params) * 100:.1f}% reduction)"
        )

        return CompressionArtifact(
            output_path=out,
            format="safetensors",
            compressed_size_bytes=total_bytes,
            applied_methods=[self.name],
            metadata={
                "strategy": strategy,
                "similarity_threshold": sim_th,
                "original_params": orig_params,
                "merged_params": new_params,
                "param_reduction_factor": float(reduction_factor),
                "layers": layer_merge_reports,
            },
        )


CompressionRegistry.register("expert_merge", ExpertMergingMethod)
CompressionRegistry.register("moe_merge", ExpertMergingMethod)
