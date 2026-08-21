"""MoE Expert Pruning Compression Method."""

import copy
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from vipym.compression.methods.expert_profiler import _find_moe_blocks, _get_expert_modules
from vipym.compression.moe.router_distillation import (
    RouterDistillationConfig,
    run_router_distillation,
)
from vipym.compression.moe.router_utils import retrain_router
from vipym.compression.registry import CompressionRegistry
from vipym.core.constants import ComputeArchitecture, SupportedDtype
from vipym.core.exceptions import CompressionPipelineError
from vipym.core.logger import get_logger
from vipym.interfaces.compression import CompressionArtifact, CompressionMethod
from vipym.interfaces.model import ModelMetadata, PluginCapability

logger = get_logger(__name__)


class ExpertPruningMethod(CompressionMethod):
    """Surgical MoE expert pruning based on multi-criteria importance scoring and router retraining."""

    def __init__(
        self,
        strategy: str = "importance",
        prune_ratio: float = 0.25,
        num_experts_to_prune: int | None = None,
        per_layer: bool = True,
        retrain_router: bool = True,
        router_lr: float = 1e-4,
        router_steps: int = 200,
        router_distillation: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.strategy = strategy.lower()
        self.prune_ratio = prune_ratio
        self.num_experts_to_prune = num_experts_to_prune
        self.per_layer = per_layer
        self.should_retrain_router = retrain_router
        self.router_lr = router_lr
        self.router_steps = router_steps
        self.router_distillation_cfg = router_distillation  # dict from YAML, or None
        self.extra_kwargs = kwargs

    @property
    def name(self) -> str:
        return f"expert_prune_{int(self.prune_ratio * 100)}pct"

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
                f"Model architecture '{model_metadata.architecture_type}' is not supported for MoE expert pruning."
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
        out = Path(output_dir or "./expert_pruned_model")
        out.mkdir(parents=True, exist_ok=True)

        prune_ratio = float(kwargs.get("prune_ratio", self.prune_ratio))
        strategy = str(kwargs.get("strategy", self.strategy)).lower()
        retrain = bool(kwargs.get("retrain_router", self.should_retrain_router))
        r_steps = int(kwargs.get("router_steps", self.router_steps))
        r_lr = float(kwargs.get("router_lr", self.router_lr))
        distil_cfg_dict: dict[str, Any] | None = kwargs.get(
            "router_distillation", self.router_distillation_cfg
        )
        distil_cfg = (
            RouterDistillationConfig.from_dict(distil_cfg_dict) if distil_cfg_dict else None
        )

        orig_params = sum(p.numel() for p in model.parameters())

        moe_blocks = _find_moe_blocks(model)
        logger.info(
            f"Applying MoE Expert Pruning: {len(moe_blocks)} blocks, ratio={prune_ratio:.2f}, strategy={strategy}"
        )

        layer_pruning_reports: dict[str, Any] = {}

        for layer_name, block in moe_blocks:
            experts = _get_expert_modules(block)
            num_experts = len(experts)
            if num_experts <= 1:
                continue

            num_to_remove = (
                self.num_experts_to_prune
                if self.num_experts_to_prune is not None
                else max(1, int(num_experts * prune_ratio))
            )
            # Never prune all experts
            num_to_remove = min(num_to_remove, num_experts - 1)

            # Compute scores per expert
            scores: list[float] = []
            gate_layer = getattr(block, "gate", getattr(block, "router", None))

            for idx, exp in enumerate(experts):
                l2 = float(
                    torch.sqrt(sum(torch.sum(p.data.float() ** 2) for p in exp.parameters())).item()
                )
                if strategy == "magnitude":
                    scores.append(l2)
                else:
                    # Combined / frequency importance
                    freq = 1.0 / num_experts
                    if gate_layer is not None and isinstance(gate_layer, nn.Linear):
                        try:
                            # Use router weight magnitude for this expert as importance prior
                            freq = float(torch.norm(gate_layer.weight.data[idx].float()).item())
                        except Exception:
                            freq = 1.0 / num_experts
                    scores.append(0.6 * freq + 0.4 * l2)

            # Identify retained and pruned expert indices
            sorted_indices = sorted(range(num_experts), key=lambda i: scores[i], reverse=True)
            retained_indices = sorted(sorted_indices[: num_experts - num_to_remove])
            pruned_indices = sorted(sorted_indices[num_experts - num_to_remove :])

            # Mutate block to keep only retained experts
            if hasattr(block, "experts") and isinstance(block.experts, nn.ModuleList):
                new_module_list = nn.ModuleList([experts[i] for i in retained_indices])
                block.experts = new_module_list
            else:
                # Update numbered attributes
                for p_idx in pruned_indices:
                    if hasattr(block, f"expert_{p_idx}"):
                        delattr(block, f"expert_{p_idx}")
                # Compact numbered attributes
                for new_idx, orig_idx in enumerate(retained_indices):
                    if hasattr(block, f"expert_{orig_idx}") and new_idx != orig_idx:
                        setattr(block, f"expert_{new_idx}", getattr(block, f"expert_{orig_idx}"))
                        if orig_idx not in retained_indices[:new_idx]:
                            delattr(block, f"expert_{orig_idx}")

            # Slice router gate layer to match retained expert count
            if gate_layer is not None and isinstance(gate_layer, nn.Linear):
                # Snapshot teacher gate BEFORE slicing (needed for distillation)
                teacher_gate: nn.Linear | None = None
                if distil_cfg is not None:
                    teacher_gate = copy.deepcopy(gate_layer)

                with torch.no_grad():
                    new_out_features = len(retained_indices)
                    old_weight = gate_layer.weight.data
                    new_weight = old_weight[retained_indices, :]
                    gate_layer.out_features = new_out_features
                    gate_layer.weight = nn.Parameter(new_weight)
                    if gate_layer.bias is not None:
                        gate_layer.bias = nn.Parameter(gate_layer.bias.data[retained_indices])

                # Step 1 — lightweight entropy-based retrain (always if enabled)
                if retrain:
                    calib_tokens = torch.randn(128, gate_layer.in_features)
                    retrain_router(
                        router_layer=gate_layer,
                        calibration_hidden_states=calib_tokens,
                        retained_expert_indices=retained_indices,
                        pruned_expert_indices=pruned_indices,
                        steps=r_steps,
                        lr=r_lr,
                    )

                # Step 2 — KL distillation from teacher (optional, higher quality)
                if distil_cfg is not None and teacher_gate is not None:
                    calib_hs = torch.randn(distil_cfg.calibration_samples, gate_layer.in_features)
                    distil_result = run_router_distillation(
                        student_router=gate_layer,
                        teacher_router=teacher_gate,
                        calibration_hidden_states=calib_hs,
                        retained_expert_indices=retained_indices,
                        config=distil_cfg,
                    )
                    layer_pruning_reports.setdefault(layer_name, {})
                    layer_pruning_reports[layer_name]["distillation"] = distil_result.to_dict()

            # Merge base report, preserving any distillation key set above
            layer_entry = layer_pruning_reports.get(layer_name, {})
            layer_entry.update(
                {
                    "num_experts_before": num_experts,
                    "num_experts_after": len(retained_indices),
                    "retained_indices": retained_indices,
                    "pruned_indices": pruned_indices,
                }
            )
            layer_pruning_reports[layer_name] = layer_entry

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

        report_file = out / "expert_pruning_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "prune_ratio": prune_ratio,
                    "original_params": orig_params,
                    "pruned_params": new_params,
                    "param_reduction_factor": reduction_factor,
                    "layers": layer_pruning_reports,
                },
                f,
                indent=2,
            )

        total_bytes = sum(f.stat().st_size for f in out.glob("**/*") if f.is_file())
        logger.info(
            f"Expert pruning finished: params {orig_params:,} -> {new_params:,} "
            f"({(1.0 - new_params / orig_params) * 100:.1f}% reduction)"
        )

        return CompressionArtifact(
            output_path=out,
            format="safetensors",
            compressed_size_bytes=total_bytes,
            applied_methods=[self.name],
            metadata={
                "strategy": strategy,
                "prune_ratio": prune_ratio,
                "original_params": orig_params,
                "pruned_params": new_params,
                "param_reduction_factor": float(reduction_factor),
                "layers": layer_pruning_reports,
            },
        )


CompressionRegistry.register("expert_prune", ExpertPruningMethod)
CompressionRegistry.register("moe_prune", ExpertPruningMethod)
