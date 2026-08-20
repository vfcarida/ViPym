"""MoE compression utilities package (profiling, similarity, merging, router retraining)."""

from vipym.compression.moe.router_distillation import (
    RouterDistillationConfig,
    RouterDistillationResult,
    distil_router,
    run_router_distillation,
)
from vipym.compression.moe.router_utils import retrain_router
from vipym.compression.moe.similarity import (
    cluster_experts_by_similarity,
    compute_expert_similarity,
    merge_experts,
)

__all__ = [
    "RouterDistillationConfig",
    "RouterDistillationResult",
    "cluster_experts_by_similarity",
    "compute_expert_similarity",
    "distil_router",
    "merge_experts",
    "retrain_router",
    "run_router_distillation",
]
