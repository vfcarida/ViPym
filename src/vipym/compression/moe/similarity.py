"""Expert similarity computation and merging strategies for MoE models."""

import torch
import torch.nn as nn

from vipym.core.logger import get_logger

logger = get_logger(__name__)


def _flatten_module_params(module: nn.Module) -> torch.Tensor:
    """Flatten all parameter tensors in an expert module into a single 1D tensor."""
    tensors = [p.detach().flatten().float() for p in module.parameters()]
    if not tensors:
        return torch.zeros(1)
    return torch.cat(tensors)


def compute_expert_similarity(
    experts: list[nn.Module],
    router_weight: torch.Tensor | None = None,
    activation_stats: list[torch.Tensor] | None = None,
    weight_factor: float = 0.6,
    router_factor: float = 0.2,
    activation_factor: float = 0.2,
) -> torch.Tensor:
    """Compute pairwise similarity matrix across MoE experts in a layer.

    Combines:
    1. Weight cosine similarity (module parameters)
    2. Router direction cosine similarity (gating weights per expert)
    3. Activation pattern similarity (on calibration tokens)
    """
    num_experts = len(experts)
    device = next(experts[0].parameters()).device if num_experts > 0 else torch.device("cpu")
    sim_matrix = torch.eye(num_experts, device=device)

    if num_experts <= 1:
        return sim_matrix

    # 1. Weight cosine similarity
    flattened_weights = torch.stack([_flatten_module_params(e) for e in experts])  # [E, P]
    norm_w = torch.nn.functional.normalize(flattened_weights, p=2, dim=-1)
    weight_sim = torch.mm(norm_w, norm_w.t()).clamp(-1.0, 1.0)

    # 2. Router similarity
    router_sim = torch.eye(num_experts, device=device)
    if router_weight is not None and router_weight.shape[0] >= num_experts:
        r_slice = router_weight[:num_experts].float()
        norm_r = torch.nn.functional.normalize(r_slice, p=2, dim=-1)
        router_sim = torch.mm(norm_r, norm_r.t()).clamp(-1.0, 1.0)

    # 3. Activation pattern similarity
    act_sim = torch.eye(num_experts, device=device)
    if activation_stats is not None and len(activation_stats) == num_experts:
        act_vectors = torch.stack([a.flatten().float().to(device) for a in activation_stats])
        norm_a = torch.nn.functional.normalize(act_vectors, p=2, dim=-1)
        act_sim = torch.mm(norm_a, norm_a.t()).clamp(-1.0, 1.0)

    # Combined similarity
    combined_sim = (
        weight_factor * weight_sim + router_factor * router_sim + activation_factor * act_sim
    )
    # Ensure diagonals are 1.0
    combined_sim.fill_diagonal_(1.0)
    return combined_sim


def cluster_experts_by_similarity(
    similarity_matrix: torch.Tensor,
    threshold: float = 0.85,
    min_clusters: int = 1,
) -> list[list[int]]:
    """Greedy agglomerative clustering of experts with similarity >= threshold."""
    num_experts = similarity_matrix.shape[0]
    sim = similarity_matrix.clone().cpu()
    sim.fill_diagonal_(-1.0)

    # Initialize each expert in its own cluster
    clusters: list[list[int]] = [[i] for i in range(num_experts)]

    while len(clusters) > min_clusters:
        # Find highest similarity between existing clusters
        best_sim = -1.0
        best_pair = (-1, -1)

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Average linkage between cluster i and cluster j
                sub_sim = sim[clusters[i]][:, clusters[j]].mean().item()
                if sub_sim > best_sim:
                    best_sim = sub_sim
                    best_pair = (i, j)

        if best_sim >= threshold and best_pair != (-1, -1):
            i, j = best_pair
            clusters[i].extend(clusters[j])
            clusters.pop(j)
        else:
            break

    return clusters


def _slerp_tensor(
    v0: torch.Tensor, v1: torch.Tensor, t: float = 0.5, eps: float = 1e-7
) -> torch.Tensor:
    """Spherical linear interpolation between two parameter tensors."""
    orig_shape = v0.shape
    v0_flat = v0.flatten().float()
    v1_flat = v1.flatten().float()

    norm_v0 = torch.norm(v0_flat)
    norm_v1 = torch.norm(v1_flat)

    if norm_v0 < eps or norm_v1 < eps:
        return ((1.0 - t) * v0 + t * v1).to(v0.dtype)

    u0 = v0_flat / norm_v0
    u1 = v1_flat / norm_v1

    dot = torch.clamp(torch.dot(u0, u1), -1.0 + eps, 1.0 - eps)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    if sin_omega.abs() < eps:
        res = (1.0 - t) * v0_flat + t * v1_flat
    else:
        res = (torch.sin((1.0 - t) * omega) / sin_omega) * v0_flat + (
            torch.sin(t * omega) / sin_omega
        ) * v1_flat

    # Interpolate magnitude as well
    res_norm = (1.0 - t) * norm_v0 + t * norm_v1
    res = (res / (torch.norm(res) + eps)) * res_norm
    return res.view(orig_shape).to(v0.dtype)


def merge_experts(
    expert_list: list[nn.Module],
    cluster_indices: list[int],
    strategy: str = "frequency_weighted",
    frequencies: list[float] | None = None,
) -> nn.Module:
    """Merge a cluster of expert modules into a single super-expert module.

    Supported strategies:
    - 'average': arithmetic mean of parameters
    - 'slerp': spherical linear interpolation
    - 'frequency_weighted': weighted average based on calibration token traffic
    """
    import copy

    if len(cluster_indices) == 1:
        return copy.deepcopy(expert_list[cluster_indices[0]])

    target_expert = copy.deepcopy(expert_list[cluster_indices[0]])
    source_experts = [expert_list[idx] for idx in cluster_indices]

    if strategy.lower() == "slerp" and len(source_experts) == 2:
        with torch.no_grad():
            for p_target, p_0, p_1 in zip(
                target_expert.parameters(),
                source_experts[0].parameters(),
                source_experts[1].parameters(),
                strict=True,
            ):
                p_target.copy_(_slerp_tensor(p_0.data, p_1.data, t=0.5))
    elif strategy.lower() == "frequency_weighted" and frequencies is not None:
        cluster_freqs = [max(1e-4, frequencies[idx]) for idx in cluster_indices]
        total_freq = sum(cluster_freqs)
        weights = [f / total_freq for f in cluster_freqs]

        with torch.no_grad():
            for p_target in target_expert.parameters():
                p_target.zero_()

            for exp, w in zip(source_experts, weights, strict=True):
                for p_target, p_src in zip(
                    target_expert.parameters(), exp.parameters(), strict=True
                ):
                    p_target.add_(p_src.data.float() * w)
    else:
        # Simple arithmetic average
        weight = 1.0 / len(source_experts)
        with torch.no_grad():
            for p_target in target_expert.parameters():
                p_target.zero_()

            for exp in source_experts:
                for p_target, p_src in zip(
                    target_expert.parameters(), exp.parameters(), strict=True
                ):
                    p_target.add_(p_src.data.float() * weight)

    return target_expert
