"""Multi-objective Pareto Frontier and Non-Dominated Sorting Engine.

Computes multi-dimensional Pareto frontiers across:
- Quality (SE composite score, maximize)
- Cost ($/1M tokens, minimize)
- Latency (latency_p50_ms, minimize)
- Throughput (tokens/sec, maximize)
- VRAM (peak_vram_gb, minimize)
- Compression Ratio (maximize)
"""

from __future__ import annotations

import math
from dataclasses import field
from typing import Any

import pydantic


class ParetoPoint(pydantic.BaseModel):
    """Data point in multi-dimensional objective space representing a compressed model variant."""

    experiment_id: str
    configuration_name: str
    compression_method: str = "none"
    quality_score: float = 0.0  # maximize (e.g. SE composite score)
    relative_quality_score: float | None = None  # % relative to uncompressed teacher (0.0 - 1.0)
    cost_per_1m_tokens: float = 0.0  # minimize ($ / 1M tokens)
    cost_usd: float = 0.0  # minimize (legacy alias)
    latency_p50_ms: float = 0.0  # minimize (p50 latency in ms)
    latency_p95_ms: float = 0.0  # minimize (p95 latency in ms)
    throughput_tok_s: float = 0.0  # maximize (tokens/sec)
    peak_vram_gb: float = 0.0  # minimize (GPU VRAM in GB)
    compression_ratio: float = 1.0  # maximize (e.g. 4.0x)
    hardware_recommendation: str = ""  # e.g. "2x H100", "1x A10G"
    suite_scores: dict[str, float] = pydantic.Field(default_factory=dict)
    is_pareto_optimal: bool = False
    pareto_rank: int = 1  # 1 = Front 1 (best), 2 = Front 2, etc.
    crowding_distance: float = 0.0
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)

    @pydantic.model_validator(mode="before")
    @classmethod
    def sync_cost_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "cost_usd" in data and "cost_per_1m_tokens" not in data:
                data["cost_per_1m_tokens"] = data["cost_usd"]
            elif "cost_per_1m_tokens" in data and "cost_usd" not in data:
                data["cost_usd"] = data["cost_per_1m_tokens"]
        return data


class ParetoFrontierOptimizer:
    """Computes non-dominated Pareto sets across arbitrary continuous dimensions."""

    def __init__(
        self,
        maximize_dimensions: list[str] | None = None,
        minimize_dimensions: list[str] | None = None,
    ) -> None:
        self.maximize_dims = maximize_dimensions or ["quality_score", "throughput_tok_s", "compression_ratio"]
        self.minimize_dims = minimize_dimensions or ["cost_per_1m_tokens", "latency_p50_ms", "peak_vram_gb"]

    def dominates(self, p1: ParetoPoint, p2: ParetoPoint) -> bool:
        """Return True if p1 dominates p2 (p1 is at least as good in all dims and strictly better in at least one)."""
        at_least_as_good = True
        strictly_better = False

        # Maximize dimensions
        for dim in self.maximize_dims:
            v1 = getattr(p1, dim, 0.0)
            v2 = getattr(p2, dim, 0.0)
            if v1 < v2:
                return False
            if v1 > v2:
                strictly_better = True

        # Minimize dimensions
        for dim in self.minimize_dims:
            v1 = getattr(p1, dim, 0.0)
            v2 = getattr(p2, dim, 0.0)
            if v1 > v2:
                return False
            if v1 < v2:
                strictly_better = True

        return at_least_as_good and strictly_better

    def compute_pareto_frontier(
        self,
        points: list[ParetoPoint],
        min_quality: float | None = None,
        max_cost: float | None = None,
        max_latency_ms: float | None = None,
        max_vram_gb: float | None = None,
    ) -> list[ParetoPoint]:
        """Identify all non-dominated points in the population with optional constraint filtering."""
        if not points:
            return []

        # 1. Apply hard constraint filters if specified
        valid_points = self.filter_by_constraints(
            points,
            min_quality=min_quality,
            max_cost=max_cost,
            max_latency_ms=max_latency_ms,
            max_vram_gb=max_vram_gb,
        )
        if not valid_points:
            return []

        # 2. Reset Pareto flags
        for p in valid_points:
            p.is_pareto_optimal = True

        # 3. Check pairwise dominance
        n = len(valid_points)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self.dominates(valid_points[j], valid_points[i]):
                    valid_points[i].is_pareto_optimal = False
                    break

        return [p for p in valid_points if p.is_pareto_optimal]

    def fast_non_dominated_sort(self, points: list[ParetoPoint]) -> list[list[ParetoPoint]]:
        """Perform fast non-dominated sorting (NSGA-II) to partition points into Pareto fronts."""
        if not points:
            return []

        fronts: list[list[ParetoPoint]] = [[]]
        domination_count: dict[int, int] = {i: 0 for i in range(len(points))}
        dominated_indices: dict[int, list[int]] = {i: [] for i in range(len(points))}

        for p_idx, p in enumerate(points):
            for q_idx, q in enumerate(points):
                if p_idx == q_idx:
                    continue
                if self.dominates(p, q):
                    dominated_indices[p_idx].append(q_idx)
                elif self.dominates(q, p):
                    domination_count[p_idx] += 1

            if domination_count[p_idx] == 0:
                p.pareto_rank = 1
                p.is_pareto_optimal = True
                fronts[0].append(p)
            else:
                p.is_pareto_optimal = False

        i = 0
        while i < len(fronts) and fronts[i]:
            next_front: list[ParetoPoint] = []
            for p in fronts[i]:
                p_idx = points.index(p)
                for q_idx in dominated_indices[p_idx]:
                    domination_count[q_idx] -= 1
                    if domination_count[q_idx] == 0:
                        q = points[q_idx]
                        q.pareto_rank = i + 2
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)

        return [f for f in fronts if f]

    def filter_by_constraints(
        self,
        points: list[ParetoPoint],
        min_quality: float | None = None,
        max_cost: float | None = None,
        max_latency_ms: float | None = None,
        max_vram_gb: float | None = None,
    ) -> list[ParetoPoint]:
        """Filter points that satisfy all hard operational thresholds."""
        filtered = []
        for p in points:
            if min_quality is not None and p.quality_score < min_quality:
                continue
            if max_cost is not None and p.cost_per_1m_tokens > max_cost:
                continue
            if max_latency_ms is not None and p.latency_p50_ms > max_latency_ms:
                continue
            if max_vram_gb is not None and p.peak_vram_gb > max_vram_gb:
                continue
            filtered.append(p)
        return filtered

    def find_closest_to_utopia(self, points: list[ParetoPoint]) -> ParetoPoint | None:
        """Find the point with minimum normalized Euclidean distance to the ideal Utopia point."""
        if not points:
            return None
        if len(points) == 1:
            return points[0]

        # Determine Utopia (ideal) bounds across dimensions
        max_q = max(p.quality_score for p in points)
        min_q = min(p.quality_score for p in points)
        q_range = max_q - min_q if max_q > min_q else 1.0

        min_c = min(p.cost_per_1m_tokens for p in points)
        max_c = max(p.cost_per_1m_tokens for p in points)
        c_range = max_c - min_c if max_c > min_c else 1.0

        min_lat = min(p.latency_p50_ms for p in points)
        max_lat = max(p.latency_p50_ms for p in points)
        lat_range = max_lat - min_lat if max_lat > min_lat else 1.0

        best_point = points[0]
        best_dist = float("inf")

        for p in points:
            # Normalized distance to (1.0 quality, 0.0 cost, 0.0 latency)
            norm_q_dist = (max_q - p.quality_score) / q_range
            norm_c_dist = (p.cost_per_1m_tokens - min_c) / c_range
            norm_lat_dist = (p.latency_p50_ms - min_lat) / lat_range

            dist = math.sqrt(norm_q_dist**2 + norm_c_dist**2 + norm_lat_dist**2)
            if dist < best_dist:
                best_dist = dist
                best_point = p

        return best_point
