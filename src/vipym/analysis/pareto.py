"""Multi-objective Pareto Frontier calculation engine."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import pydantic


class ParetoPoint(pydantic.BaseModel):
    """Data point in multi-dimensional objective space."""
    experiment_id: str
    configuration_name: str
    quality_score: float  # maximize
    latency_p50_ms: float  # minimize
    peak_vram_gb: float  # minimize
    cost_usd: float  # minimize
    compression_ratio: float  # maximize
    is_pareto_optimal: bool = False


class ParetoFrontierOptimizer:
    """Computes non-dominated Pareto sets across arbitrary continuous dimensions."""

    def __init__(self, maximize_dimensions: Optional[List[str]] = None, minimize_dimensions: Optional[List[str]] = None) -> None:
        self.maximize_dims = maximize_dimensions or ["quality_score", "compression_ratio"]
        self.minimize_dims = minimize_dimensions or ["latency_p50_ms", "peak_vram_gb", "cost_usd"]

    def compute_pareto_frontier(self, points: List[ParetoPoint]) -> List[ParetoPoint]:
        """Identify all non-dominated points in the population."""
        if not points:
            return []

        # Convert to matrix for vectorized comparison
        n = len(points)
        for i in range(n):
            points[i].is_pareto_optimal = True

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Check if point j strictly dominates point i
                j_dominates_i = True
                
                # Check maximize dimensions (j must be >= i, and at least one strictly >)
                for dim in self.maximize_dims:
                    val_i = getattr(points[i], dim)
                    val_j = getattr(points[j], dim)
                    if val_j < val_i:
                        j_dominates_i = False
                        break

                if not j_dominates_i:
                    continue

                # Check minimize dimensions (j must be <= i)
                for dim in self.minimize_dims:
                    val_i = getattr(points[i], dim)
                    val_j = getattr(points[j], dim)
                    if val_j > val_i:
                        j_dominates_i = False
                        break

                if j_dominates_i:
                    # Verify at least one strictly better
                    strictly_better = any(
                        getattr(points[j], dim) > getattr(points[i], dim) for dim in self.maximize_dims
                    ) or any(
                        getattr(points[j], dim) < getattr(points[i], dim) for dim in self.minimize_dims
                    )
                    if strictly_better:
                        points[i].is_pareto_optimal = False
                        break

        return [p for p in points if p.is_pareto_optimal]
