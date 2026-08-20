"""Pareto-Optimal Deployment Recommendation Engine.

Given compression candidates and operational constraints (quality threshold, budget, latency, hardware),
evaluates Pareto dominance and recommends ranked deployment configurations with formatted output tables.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from vipym.analysis.pareto import ParetoFrontierOptimizer, ParetoPoint
from vipym.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RecommendationReport:
    """Complete recommendation report with top choice, ranked candidates, and summary."""

    recommended_variant: ParetoPoint | None
    ranked_options: list[ParetoPoint]
    constraints_applied: dict[str, Any]
    strategy: str
    executive_summary: str
    ascii_table: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommended_variant": self.recommended_variant.model_dump() if self.recommended_variant else None,
            "ranked_options": [p.model_dump() for p in self.ranked_options],
            "constraints_applied": self.constraints_applied,
            "strategy": self.strategy,
            "executive_summary": self.executive_summary,
            "ascii_table": self.ascii_table,
        }


class DeploymentRecommender:
    """Selects and ranks Pareto-optimal compression configurations under user constraints."""

    def __init__(
        self,
        optimizer: ParetoFrontierOptimizer | None = None,
    ) -> None:
        self.optimizer = optimizer or ParetoFrontierOptimizer()

    def recommend(
        self,
        candidates: list[ParetoPoint],
        min_quality: float = 0.65,
        max_cost_per_1m: float | None = None,
        max_latency_p50_ms: float | None = None,
        max_vram_gb: float | None = None,
        strategy: str = "balanced",
    ) -> RecommendationReport:
        """Filter and rank candidate model configurations to determine optimal deployment."""
        if not candidates:
            return RecommendationReport(
                recommended_variant=None,
                ranked_options=[],
                constraints_applied={
                    "min_quality": min_quality,
                    "max_cost_per_1m": max_cost_per_1m,
                    "max_latency_p50_ms": max_latency_p50_ms,
                    "max_vram_gb": max_vram_gb,
                },
                strategy=strategy,
                executive_summary="No candidate models provided for recommendation.",
                ascii_table="No candidates available.",
            )

        # 1. Filter candidates matching hard constraints
        valid = self.optimizer.filter_by_constraints(
            candidates,
            min_quality=min_quality,
            max_cost=max_cost_per_1m,
            max_latency_ms=max_latency_p50_ms,
            max_vram_gb=max_vram_gb,
        )

        if not valid:
            return RecommendationReport(
                recommended_variant=None,
                ranked_options=[],
                constraints_applied={
                    "min_quality": min_quality,
                    "max_cost_per_1m": max_cost_per_1m,
                    "max_latency_p50_ms": max_latency_p50_ms,
                    "max_vram_gb": max_vram_gb,
                },
                strategy=strategy,
                executive_summary="No candidate models met the specified quality/budget constraints.",
                ascii_table="No candidates satisfied constraints.",
            )

        # 2. Compute Pareto optimality
        pareto_points = self.optimizer.compute_pareto_frontier(valid)
        pareto_names = {p.configuration_name for p in pareto_points}
        for p in valid:
            p.is_pareto_optimal = p.configuration_name in pareto_names

        # 3. Sort / rank according to strategy
        ranked = self._rank_candidates(valid, strategy)
        recommended = ranked[0] if ranked else None

        # 4. Generate executive summary
        summary = self._generate_summary(recommended, len(ranked), strategy, min_quality)

        # 5. Generate ASCII Table
        ascii_tbl = self._generate_ascii_table(
            ranked,
            min_quality=min_quality,
            max_cost=max_cost_per_1m,
        )

        return RecommendationReport(
            recommended_variant=recommended,
            ranked_options=ranked,
            constraints_applied={
                "min_quality": min_quality,
                "max_cost_per_1m": max_cost_per_1m,
                "max_latency_p50_ms": max_latency_p50_ms,
                "max_vram_gb": max_vram_gb,
            },
            strategy=strategy,
            executive_summary=summary,
            ascii_table=ascii_tbl,
        )

    def _rank_candidates(
        self,
        points: list[ParetoPoint],
        strategy: str,
    ) -> list[ParetoPoint]:
        strat = strategy.lower()

        if strat == "cost_first":
            # Lowest cost first, then highest quality
            return sorted(points, key=lambda p: (not p.is_pareto_optimal, p.cost_per_1m_tokens, -p.quality_score))
        elif strat == "quality_first":
            # Highest quality first, then lowest cost
            return sorted(points, key=lambda p: (not p.is_pareto_optimal, -p.quality_score, p.cost_per_1m_tokens))
        elif strat == "throughput_first":
            # Highest throughput first
            return sorted(points, key=lambda p: (not p.is_pareto_optimal, -p.throughput_tok_s, p.cost_per_1m_tokens))
        else:
            # Balanced: Utopia distance
            utopia_choice = self.optimizer.find_closest_to_utopia(points)
            if utopia_choice:
                remaining = [p for p in points if p.configuration_name != utopia_choice.configuration_name]
                remaining_sorted = sorted(
                    remaining,
                    key=lambda p: (not p.is_pareto_optimal, -p.quality_score, p.cost_per_1m_tokens),
                )
                return [utopia_choice] + remaining_sorted
            return sorted(points, key=lambda p: (not p.is_pareto_optimal, -p.quality_score, p.cost_per_1m_tokens))

    def _generate_summary(
        self,
        top_choice: ParetoPoint | None,
        total_valid: int,
        strategy: str,
        min_quality: float,
    ) -> str:
        if not top_choice:
            return "No viable deployment candidate found."

        qual_pct = (
            f"{top_choice.relative_quality_score * 100:.1f}%"
            if top_choice.relative_quality_score is not None
            else f"{top_choice.quality_score * 100:.1f}%"
        )
        return (
            f"Recommended configuration: '{top_choice.configuration_name}' ({top_choice.compression_method}). "
            f"Achieves {qual_pct} quality at ${top_choice.cost_per_1m_tokens:.2f}/1M tokens "
            f"with {top_choice.throughput_tok_s:.0f} tok/s on {top_choice.hardware_recommendation or 'standard GPU'}. "
            f"Selected from {total_valid} candidates under '{strategy}' strategy (min_quality={min_quality:.2f})."
        )

    def _generate_ascii_table(
        self,
        ranked: list[ParetoPoint],
        min_quality: float,
        max_cost: float | None,
    ) -> str:
        cost_str = f", Budget < ${max_cost:.2f}/1M" if max_cost else ""
        header_constraint = f"Constraint: SE Quality >= {min_quality*100:.0f}%{cost_str}"

        lines = [
            "╔══════════════════════════════════════════════════════════════════════════════╗",
            "║                      ViPym Compression Recommendations                      ║",
            "╠══════════════════════════════════════════════════════════════════════════════╣",
            f"║ {header_constraint:<76} ║",
            "╠════════╦═════════════════════╦═══════╦══════════╦════════════╦═══════════════╣",
            "║ Rank   ║ Variant             ║ Qual% ║ $/1M Tok ║ Tok/s      ║ HW Instance   ║",
            "╠════════╬═════════════════════╬═══════╬══════════╬════════════╬═══════════════╣",
        ]

        for rank, p in enumerate(ranked, 1):
            qual = (
                f"{p.relative_quality_score * 100:.0f}%"
                if p.relative_quality_score is not None
                else f"{p.quality_score * 100:.0f}%"
            )
            cost = f"${p.cost_per_1m_tokens:.2f}"
            tok_s = f"{p.throughput_tok_s:.0f}"
            hw = p.hardware_recommendation or "1x H100"
            lines.append(
                f"║ {rank:<6} ║ {p.configuration_name:<19} ║ {qual:<5} ║ {cost:<8} ║ {tok_s:<10} ║ {hw:<13} ║"
            )

        lines.append(
            "╚════════╩═════════════════════╩═══════╩══════════╩════════════╩═══════════════╝"
        )
        return "\n".join(lines)
