"""Analysis, Pareto Frontier, Cost Modeling, and Decision Engine Package."""

from vipym.analysis.cost_model import (
    COMMERCIAL_APIS,
    STANDARD_INSTANCES,
    DeploymentCostModel,
    EnterpriseWorkloadConfig,
    InstancePricing,
    VariantCostBreakdown,
)
from vipym.analysis.pareto import ParetoFrontierOptimizer, ParetoPoint
from vipym.analysis.recommender import DeploymentRecommender, RecommendationReport
from vipym.analysis.report import HTMLReportGenerator
from vipym.analysis.statistics import StatisticalAnalyzer
from vipym.analysis.trade_offs import TradeOffAnalyzer
from vipym.analysis.visualizations import (
    create_cost_comparison_bar_chart,
    create_pareto_scatter_plot,
    create_quality_retention_heatmap,
)

__all__ = [
    "COMMERCIAL_APIS",
    "DeploymentCostModel",
    "DeploymentRecommender",
    "EnterpriseWorkloadConfig",
    "HTMLReportGenerator",
    "InstancePricing",
    "ParetoFrontierOptimizer",
    "ParetoPoint",
    "RecommendationReport",
    "STANDARD_INSTANCES",
    "StatisticalAnalyzer",
    "TradeOffAnalyzer",
    "VariantCostBreakdown",
    "create_cost_comparison_bar_chart",
    "create_pareto_scatter_plot",
    "create_quality_retention_heatmap",
]
