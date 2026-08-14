"""Unit tests for the ViPym Recipe Hub and Registry."""

import pytest

from vipym.config.exceptions import ConfigurationError
from vipym.config.schema import ViPymExperimentConfig
from vipym.recipes.registry import RecipeRegistry


def test_recipe_discovery() -> None:
    """Test that all curated recipes in the workspace are discovered."""
    recipes = RecipeRegistry.discover()
    assert len(recipes) >= 4
    assert "kimi_k3_software_engineering_matrix" in recipes
    assert "qwen2.5_coder_32b_w4a16_quarot" in recipes
    assert "llama3_3_70b_awq_w4a16" in recipes
    assert "deepseek_v3_moe_mxfp4_sparse" in recipes


def test_recipe_metadata_attributes() -> None:
    """Test metadata extraction from recipe YAML files."""
    recipe = RecipeRegistry.get("kimi_k3_software_engineering_matrix")
    assert recipe.recipe_id == "kimi_k3_software_engineering_matrix"
    assert recipe.target_model_family == "moonshotai/Kimi-K3"
    assert recipe.domain == "Software Engineering"
    assert recipe.expected_compression_ratio > 1.0


def test_recipe_load_config() -> None:
    """Test loading and Pydantic validation of recipe configuration."""
    cfg = RecipeRegistry.load_config("qwen2.5_coder_32b_w4a16_quarot")
    assert isinstance(cfg, ViPymExperimentConfig)
    assert cfg.model.id == "Qwen/Qwen2.5-Coder-32B-Instruct"
    assert len(cfg.compression_pipeline) == 3


def test_recipe_not_found() -> None:
    """Test exception raising on non-existent recipe."""
    with pytest.raises(ConfigurationError):
        RecipeRegistry.get("non_existent_recipe_xyz_999")
