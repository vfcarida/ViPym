"""Unit tests for Software Engineering Lifecycle Demo and Matrix Recipes."""

from __future__ import annotations

from pathlib import Path

import pytest

from vipym.config.schema import ViPymExperimentConfig
from vipym.experiments.runner import ResumableExperimentRunner


class TestSELifecycleRecipes:
    def test_se_lifecycle_demo_recipe_schema(self):
        """Verify recipes/se-lifecycle-demo.yaml matches ViPymExperimentConfig schema."""
        recipe_path = Path("recipes/se-lifecycle-demo.yaml")
        assert recipe_path.exists(), "recipes/se-lifecycle-demo.yaml must exist"

        cfg = ViPymExperimentConfig.from_yaml(recipe_path)
        assert cfg.experiment_id == "se-lifecycle-demo-gpt2"
        assert cfg.model.id == "openai-community/gpt2"
        assert len(cfg.compression_pipeline) == 2
        assert cfg.compression_pipeline[0].method == "prune_wanda"
        assert cfg.compression_pipeline[1].method == "gptq"
        assert "humaneval" in cfg.evaluation.suites
        assert cfg.evaluation.task_limit == 5

    def test_se_lifecycle_kimi_k3_recipe_schema(self):
        """Verify recipes/se-lifecycle-kimi-k3.yaml matches ViPymExperimentConfig schema."""
        recipe_path = Path("recipes/se-lifecycle-kimi-k3.yaml")
        assert recipe_path.exists(), "recipes/se-lifecycle-kimi-k3.yaml must exist"

        cfg = ViPymExperimentConfig.from_yaml(recipe_path)
        assert cfg.experiment_id == "se-lifecycle-kimi-k3-matrix"
        assert cfg.model.id == "moonshotai/Kimi-K3"
        assert len(cfg.compression_pipeline) == 7
        assert "humaneval" in cfg.evaluation.suites
        assert "bigcodebench" in cfg.evaluation.suites
        assert "aider" in cfg.evaluation.suites
    def test_prebuilt_recipes_hub_schemas(self):
        """Verify all 5 pre-built hub recipes parse and validate cleanly."""
        hub_recipes = [
            "recipes/quick-demo-gpt2.yaml",
            "recipes/mixtral-compression.yaml",
            "recipes/kimi-k3-full.yaml",
            "recipes/cost-optimized-se.yaml",
            "recipes/quality-first-se.yaml",
        ]
        for r_path_str in hub_recipes:
            r_path = Path(r_path_str)
            assert r_path.exists(), f"Recipe file {r_path_str} must exist"
            cfg = ViPymExperimentConfig.from_yaml(r_path)
            assert cfg.experiment_id is not None
            assert cfg.model.id is not None
            assert len(cfg.compression_pipeline) >= 1
            assert len(cfg.evaluation.suites) >= 1

    def test_demo_recipe_dry_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Verify demo runner initializes and generates all reports without failure."""
        monkeypatch.setenv("VIPYM_ALLOW_UNSAFE", "1")
        cfg = ViPymExperimentConfig.from_yaml("recipes/quick-demo-gpt2.yaml")
        cfg.evaluation.task_limit = 1

        runner = ResumableExperimentRunner(config=cfg, artifacts_dir=tmp_path)
        assert runner.exp_dir.exists()
        assert runner.manifest is not None
