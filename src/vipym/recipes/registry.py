"""Recipe Registry and Catalog Loader for ViPym."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from vipym.config.exceptions import ConfigurationError
from vipym.config.schema import ViPymExperimentConfig


@dataclass
class CompressionRecipeMetadata:
    """Metadata describing a curated compression recipe."""

    recipe_id: str
    name: str
    target_model_family: str
    domain: str
    expected_compression_ratio: float
    expected_quality_retention: str
    hardware_target: str
    description: str
    tags: list[str] = field(default_factory=list)
    config_path: Path = field(default_factory=Path)


class RecipeRegistry:
    """Discovers, catalogs, and retrieves built-in and user compression recipes."""

    _recipes: dict[str, CompressionRecipeMetadata] = {}
    _initialized: bool = False

    @classmethod
    def get_recipes_dir(cls) -> Path:
        """Find the root recipes directory in the repository or package."""
        # 1. Check workspace recipes/ directory
        workspace_recipes = Path("recipes")
        if workspace_recipes.exists() and workspace_recipes.is_dir():
            return workspace_recipes.resolve()

        # 2. Check package relative directory
        pkg_recipes = Path(__file__).parent.parent.parent.parent / "recipes"
        if pkg_recipes.exists():
            return pkg_recipes.resolve()

        return Path("./recipes").resolve()

    @classmethod
    def discover(cls, custom_dir: Path | None = None) -> dict[str, CompressionRecipeMetadata]:
        """Scan directory and register all valid YAML compression recipes."""
        cls._recipes.clear()
        search_dirs = [cls.get_recipes_dir()]
        if custom_dir:
            search_dirs.append(Path(custom_dir).resolve())

        for sdir in search_dirs:
            if not sdir.exists():
                continue
            for f in sdir.glob("*.yaml"):
                try:
                    with open(f, encoding="utf-8") as yf:
                        raw_data = yaml.safe_load(yf)

                    if not raw_data or "experiment_id" not in raw_data:
                        continue

                    # Extract descriptive metadata
                    recipe_id = f.stem
                    name = raw_data.get("description", recipe_id).split(".")[0]
                    model_id = raw_data.get("model", {}).get("id", "Unknown")
                    stages = raw_data.get("compression_pipeline", [])
                    stage_names = [s.get("method", "") for s in stages if isinstance(s, dict)]

                    meta = CompressionRecipeMetadata(
                        recipe_id=recipe_id,
                        name=name,
                        target_model_family=model_id,
                        domain="Software Engineering"
                        if "humaneval" in str(raw_data)
                        else "General Reasoning",
                        expected_compression_ratio=4.0
                        if "W4A16" in str(raw_data) or "MXFP4" in str(raw_data)
                        else 2.0,
                        expected_quality_retention="> 95% Pass@1"
                        if "quarot" in stage_names
                        else "> 90% Pass@1",
                        hardware_target=raw_data.get("infrastructure", {}).get(
                            "instance_type", "local"
                        ),
                        description=raw_data.get(
                            "description", "Curated ViPym compression recipe."
                        ),
                        tags=stage_names + [model_id.split("/")[-1]],
                        config_path=f.resolve(),
                    )
                    cls._recipes[recipe_id] = meta
                except Exception:
                    continue

        cls._initialized = True
        return cls._recipes

    @classmethod
    def list_recipes(cls) -> dict[str, CompressionRecipeMetadata]:
        """Return catalog of all available recipes."""
        if not cls._initialized or not cls._recipes:
            cls.discover()
        return cls._recipes

    @classmethod
    def get(cls, recipe_id: str) -> CompressionRecipeMetadata:
        """Retrieve recipe metadata by ID."""
        recipes = cls.list_recipes()
        # Direct lookup or clean normalized lookup
        clean_id = recipe_id.replace(".yaml", "")
        if clean_id in recipes:
            return recipes[clean_id]

        # Search by partial match or tag
        for k, v in recipes.items():
            if clean_id.lower() in k.lower():
                return v

        raise ConfigurationError(
            f"Recipe '{recipe_id}' not found in catalog. Available recipes: {list(recipes.keys())}"
        )

    @classmethod
    def load_config(cls, recipe_id: str) -> ViPymExperimentConfig:
        """Load and validate the ViPymExperimentConfig object for a recipe."""
        meta = cls.get(recipe_id)
        return ViPymExperimentConfig.from_yaml(meta.config_path)
