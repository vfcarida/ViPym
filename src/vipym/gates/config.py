"""Quality Gate Threshold Configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pydantic
import yaml


class GateThresholds(pydantic.BaseModel):
    """Quality and performance threshold specifications for an evaluation gate."""

    name: str = "se_production"
    min_se_composite: float = 0.65  # Relative retention or absolute minimum SE composite score
    min_humaneval_pass1: float = 0.70
    min_aider_edit: float = 0.70
    min_bigcodebench: float = 0.50
    min_swebench: float = 0.35
    min_testgeneval_coverage: float = 0.60
    min_crqbench_precision: float = 0.40
    max_latency_p95_ms: float = 5000.0
    max_quality_drop_any_suite: float = (
        0.50  # Max allowable quality drop (retention must be >= 50%)
    )
    suite_thresholds: dict[str, float] = pydantic.Field(default_factory=dict)
    use_relative_scoring: bool = (
        True  # True = compare against teacher baseline, False = absolute thresholds
    )


class GatesConfig(pydantic.BaseModel):
    """Container for multiple named gate definitions."""

    gates: dict[str, GateThresholds] = pydantic.Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> GatesConfig:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Gate config file not found: {p}")
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GatesConfig:
        gates_dict = data.get("gates", data)
        parsed_gates: dict[str, GateThresholds] = {}
        for name, cfg in gates_dict.items():
            if isinstance(cfg, dict):
                parsed_gates[name] = GateThresholds(name=name, **cfg)
            elif isinstance(cfg, GateThresholds):
                parsed_gates[name] = cfg
        if not parsed_gates:
            parsed_gates["se_production"] = GateThresholds(name="se_production")
        return cls(gates=parsed_gates)

    def get_gate(self, name: str = "se_production") -> GateThresholds:
        if name in self.gates:
            return self.gates[name]
        return next(iter(self.gates.values()), GateThresholds(name=name))
