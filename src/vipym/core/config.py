"""Typed Pydantic configuration schemas for ViPym experiments."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from vipym.core.constants import (
    DEFAULT_AWS_HOURLY_RATE_P5,
    DEFAULT_AWS_REGION,
    DEFAULT_DATA_TRANSFER_RATE,
    DEFAULT_S3_STORAGE_RATE,
)
from vipym.core.exceptions import ConfigurationError


class ModelConfig(BaseModel):
    """Configuration for target foundational LLM."""

    id: str = Field(..., description="HuggingFace model ID or S3 URI")
    revision: str = Field(default="main", description="Explicit git commit SHA or tag")
    trust_remote_code: bool = Field(default=False, description="Allow remote code execution")
    custom_adapter_cls: str | None = Field(
        default=None, description="Custom ModelAdapter class name"
    )
    device_map: str = Field(default="auto", description="PyTorch device map")


class CalibrationConfig(BaseModel):
    """Configuration for compression calibration datasets."""

    dataset_name: str = Field(default="wikitext", description="Dataset identifier")
    dataset_config: str | None = Field(default="wikitext-2-raw-v1")
    split: str = Field(default="train")
    num_samples: int = Field(default=512, ge=1)
    sequence_length: int = Field(default=2048, ge=128)
    shuffle: bool = Field(default=True)
    seed: int = Field(default=42)


class CompressionStageConfig(BaseModel):
    """Configuration for a discrete compression stage in the execution DAG."""

    stage_id: str = Field(..., pattern=r"^[a-zA-Z0-9_-]+$")
    method: Literal[
        "rtn",
        "awq",
        "gptq",
        "smoothquant",
        "autoround",
        "spinquant",
        "quarot",
        "mxfp",
        "fp8",
        "prune_magnitude",
        "prune_nm",
        "prune_wanda",
        "distill_response",
        "distill_logit",
        "kv_cache_fp8",
        "kv_cache_int4",
    ]
    scheme: Literal[
        "W4A16",
        "W8A8",
        "W8A16",
        "FP8",
        "MXFP4",
        "MXFP8",
        "INT4",
        "INT8",
        "2:4_SPARSITY",
        "STRUCTURED_SPARSITY",
        "UNSTRUCTURED_SPARSITY",
        "DISTILL_STUDENT",
    ]
    calibration: CalibrationConfig | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)


class ServingConfig(BaseModel):
    """Inference serving engine configuration."""

    backend: Literal["vllm", "sglang", "hf"] = Field(default="vllm")
    tensor_parallel_size: int = Field(default=1, ge=1)
    pipeline_parallel_size: int = Field(default=1, ge=1)
    max_model_len: int = Field(default=4096, ge=512)
    gpu_memory_utilization: float = Field(default=0.90, gt=0.0, le=1.0)
    kv_cache_dtype: Literal["auto", "fp8", "fp8_e4m3", "fp8_e5m2", "int4"] = Field(default="auto")
    enable_speculative_decoding: bool = Field(default=False)
    speculative_draft_model: str | None = None
    max_num_seqs: int = Field(default=256, ge=1)


class EvaluationConfig(BaseModel):
    """Evaluation suite configuration."""

    suites: list[str] = Field(default_factory=lambda: ["humaneval"])
    timeout_per_task_sec: int = Field(default=15, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    max_new_tokens: int = Field(default=2048, ge=64)
    pass_k_values: list[int] = Field(default_factory=lambda: [1])
    isolate_with_gvisor: bool = Field(default=True)
    task_limit: int | None = Field(
        default=None, description="Optional cap on number of tasks for smoke runs"
    )


class CostAssumptionConfig(BaseModel):
    """Traceable AWS cloud cost assumptions."""

    aws_ec2_hourly_rate: float = Field(default=DEFAULT_AWS_HOURLY_RATE_P5)
    s3_storage_cost_per_gb_month: float = Field(default=DEFAULT_S3_STORAGE_RATE)
    data_transfer_per_gb: float = Field(default=DEFAULT_DATA_TRANSFER_RATE)
    active_gpu_count: int = Field(default=8, ge=1)


class InfrastructureConfig(BaseModel):
    """Infrastructure target configuration."""

    provider: Literal["local", "aws_ec2", "sagemaker"] = Field(default="local")
    aws_region: str = Field(default=DEFAULT_AWS_REGION)
    instance_type: str | None = None
    s3_bucket: str | None = None
    auto_teardown: bool = Field(default=True)


class ViPymExperimentConfig(BaseModel):
    """Root configuration object for an entire ViPym experiment run."""

    experiment_id: str = Field(..., pattern=r"^[a-zA-Z0-9_-]+$")
    seed: int = Field(default=42)
    description: str | None = Field(default=None)
    model: ModelConfig
    compression_pipeline: list[CompressionStageConfig] = Field(default_factory=list)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    cost_assumptions: CostAssumptionConfig = Field(default_factory=CostAssumptionConfig)
    infrastructure: InfrastructureConfig = Field(default_factory=InfrastructureConfig)

    @field_validator("compression_pipeline")
    @classmethod
    def validate_stage_ids(
        cls, stages: list[CompressionStageConfig]
    ) -> list[CompressionStageConfig]:
        seen = set()
        for s in stages:
            if s.stage_id in seen:
                raise ValueError(f"Duplicate stage_id '{s.stage_id}' in compression_pipeline")
            seen.add(s.stage_id)
        return stages

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ViPymExperimentConfig":
        p = Path(path)
        if not p.exists():
            raise ConfigurationError(f"Config file not found: {p}")
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(), f, sort_keys=False)
