"""Pydantic v2 configuration schemas for ViPym experiments."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from vipym.config.constants import (
    DEFAULT_AWS_HOURLY_RATE_P5,
    DEFAULT_AWS_REGION,
    DEFAULT_DATA_TRANSFER_RATE,
    DEFAULT_S3_STORAGE_RATE,
    OptimizationObjective,
)
from vipym.config.exceptions import ConfigurationError


class ModelConfig(BaseModel):
    """Configuration for target foundational LLM."""

    id: str = Field(..., description="HuggingFace model ID, local checkpoint path, or S3 URI")
    revision: str = Field(default="main", description="Explicit immutable git commit SHA or tag")
    trust_remote_code: bool = Field(
        default=False, description="Whether to allow remote code execution"
    )
    custom_adapter_cls: str | None = Field(
        default=None, description="Optional custom ModelAdapter class name"
    )
    device_map: str = Field(
        default="auto", description="PyTorch device map for offline compression loading"
    )
    torch_dtype: str | None = Field(
        default=None, description="Explicit PyTorch precision (e.g. bfloat16, float16)"
    )


class CalibrationConfig(BaseModel):
    """Configuration for compression calibration dataset."""

    dataset_name: str = Field(
        default="wikitext", description="Dataset identifier or Hugging Face dataset path"
    )
    dataset_config: str | None = Field(
        default="wikitext-2-raw-v1", description="Dataset subset/configuration"
    )
    split: str = Field(default="train", description="Dataset split to calibrate on")
    num_samples: int = Field(default=512, ge=1, description="Number of calibration sequences")
    sequence_length: int = Field(
        default=2048, ge=128, description="Token length per calibration sample"
    )
    shuffle: bool = Field(default=True)
    seed: int = Field(default=42)


class CompressionStageConfig(BaseModel):
    """Configuration for a discrete compression stage in the execution DAG."""

    stage_id: str = Field(..., pattern=r"^[a-zA-Z0-9_-]+$", description="Unique stage identifier")
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
    ] = Field(..., description="Compression method name")
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
    ] = Field(..., description="Target quantization or sparsity scheme")
    calibration: CalibrationConfig | None = Field(
        default=None, description="Calibration parameters if required"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Algorithm-specific hyper-parameters"
    )
    dependencies: list[str] = Field(
        default_factory=list, description="IDs of prerequisite DAG stages"
    )


class ServingConfig(BaseModel):
    """Inference serving engine configuration."""

    backend: Literal["vllm", "sglang", "hf"] = Field(
        default="vllm", description="Serving runtime backend"
    )
    tensor_parallel_size: int = Field(default=1, ge=1, description="Tensor parallel degree")
    pipeline_parallel_size: int = Field(default=1, ge=1, description="Pipeline parallel degree")
    max_model_len: int = Field(default=4096, ge=256, description="Maximum context length")
    gpu_memory_utilization: float = Field(
        default=0.90, gt=0.0, le=1.0, description="VRAM quota fraction"
    )
    kv_cache_dtype: Literal["auto", "fp8", "fp8_e4m3", "fp8_e5m2", "int4"] = Field(
        default="auto", description="KV-cache precision"
    )
    enable_speculative_decoding: bool = Field(
        default=False, description="Enable speculative draft model decoding"
    )
    speculative_draft_model: str | None = Field(default=None, description="Draft model identifier")
    num_speculative_tokens: int = Field(default=5, ge=1, description="Speculative tokens per step")
    max_num_seqs: int = Field(default=256, ge=1, description="Max concurrent sequences in engine")


class EvaluationConfig(BaseModel):
    """Evaluation suite configuration."""

    suites: list[str] = Field(
        default_factory=lambda: ["humaneval"],
        min_length=1,
        description="List of benchmark suite names",
    )
    timeout_per_task_sec: int = Field(
        default=15, ge=1, description="Per-task execution timeout in sandbox"
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, gt=0.0, le=1.0, description="Top-p sampling")
    max_new_tokens: int = Field(default=2048, ge=64, description="Max generated tokens per problem")
    pass_k_values: list[int] = Field(
        default_factory=lambda: [1], description="Values of k for pass@k metric"
    )
    isolate_with_gvisor: bool = Field(
        default=True, description="Enforce gVisor sandbox container isolation"
    )
    task_limit: int | None = Field(
        default=None, description="Optional cap on number of tasks for smoke tests"
    )
    num_measurement_repeats: int = Field(
        default=3, ge=1, description="Repeats per benchmark task for statistical rigor"
    )


class CostAssumptionConfig(BaseModel):
    """Traceable AWS cloud cost modeling parameters."""

    provider: Literal["aws", "gcp", "azure", "custom"] = Field(default="aws")
    region: str = Field(default=DEFAULT_AWS_REGION)
    pricing_source: str = Field(
        default="config", description="Source of pricing assumptions (config/api)"
    )
    aws_ec2_hourly_rate: float = Field(
        default=DEFAULT_AWS_HOURLY_RATE_P5, description="Hourly rate of instance ($/hr)"
    )
    s3_storage_cost_per_gb_month: float = Field(
        default=DEFAULT_S3_STORAGE_RATE, description="Monthly storage rate ($/GB)"
    )
    data_transfer_per_gb: float = Field(
        default=DEFAULT_DATA_TRANSFER_RATE, description="Egress transfer rate ($/GB)"
    )
    active_gpu_count: int = Field(default=8, ge=1, description="Active GPU count utilized")


class OptimizationConfig(BaseModel):
    """Objectives and constraints for multi-dimensional Pareto frontier calculation."""

    objectives: list[OptimizationObjective] = Field(
        default_factory=lambda: [
            OptimizationObjective.MAXIMIZE_QUALITY,
            OptimizationObjective.MINIMIZE_LATENCY,
            OptimizationObjective.MINIMIZE_MEMORY,
            OptimizationObjective.MINIMIZE_COST,
        ]
    )
    min_acceptable_pass_at_1: float | None = Field(default=None, ge=0.0, le=1.0)
    max_acceptable_vram_gb: float | None = Field(default=None, gt=0.0)
    max_acceptable_latency_ms: float | None = Field(default=None, gt=0.0)


class InfrastructureConfig(BaseModel):
    """Infrastructure execution target."""

    provider: Literal["local", "aws_ec2", "sagemaker"] = Field(default="local")
    aws_region: str = Field(default=DEFAULT_AWS_REGION)
    instance_type: str | None = Field(default=None, description="e.g. p5.48xlarge, p4de.24xlarge")
    s3_bucket: str | None = Field(default=None, description="S3 bucket for artifacts & manifests")
    auto_teardown: bool = Field(
        default=True, description="Automatically terminate ephemeral nodes after run"
    )
    idle_shutdown_timeout_min: int = Field(
        default=20, ge=5, description="Watchdog poweroff minutes on idle"
    )


class ViPymExperimentConfig(BaseModel):
    """Root configuration object for an entire ViPym experiment run."""

    experiment_id: str = Field(
        ..., pattern=r"^[a-zA-Z0-9_-]+$", description="Unique alphanumeric experiment identifier"
    )
    seed: int = Field(default=42, description="Global random seed")
    description: str | None = Field(
        default=None, description="Human-readable experiment description"
    )
    model: ModelConfig
    compression_pipeline: list[CompressionStageConfig] = Field(default_factory=list)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    cost_assumptions: CostAssumptionConfig = Field(default_factory=CostAssumptionConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
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
