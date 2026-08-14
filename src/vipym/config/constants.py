"""Core constants, enums, and default configuration values for ViPym."""

from enum import StrEnum


class ExperimentState(StrEnum):
    """Lifecycle states of a ViPym experiment run."""

    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    BASELINE_RUNNING = "BASELINE_RUNNING"
    BASELINE_COMPLETED = "BASELINE_COMPLETED"
    COMPRESSION_RUNNING = "COMPRESSION_RUNNING"
    COMPRESSION_COMPLETED = "COMPRESSION_COMPLETED"
    INFERENCE_VALIDATED = "INFERENCE_VALIDATED"
    EVALUATION_RUNNING = "EVALUATION_RUNNING"
    EVALUATION_COMPLETED = "EVALUATION_COMPLETED"
    ANALYSIS_COMPLETED = "ANALYSIS_COMPLETED"
    REPORT_COMPLETED = "REPORT_COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class ComputeArchitecture(StrEnum):
    """Supported model compute architectures."""

    DENSE = "dense"
    MOE = "moe"
    HYBRID_ATTENTION = "hybrid_attention"  # e.g., Kimi KDA + Gated MLA


class SupportedDtype(StrEnum):
    """Supported numerical precision formats."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    MXFP4 = "mxfp4"
    MXFP8 = "mxfp8"
    INT8 = "int8"
    INT4 = "int4"


class CompressionMethodType(StrEnum):
    """Catalog of supported compression algorithm identifiers."""

    RTN = "rtn"
    AWQ = "awq"
    GPTQ = "gptq"
    SMOOTHQUANT = "smoothquant"
    AUTOROUND = "autoround"
    SPINQUANT = "spinquant"
    QUAROT = "quarot"
    MXFP = "mxfp"
    FP8 = "fp8"
    PRUNE_MAGNITUDE = "prune_magnitude"
    PRUNE_NM = "prune_nm"
    PRUNE_WANDA = "prune_wanda"
    DISTILL_RESPONSE = "distill_response"
    DISTILL_LOGIT = "distill_logit"
    KV_CACHE_FP8 = "kv_cache_fp8"
    KV_CACHE_INT4 = "kv_cache_int4"


class ServingBackendType(StrEnum):
    """Supported serving runtime engines."""

    VLLM = "vllm"
    SGLANG = "sglang"
    HUGGINGFACE = "hf"


class OptimizationObjective(StrEnum):
    """Optimization objectives for Pareto analysis."""

    MAXIMIZE_QUALITY = "maximize_quality"
    MINIMIZE_LATENCY = "minimize_latency"
    MINIMIZE_MEMORY = "minimize_memory"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_COMPRESSION_RATIO = "maximize_compression_ratio"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"


# Default Cloud Assumptions
DEFAULT_AWS_REGION = "us-east-1"
DEFAULT_AWS_HOURLY_RATE_P5 = 32.77
DEFAULT_AWS_HOURLY_RATE_P4DE = 40.96
DEFAULT_AWS_HOURLY_RATE_G5 = 1.006
DEFAULT_S3_STORAGE_RATE = 0.023
DEFAULT_DATA_TRANSFER_RATE = 0.09
