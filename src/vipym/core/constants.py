"""Core constants used throughout ViPym."""

from enum import Enum


class ComputeArchitecture(str, Enum):
    DENSE = "dense"
    MOE = "moe"
    HYBRID_ATTENTION = "hybrid_attention"


class SupportedDtype(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    MXFP4 = "mxfp4"
    MXFP8 = "mxfp8"
    INT8 = "int8"
    INT4 = "int4"


class CompressionMethodType(str, Enum):
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


class ServingBackendType(str, Enum):
    VLLM = "vllm"
    SGLANG = "sglang"
    HUGGINGFACE = "hf"


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


DEFAULT_AWS_REGION = "us-east-1"
DEFAULT_AWS_HOURLY_RATE_P5 = 32.77
DEFAULT_S3_STORAGE_RATE = 0.023
DEFAULT_DATA_TRANSFER_RATE = 0.09
