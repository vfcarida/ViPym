# ADR-004: Pydantic v2 for Unified Schema Configuration and Validation

## Status
Accepted

## Context
Compression experiments involve complex multi-tiered configurations spanning model metadata, DAG pipeline stages, algorithm hyperparameters, inference serving parameters, benchmark thresholds, cost assumptions, and cloud infrastructure targets. 

Unvalidated configuration files lead to silent parameter mistranslations, runtime crashes hours into execution, and non-reproducible research results.

## Decision
ViPym enforces strict, compile-time and runtime validation using **Pydantic v2**:
1. **Schema Centralization**: `ViPymExperimentConfig` serves as the authoritative source of truth across CLI, REST API, Studio Web UI, and Python SDK.
2. **Strict Field Validation**: Literal constraints on methods (`awq`, `gptq`, `fp8`, `quarot`), schemes (`W4A16`, `FP8`, `2:4_SPARSITY`), and bounds on hyperparameters.
3. **Automated Serialization**: Seamless bidirectional conversion between YAML, JSON, Python dictionaries, and OpenAPI / JSON Schema schemas.
4. **Reproducibility Manifests**: Pydantic models automatically hash experiment configs into cryptographic SHA-256 manifest IDs (`manifest-<exp_id>-<hash>`).

## Consequences
### Positive
- Errors in recipe YAMLs or CLI arguments are flagged instantly before any expensive GPU memory is allocated.
- Web UI and CLI share identical validation logic.
- High-performance Rust-backed validation core in Pydantic v2.

### Negative
- Configuration additions require updating Pydantic schemas in `src/vipym/config/schema.py`.
