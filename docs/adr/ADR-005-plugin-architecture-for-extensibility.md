# ADR-005: Decoupled Plugin Registries for Extensibility

## Status
Accepted

## Context
The landscape of LLM compression, model architectures, inference engines, and evaluation benchmarks is rapidly evolving. Hard-coding algorithms or tightly coupling compression logic to specific serving runtimes prevents researchers from integrating emerging techniques (e.g. newly published quantization papers or new evaluation datasets).

## Decision
ViPym establishes **Decoupled Plugin Registries** across four core axes:
1. `CompressionRegistry`: Registers compression methods extending `CompressionMethod` (`awq`, `gptq`, `fp8`, `quarot`, `mxfp`, `prune_wanda`, `distill_response`, etc.).
2. `EvaluationRegistry`: Registers benchmark evaluation suites extending `EvaluationSuite` (`humaneval`, `mbpp`, `bigcodebench`, `aider`, `swebench`, etc.).
3. `ModelRegistry`: Registers architecture adapters extending `ModelAdapter` (`hf`, `kimi_k3`, `mixtral`, `deepseek_v3`, `qwen`, `llama`).
4. `InferenceRegistry`: Registers high-throughput inference backends extending `InferenceBackend` (`vllm`, `sglang`, `hf`).

New methods and suites register via simple decorators or class registration (`@CompressionRegistry.register("my_method")` or `CompressionRegistry.register("my_method", MyMethodClass)`).

## Consequences
### Positive
- Third-party researchers can add new compression algorithms in a standalone file without modifying core execution or reporting engines.
- Clean separation of concerns between compression math, inference serving, and benchmark scoring.
- Comprehensive dynamic inspection via `vipym methods`, `vipym suites`, and `vipym models`.

### Negative
- Requires maintaining abstract base interfaces with rigorous typing and capability declarations.
