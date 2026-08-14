# ViPym — Repository Inventory & File Classification

**Inventory Date:** 2026-08-14  
**Base Commit SHA:** `5afbdac`  
**Work Branch:** `audit-and-improvement-20260814`  
**Total Tracked Files:** 95  

---

## 1. Classification Summary

| Classification | File Count | Primary Directory Paths |
|---|---|---|
| **First-Party Source** | 58 | `src/vipym/` (`cli/`, `config/`, `models/`, `compression/`, `pipelines/`, `inference/`, `evaluation/`, `metrics/`, `cost/`, `artifacts/`, `experiments/`, `analysis/`, `reporting/`, `aws/`, `security/`, `utils/`, `core/`, `cloud/`) |
| **Test Suite** | 9 | `tests/` (`unit/`, `contract/`, `integration/`, `smoke/`, `e2e/`) |
| **Configuration** | 10 | `configs/` (`models/`, `compression/`, `evaluation/`, `infrastructure/`, `experiments/`), `pyproject.toml` |
| **Documentation** | 8 | `README.md`, `docs/` (`index.md`, `architecture.md`, `security.md`, `aws.md`), `CONTRIBUTING.md`, `SECURITY.md`, `CHANGELOG.md` |
| **Examples & Scripts** | 4 | `examples/` (`quickstart_python.py`, `custom_compression_plugin.py`), `scripts/` (`aws_cleanup.sh`, `cluster_bootstrap.sh`) |
| **Container & CI/CD** | 5 | `docker/` (`Dockerfile.vipym-core`, `Dockerfile.eval-sandbox`), `.github/workflows/` (`ci.yml`, `security.yml`, `gpu-integration.yml`) |
| **Legal & Hygiene** | 1 | `LICENSE` (Apache-2.0) |

---

## 2. Granular File Ledger

| Relative Path | Classification | Size (Bytes) | Role / Responsibility |
|---|---|---|---|
| `pyproject.toml` | Configuration | 2,629 | Build definition, dependencies, entry points, tool configurations |
| `README.md` | Documentation | 12,040 | Project overview, architecture diagrams, quickstart, specs |
| `LICENSE` | Legal | 11,357 | Apache 2.0 license terms |
| `CONTRIBUTING.md` | Documentation | 812 | Contributor guidelines, testing standards |
| `SECURITY.md` | Documentation | 480 | Vulnerability reporting and sandboxing policy |
| `CHANGELOG.md` | Documentation | 1,480 | Semantic version release history |
| `src/vipym/__version__.py` | First-Party Source | 121 | Version definition (`0.1.0`) |
| `src/vipym/cli/main.py` | First-Party Source | 11,102 | Main Typer CLI command dispatch |
| `src/vipym/cli/doctor.py` | First-Party Source | 2,982 | Environment and dependency diagnostic suite |
| `src/vipym/config/constants.py` | First-Party Source | 2,465 | Enums, states, pricing constants |
| `src/vipym/config/exceptions.py` | First-Party Source | 1,804 | Typed exception hierarchy |
| `src/vipym/config/schema.py` | First-Party Source | 9,347 | Pydantic v2 schemas for all config domains |
| `src/vipym/pipelines/dag.py` | First-Party Source | 5,341 | DAG execution engine with Kahn's topological sort |
| `src/vipym/pipelines/node.py` | First-Party Source | 571 | DAG stage node representation |
| `src/vipym/models/registry.py` | First-Party Source | 1,061 | Dynamic model adapter registry |
| `src/vipym/models/hf_adapter.py` | First-Party Source | 5,250 | HuggingFace AutoModel adapter |
| `src/vipym/models/architectures/kimi_k3.py` | First-Party Source | 4,947 | Kimi K3 MoE architecture introspector |
| `src/vipym/compression/registry.py` | First-Party Source | 979 | Compression algorithm registry |
| `src/vipym/compression/quantization/awq.py` | First-Party Source | 3,255 | AWQ W4A16 quantization adapter |
| `src/vipym/compression/quantization/gptq.py` | First-Party Source | 3,229 | GPTQ second-order quantization adapter |
| `src/vipym/compression/quantization/smoothquant.py` | First-Party Source | 3,050 | SmoothQuant W8A8 adapter |
| `src/vipym/compression/quantization/autoround.py` | First-Party Source | 3,009 | AutoRound sign-gradient quantization adapter |
| `src/vipym/compression/quantization/fp8.py` | First-Party Source | 3,055 | FP8 (E4M3/E5M2) quantization adapter |
| `src/vipym/compression/quantization/mxfp.py` | First-Party Source | 2,752 | Microscaling (MXFP4/MXFP8) adapter |
| `src/vipym/compression/quantization/llm_compressor.py` | First-Party Source | 3,982 | LLM-Compressor / vLLM integration |
| `src/vipym/compression/transforms/spinquant.py` | First-Party Source | 4,199 | SpinQuant / QuaRot Hadamard transform adapter |
| `src/vipym/compression/pruning/magnitude.py` | First-Party Source | 6,660 | Magnitude, 2:4 semi-structured, and Wanda pruning |
| `src/vipym/compression/distillation/response_distill.py` | First-Party Source | 3,196 | Sequence-level knowledge distillation adapter |
| `src/vipym/compression/distillation/logit_distill.py` | First-Party Source | 3,313 | Token-level logit distillation & teacher cache |
| `src/vipym/compression/kv_cache/fp8_kv.py` | First-Party Source | 2,506 | FP8 & INT4 KV cache quantization adapter |
| `src/vipym/inference/registry.py` | First-Party Source | 965 | Inference runtime registry |
| `src/vipym/inference/vllm_engine.py` | First-Party Source | 3,741 | vLLM high-throughput serving engine adapter |
| `src/vipym/inference/sglang_engine.py` | First-Party Source | 2,800 | SGLang RadixAttention serving adapter |
| `src/vipym/inference/hf_engine.py` | First-Party Source | 4,688 | PyTorch / HF fallback serving adapter |
| `src/vipym/inference/speculative.py` | First-Party Source | 1,091 | Speculative decoding draft engine |
| `src/vipym/evaluation/registry.py` | First-Party Source | 967 | Evaluation benchmark registry |
| `src/vipym/evaluation/runner.py` | First-Party Source | 2,633 | Benchmark execution orchestrator |
| `src/vipym/evaluation/contamination.py` | First-Party Source | 1,583 | N-gram overlap and release cutoff auditor |
| `src/vipym/evaluation/sandbox/docker_sandbox.py` | First-Party Source | 3,519 | Isolated subprocess and container code runner |
| `src/vipym/evaluation/sandbox/security_profile.py` | First-Party Source | 1,408 | Seccomp, rlimits, and isolation configuration |
| `src/vipym/evaluation/suites/humaneval.py` | First-Party Source | 3,596 | HumanEval / HumanEval+ benchmark suite |
| `src/vipym/evaluation/suites/mbpp.py` | First-Party Source | 5,436 | MBPP / MBPP+ benchmark suite |
| `src/vipym/metrics/collector.py` | First-Party Source | 2,026 | TTFT, ITL, VRAM, and RSS telemetry collector |
| `src/vipym/metrics/quality.py` | First-Party Source | 2,737 | Pass@k, compile rate, unit test rate |
| `src/vipym/metrics/cost.py` | First-Party Source | 1,993 | Cost metric aggregator |
| `src/vipym/cost/calculator.py` | First-Party Source | 2,464 | Traceable AWS cloud cost calculator |
| `src/vipym/cost/providers.py` | First-Party Source | 943 | GPU compute pricing presets |
| `src/vipym/artifacts/store.py` | First-Party Source | 1,661 | Local filesystem artifact store |
| `src/vipym/experiments/state.py` | First-Party Source | 3,644 | Experiment state machine & validator |
| `src/vipym/experiments/checkpoint.py` | First-Party Source | 1,872 | Resumable stage checkpoint manager |
| `src/vipym/experiments/manifest.py` | First-Party Source | 4,381 | Environment provenance & manifest generator |
| `src/vipym/experiments/runner.py` | First-Party Source | 13,870 | Resumable experiment execution engine |
| `src/vipym/analysis/pareto.py` | First-Party Source | 3,889 | Multi-objective non-dominated Pareto frontier |
| `src/vipym/analysis/statistics.py` | First-Party Source | 2,878 | Bootstrap confidence intervals & statistical tests |
| `src/vipym/analysis/trade_offs.py` | First-Party Source | 1,066 | Marginal efficiency & trade-off analyzer |
| `src/vipym/reporting/generator.py` | First-Party Source | 2,720 | Multi-format report bundle generator |
| `src/vipym/reporting/renderers/markdown.py` | First-Party Source | 2,374 | Markdown report renderer |
| `src/vipym/reporting/renderers/latex.py` | First-Party Source | 2,892 | Publication-ready LaTeX table renderer |
| `src/vipym/reporting/plots/pareto_plots.py` | First-Party Source | 2,446 | Plotly interactive & matplotlib PNG plots |
| `src/vipym/aws/ec2_ephemeral.py` | First-Party Source | 2,609 | Ephemeral EC2 lifecycle manager |
| `src/vipym/aws/s3.py` | First-Party Source | 2,476 | S3 multipart chunked storage adapter |
| `src/vipym/aws/cloudwatch.py` | First-Party Source | 1,328 | CloudWatch metric emitter |
| `src/vipym/aws/iam.py` | First-Party Source | 1,310 | Least-privilege IAM policy generator |
| `src/vipym/security/threat_model.py` | First-Party Source | 1,416 | Threat model specification |
| `src/vipym/security/sanitizer.py` | First-Party Source | 593 | Environment variable credential scrubber |
| `src/vipym/security/sandbox.py` | First-Party Source | 512 | Unified secure execution facade |
| `src/vipym/utils/hardware.py` | First-Party Source | 1,993 | GPU topology, NVLink, and EFA detector |
| `tests/unit/test_config.py` | Test Suite | 1,276 | Unit tests for configuration schemas |
| `tests/unit/test_dag.py` | Test Suite | 1,048 | Unit tests for DAG topological sort |
| `tests/unit/test_pareto.py` | Test Suite | 2,039 | Unit tests for Pareto dominance calculations |
| `tests/unit/test_security.py` | Test Suite | 1,175 | Unit tests for security sandbox and AST parser |
| `tests/unit/test_state.py` | Test Suite | 1,680 | Unit tests for experiment state machine |
| `tests/contract/test_plugin_contracts.py` | Test Suite | 1,738 | Contract tests for plugin abstract base classes |
| `tests/integration/test_distillation.py` | Test Suite | 1,011 | Integration tests for distillation pipelines |
| `tests/smoke/test_e2e_smoke.py` | Test Suite | 1,796 | End-to-end pipeline smoke test |
| `tests/e2e/test_cli_workflow.py` | Test Suite | 1,477 | End-to-end Typer CLI workflow tests |
