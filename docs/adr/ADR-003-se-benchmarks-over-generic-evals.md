# ADR-003: Software Engineering Benchmarks over Generic LLM Evaluations

## Status
Accepted

## Context
Standard LLM evaluation suites (such as MMLU, GSM8K, ARC, or HellaSwag) measure general knowledge and basic reasoning. However, enterprise developer productivity tools (copilots, automated code review, refactoring agents) demand rigorous software engineering capabilities:
- Accurate syntax generation (no hallucinated semicolons, brackets, or invalid ASTs).
- Correct API usage across diverse libraries (Pandas, PyTorch, FastAPI, Boto3).
- Strict adherence to diff formats in multi-file refactoring tools (Aider, Cursor).
- Solving end-to-end repository GitHub issues with full pytest suite validation (SWE-bench).

Aggressive quantization frequently preserves surface prose fluency while degrading subtle programming precision (e.g. indentation, off-by-one indexing, type errors).

## Decision
ViPym prioritizes **Software Engineering Benchmarks as the Primary Optimization Objective**:
1. **Core Coding Benchmarks**:
   - `HumanEval` & `HumanEval+`: Algorithmic correctness with 80x test amplification.
   - `MBPP` & `MBPP+`: Python programming challenges.
   - `BigCodeBench`: Multi-library API composition across 163 standard packages.
   - `Aider Multi-File Edit`: Evaluating multi-file patch generation and diff formatting.
   - `SWE-bench Lite`: Real-world end-to-end software engineering issue resolution on GitHub repositories.
2. **Deterministic Sandboxed Execution**: All evaluation runs untrusted generated code within isolated gVisor / containerized sandboxes with total network lockdown (`--network=none`), non-executable filesystems, and strict timeouts.

## Consequences
### Positive
- Prevents deploying compressed models that pass generic QA benchmarks but fail in developer code generation pipelines.
- Direct alignment between compression research and enterprise developer ROI.

### Negative
- Sandbox execution of full test suites (SWE-bench, BigCodeBench) requires more compute and execution time than simple log-likelihood perplexity checks.
