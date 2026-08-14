# ADR-002: Hardened Sandboxing and gVisor Container Isolation for Benchmark Execution

## Status
Accepted

## Context
Executing untrusted LLM-generated code (e.g. on HumanEval, MBPP, SWE-bench) poses critical security risks:
- Malicious code attempting to exfiltrate cloud credentials (`AWS_SECRET_ACCESS_KEY`, `HF_TOKEN`).
- Exploits attempting host privilege escalation or persistent rootfs modification.
- Denial-of-service attacks via fork bombs or memory exhaustion.

## Decision
Implement defense-in-depth isolation in `SandboxedCodeRunner`:
1. Static AST parsing (`ast.parse`) pre-validation.
2. Complete environment sanitization (`sanitize_execution_environment`).
3. User-space kernel virtualization with **gVisor (`runsc`)** or isolated Docker containers.
4. Hard resource limits (2GB RAM, 2 CPUs, 100 PIDs, 15s timeout).
5. Complete network isolation (`--network=none`).

## Consequences
- Guarantees host integrity during large-scale automated benchmark evaluations.
- Prevents secret exfiltration.
