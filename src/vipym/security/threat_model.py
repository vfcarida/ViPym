"""Threat model and security constraints for untrusted LLM-generated code execution."""

THREAT_MODEL_SPECIFICATION = """
# ViPym Security Threat Model

## 1. Adversarial Code Capabilities
Generated programs produced by LLMs (or crafted benchmark payloads) can potentially:
- Attempt to read AWS credentials from environment variables (`AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`).
- Attempt to access host filesystems, reading SSH keys or modifying operating system binaries.
- Attempt outbound socket communication to exfiltrate data or initiate command-and-control botnet traffic.
- Attempt denial-of-service via resource exhaustion (fork-bombs, memory leaks, infinite while loops).

## 2. ViPym Defense-in-Depth Controls
1. **Static AST Analysis:** All code is parsed with `ast.parse` prior to execution.
2. **User-Space Kernel Isolation:** Code executes within **gVisor (`runsc`)** or hardened ephemeral containers.
3. **Zero Network:** All containers run with `--network=none`.
4. **Environment Sanitization:** Environment variables passed to untrusted processes are strictly scrubbed.
5. **Read-Only Root Filesystem:** Filesystem is mounted read-only, with an isolated ephemeral `/tmp` mounted with `noexec,nosuid,nodev`.
6. **Hard Resource Limits:** RLIMIT_CPU, RLIMIT_AS (Memory 2GB), and RLIMIT_NPROC (100 PIDs) enforced.
"""


def get_threat_model_summary() -> str:
    return THREAT_MODEL_SPECIFICATION
