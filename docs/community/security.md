# Security Policy & Vulnerability Reporting

The ViPym project takes the security of AI model compression and code evaluation pipelines seriously. Evaluating untrusted model-generated code introduces unique attack vectors (e.g., arbitrary code execution, filesystem tampering, and credential exfiltration). We appreciate the security community's efforts to responsibly disclose vulnerabilities.

---

## 1. Supported Versions

Security updates are actively provided for the following versions:

| Version | Supported | Notes |
| :--- | :---: | :--- |
| **0.1.x** | **Yes** | Current active release line |
| **< 0.1.0** | **No** | Deprecated / pre-release versions |

---

## 2. Reporting a Vulnerability

Please **do not** open public GitHub issues or discussions for security-sensitive vulnerabilities.

### Preferred Method: GitHub Security Advisories
Report the issue privately through GitHub's Coordinated Vulnerability Disclosure:
👉 **[Open a Private Security Advisory](https://github.com/vfcarida/ViPym/security/advisories/new)**

### Alternative Method
If GitHub Advisories are unavailable, email **`vfcarida@gmail.com`** with:
- A descriptive title and vulnerability severity assessment (CVSS if applicable).
- Step-by-step reproduction instructions or a minimal Proof-of-Concept (PoC).
- Affected ViPym version, operating system, and container runtime (Docker / gVisor).

### Response Timeline & SLAs
- **Initial Acknowledgment:** Within **48 business hours**.
- **Triage & Assessment:** Within **5 business days**.
- **Fix & Advisory Release:** Coordinated disclosure typically within **30 days** of validation.

---

## 3. Sandboxing & Threat Model

ViPym's primary defense-in-depth barrier is the sandboxed code execution engine ([`docker_sandbox.py`](https://github.com/vfcarida/ViPym/blob/main/src/vipym/evaluation/sandbox/docker_sandbox.py)):

1. **Mandatory Double Opt-in:** Degraded execution (bare subprocess) is strictly forbidden unless **both** `allow_unsafe_execution: true` is configured in the recipe AND the environment variable `VIPYM_ALLOW_UNSAFE=1` is explicitly exported.
2. **Container Isolation:** Untrusted benchmark code must run inside Docker or gVisor (`runsc`) containers with:
   - `--network none` (complete network interface lockdown).
   - `--read-only` root filesystem.
   - Resource quotas (CPU limits, memory caps, PID process count limits).
   - Dropped capabilities (`--cap-drop ALL`).
3. **AST Pre-Validation:** Syntax validation (`ast.parse`) is performed prior to container instantiation to eliminate syntax-based denial-of-service.

### Out-of-Scope Items
- Exploits requiring explicit enablement of double opt-in degraded mode (`VIPYM_ALLOW_UNSAFE=1`).
- Social engineering against project maintainers.
- Denial of service on local single-user CLI executions.

---

## 4. Safe Harbor

Security researchers operating within the scope of this policy and adhering to coordinated disclosure will not face legal action from ViPym maintainers.
