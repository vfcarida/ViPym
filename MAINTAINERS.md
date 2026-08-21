# ViPym Maintainers & Governance Guide

This document outlines the governance structure, maintainer responsibilities, and decision-making processes for the **ViPym** open-source project.

---

## 👥 Current Maintainers

| Name / GitHub Handle | Role | Primary Focus Area |
| :--- | :--- | :--- |
| **[@vfcarida](https://github.com/vfcarida)** | Project Lead / Core Maintainer | Architecture, Compression DAG, Orchestration |
| **ViPym Core Contributors** | Maintainers | Evaluation Suites, Studio SPA, Cloud Serving |

---

## 🛠️ Maintainer Responsibilities

1. **Code Review & Quality Gate**:
   - Ensure all Pull Requests meet architectural standards, include unit/contract tests, and keep total test coverage $\ge 90\%$.
   - Maintain 100% green status across multi-platform CI pipelines.
2. **Security & Sandbox Isolation**:
   - Review changes to `src/vipym/evaluation/sandbox/` and ensure gVisor / AST security controls are never bypassed without explicit flags.
   - Address vulnerability disclosures promptly in accordance with [`SECURITY.md`](SECURITY.md).
3. **Release Management**:
   - Maintain semantic versioning (`MAJOR.MINOR.PATCH`).
   - Keep `CHANGELOG.md` updated with every release.
4. **Community & Triage**:
   - Triage newly opened GitHub Issues and Pull Requests within 48 business hours.

---

## 🤝 Decision Making & ADRs

- **Architectural Changes**: Any major change to the DAG engine, FSM states, plugin registries, or serialization formats requires an **Architecture Decision Record (ADR)** submitted under `docs/adr/` and approved by at least one core maintainer.
- **Breaking Changes**: Breaking configuration changes must follow a deprecation cycle of at least one minor release with explicit `DeprecationWarning` logs.
