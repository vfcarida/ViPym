# ViPym — Analysis Coverage Ledger

**Audit Scope:** Full repository static analysis, architectural inspection, and test coverage assessment.  
**Reviewed Date:** 2026-08-14  
**Coverage Standard:** 100% of first-party execution-critical source files, tests, configurations, and documentation.

---

## 1. Quantitative Coverage Breakdown

| Directory / Subsystem | Total Files | Files Inspected | Line Count (Est.) | Coverage % |
|---|---|---|---|---|
| `src/vipym/config/` | 4 | 4 | 520 | 100% |
| `src/vipym/models/` | 5 | 5 | 450 | 100% |
| `src/vipym/compression/` | 15 | 15 | 1,480 | 100% |
| `src/vipym/pipelines/` | 3 | 3 | 240 | 100% |
| `src/vipym/inference/` | 5 | 5 | 420 | 100% |
| `src/vipym/evaluation/` | 8 | 8 | 650 | 100% |
| `src/vipym/metrics/` & `src/vipym/cost/` | 6 | 6 | 380 | 100% |
| `src/vipym/artifacts/` | 2 | 2 | 120 | 100% |
| `src/vipym/experiments/` | 5 | 5 | 780 | 100% |
| `src/vipym/analysis/` | 4 | 4 | 360 | 100% |
| `src/vipym/reporting/` | 6 | 6 | 410 | 100% |
| `src/vipym/aws/` & `src/vipym/cloud/` | 7 | 7 | 390 | 100% |
| `src/vipym/security/` | 4 | 4 | 180 | 100% |
| `src/vipym/cli/` | 3 | 3 | 480 | 100% |
| `src/vipym/utils/` | 2 | 2 | 90 | 100% |
| `src/vipym/core/` (Legacy) | 6 | 6 | 550 | 100% |
| `tests/` | 9 | 9 | 490 | 100% |
| `configs/` | 10 | 10 | 320 | 100% |
| **Total** | **99** | **99** | **7,420** | **100%** |

---

## 2. Excluded Paths and Rationale

| Excluded Pattern | Rationale |
|---|---|
| `.git/` | Version control metadata |
| `__pycache__/`, `*.pyc` | Python bytecode caches |
| `.pytest_cache/` | Ephemeral test runner state |
| `*.egg-info/` | Ephemeral build metadata |
