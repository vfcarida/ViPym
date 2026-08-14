# ViPym — Prioritized Engineering Implementation Plan

**Plan Date:** 2026-08-14  
**Status:** Approved for Implementation (Mode: `PLAN_THEN_IMPLEMENT`)  
**Target Quality Standard:** Production-grade open-source framework (Zero lint errors, 100% passing tests, rigorous typing, complete documentation).

---

## 1. Prioritized Change Matrix

| Change ID | Finding IDs | Severity / Priority | Target Subsystem | Action & Description |
|---|---|---|---|---|
| `C-001` | `F-001` | P0 / High | Repository-wide | Run automated code formatting (`ruff format`) and lint fixes (`ruff check --fix`) across all files. Fix remaining manual lint violations (unused imports, loop variables). |
| `C-002` | `F-002` | P1 / Medium | `src/vipym/core/`, `src/vipym/cloud/` | Refactor legacy modules to re-export from new modular subpackages (`vipym.config`, `vipym.experiments`, `vipym.aws`) ensuring full backward compatibility and zero code duplication. |
| `C-003` | `F-003` | P2 / Low | `src/vipym/reporting/plots/` | Refactor Matplotlib subplots layout in `pareto_plots.py` to prevent `tight_layout` warnings on 2x2 grid visualizations. |
| `C-004` | `F-004` | P2 / Low | Governance & Legal | Create Architecture Decision Record `ADR-004` documenting Apache-2.0 and MIT compatibility considerations. |
| `C-005` | `F-005` | P1 / Medium | `pyproject.toml`, Testing | Optimize MyPy configuration with precise module inclusions and build a comprehensive test strategy document (`docs/testing/TEST_STRATEGY.md`). |

---

## 2. Granular Implementation Steps

### Phase 6.1: Code Hygiene & Formatting (`C-001`)
- Run `ruff format .`
- Run `ruff check --fix .`
- Resolve remaining linting errors (B007, F401, UP035) manually.

### Phase 6.2: Subpackage Consolidation & Backward Compatibility (`C-002`)
- Update `src/vipym/core/` and `src/vipym/cloud/` to clean proxy re-exports.
- Ensure all public classes, exceptions, and constants import identically across legacy and new paths.

### Phase 6.3: Plotting & Telemetry Polish (`C-003`)
- Clean up figure margins and layout parameters in `src/vipym/reporting/plots/pareto_plots.py`.

### Phase 6.4: Documentation, ADRs & Traceability (`C-004`, `C-005`)
- Author ADRs (`ADR-001`, `ADR-002`, `ADR-003`, `ADR-004`).
- Generate `docs/audit/TRACEABILITY_MATRIX.md`.
- Generate `docs/testing/TEST_STRATEGY.md`.
