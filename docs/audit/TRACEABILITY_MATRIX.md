# ViPym — Full Traceability Matrix

| Finding ID | External Ref ID | Planned Change ID | Description | Implementation Target | Verification Check ID | Status |
|---|---|---|---|---|---|---|
| `F-001` | `R-001`, `R-002` | `C-001` | Code formatting & lint hygiene across repository | Run `ruff format .` & `ruff check --fix .` | `V-001`, `V-002` | PLANNED |
| `F-002` | `R-001` | `C-002` | Subpackage consolidation and backward-compatible re-exports | `src/vipym/core/`, `src/vipym/cloud/` | `V-003` | PLANNED |
| `F-003` | `R-010` | `C-003` | Matplotlib tight layout warning mitigation on Pareto plots | `src/vipym/reporting/plots/pareto_plots.py` | `V-004` | PLANNED |
| `F-004` | `R-001` | `C-004` | Open-source licensing compatibility assessment | `docs/architecture/decisions/ADR-004-license-dual-compatibility.md` | `V-005` | PLANNED |
| `F-005` | `R-001` | `C-005` | MyPy type checking configuration and test strategy | `pyproject.toml`, `docs/testing/TEST_STRATEGY.md` | `V-006` | PLANNED |
