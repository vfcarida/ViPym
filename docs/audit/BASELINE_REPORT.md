# ViPym — Baseline Execution & Quality Report

**Baseline Date:** 2026-08-14  
**Python Runtime:** 3.12.10 (win32)  
**Host Platform:** Windows 10 / 11 AMD64  
**Commit SHA:** `5afbdac`  

---

## 1. Initial Baseline Command Executions

| Command | Category | Status | Exit Code | Notes |
|---|---|---|---|---|
| `git status` | VCS | PASS | 0 | Clean worktree on branch `main` |
| `python -m pytest tests/ -v` | Test Suite | PASS | 0 | 25 passed in 43.53s (3 deprecation/layout warnings) |
| `ruff check .` | Static Linting | FAIL | 1 | 525 lint errors (import sorting `I001`, unused imports `F401`, loop variable names `B007`) |
| `ruff format --check .` | Formatting | FAIL | 1 | 53 files would be reformatted |
| `vipym --help` | CLI Entrypoint | PASS | 0 | Help menu rendered with all 14 subcommands |
| `vipym doctor` | Diagnostics | PASS | 0 | Diagnostic table rendered across Python, Torch, Docker, Disk |
| `vipym inspect-model --model moonshotai/Kimi-K3` | Model Introspection | PASS | 0 | Output verified: 2.8T parameters, 104B active, 896 routed experts |

---

## 2. Pre-Existing Defects & Material Observations

1. **Linting & Formatting Debt (`F-001`):** 525 violations in `ruff check` and 53 unformatted files in `ruff format`. Primarily un-sorted imports (`I001`), unused import statements (`F401`), and loop variable naming (`B007`).
2. **Subpackage Redundancy / Coexistence (`F-002`):** Legacy `src/vipym/core/` modules (`config.py`, `manifest.py`, `runner.py`, `constants.py`, `exceptions.py`) and `src/vipym/cloud/` coexist alongside modular packages `src/vipym/config/`, `src/vipym/experiments/`, and `src/vipym/aws/`. These should be unified with backward-compatible aliases to avoid drift.
3. **Matplotlib Tight Layout Warning (`F-003`):** In `src/vipym/reporting/plots/pareto_plots.py`, `plt.tight_layout()` emits a warning due to axis decoration width constraints on 2x2 subplot grids.
4. **License Consistency (`F-004`):** `pyproject.toml` and `LICENSE` state Apache-2.0, while prompt metadata mentions MIT. Needs explicit documentation and decision record.
5. **Static Type Checking (`F-005`):** MyPy needs to be configured with explicit per-module overrides and cached execution to avoid timeouts during full-tree scans.
