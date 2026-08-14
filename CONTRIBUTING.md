# Contributing to ViPym

Thank you for your interest in contributing to ViPym!

## Development Setup

```bash
git clone https://github.com/vfcarida/ViPym.git
cd ViPym
pip install -e ".[all]"
pre-commit install
```

## Code Quality Standards

* **Formatting:** `ruff format .`
* **Linting:** `ruff check .`
* **Type Checking:** `mypy src/`
* **Testing:** `pytest tests/unit tests/contract tests/smoke`

## Adding a New Compression Plugin

1. Subclass `CompressionMethod` in `src/vipym/compression/`.
2. Implement `name`, `get_capabilities()`, `validate_applicability()`, and `compress()`.
3. Register using `CompressionRegistry.register("my_method", MyMethodClass)`.
4. Add unit and contract tests in `tests/`.

## Adding a New Evaluation Suite

1. Subclass `EvaluationSuite` in `src/vipym/evaluation/suites/`.
2. Implement `name`, `version`, `load_tasks()`, `format_prompt()`, and `evaluate_response()`.
3. Register using `EvaluationRegistry.register("my_suite", MySuiteClass)`.
