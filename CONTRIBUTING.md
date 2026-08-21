# Contributing to ViPym

Thank you for your interest in contributing to **ViPym**! We welcome contributions from researchers, machine learning engineers, and open-source developers to advance the state of LLM compression and benchmark evaluation.

---

## 1. Development Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git
- PyTorch 2.4+

### Clone & Install
```bash
git clone https://github.com/vfcarida/ViPym.git
cd ViPym

# Install ViPym in editable development mode with dev tools
pip install -e ".[dev]"

# Verify setup
vipym doctor
```

---

## 2. Adding a New Compression Method

All compression techniques inherit from `CompressionMethod` and register in `CompressionRegistry`:

```python
# src/vipym/compression/methods/my_method.py
from typing import Any
from pathlib import Path
import torch.nn as nn

from vipym.interfaces.compression import CompressionMethod, PluginCapability
from vipym.compression.registry import CompressionRegistry
from vipym.core.types import CompressionArtifact, SupportedDtype, ComputeArchitecture


@CompressionRegistry.register("my_method")
class MyCustomCompression(CompressionMethod):
    def __init__(self, bits: int = 4, **kwargs: Any) -> None:
        self.bits = bits
        self.kwargs = kwargs

    @property
    def name(self) -> str:
        return f"my_method_w{self.bits}"

    def get_capabilities(self) -> PluginCapability:
        return PluginCapability(
            supported_dtypes=[SupportedDtype.INT4, SupportedDtype.FP16],
            supported_architectures=[ComputeArchitecture.DENSE, ComputeArchitecture.MOE],
            requires_calibration=True,
            supports_moe=True,
            supported_runtimes=["vllm", "hf"],
        )

    def compress(
        self,
        model: nn.Module,
        tokenizer: Any,
        calibration_data: Any | None = None,
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> CompressionArtifact:
        # 1. Apply your custom compression algorithm
        # 2. Save weights and config
        out_path = Path(output_dir or "./compressed_model")
        out_path.mkdir(parents=True, exist_ok=True)

        return CompressionArtifact(
            output_path=out_path,
            format="compressed-tensors",
            compressed_size_bytes=1024,
            applied_methods=[self.name],
            metadata={"bits": self.bits},
        )
```

---

## 3. Adding a New Evaluation Benchmark Suite

All benchmark suites inherit from `EvaluationSuite` and register in `EvaluationRegistry`:

```python
# src/vipym/evaluation/suites/my_suite.py
from typing import Any
from vipym.interfaces.evaluation import EvaluationSuite, BenchmarkTask, BenchmarkTaskResult
from vipym.evaluation.registry import EvaluationRegistry


@EvaluationRegistry.register("my_suite")
class MyCustomEvaluationSuite(EvaluationSuite):
    @property
    def name(self) -> str:
        return "my_suite"

    @property
    def version(self) -> str:
        return "v1.0.0"

    def load_tasks(self, limit: int | None = None) -> list[BenchmarkTask]:
        # Return list of BenchmarkTask items
        return [
            BenchmarkTask(
                task_id="task_001",
                prompt="def add(a, b):\n    '''Return sum of a and b.'''\n",
                test_code="assert add(2, 3) == 5\nassert add(-1, 1) == 0",
                entry_point="add",
            )
        ][:limit]

    def evaluate_task(
        self, task: BenchmarkTask, generated_code: str, sandbox: Any
    ) -> BenchmarkTaskResult:
        # Execute in sandbox and return task result
        result = sandbox.run_code(task.prompt + generated_code + "\n" + task.test_code)
        return BenchmarkTaskResult(
            task_id=task.task_id,
            passed=result.exit_code == 0,
            compiled=result.compiled,
            execution_time_sec=result.duration_sec,
            error_message=result.stderr if result.exit_code != 0 else None,
        )
```

---

## 4. Code Style & Linting

We enforce strict formatting and linting using `ruff`:

```bash
# Check code style and imports
ruff check .

# Auto-fix fixable linter warnings
ruff check --fix .

# Format code
ruff format .
```

---

## 5. Testing Requirements

All contributions must include unit tests and maintain passing test suites:

```bash
# Run unit tests
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v
```

---

## 6. Pull Request Process

1. Fork the repository and create a feature branch (`git checkout -b feature/my-new-method`).
2. Implement your changes, documentation, and unit tests.
3. Verify that all tests pass (`pytest tests/ -v`) and ruff is clean (`ruff check .`).
4. Commit your changes with clear commit messages.
5. Push to your branch and open a Pull Request against `main`.
