.PHONY: help install install-dev test test-unit test-integration test-contract lint format doctor demo clean

help:
	@echo "ViPym Development Automation Commands:"
	@echo "  make install           Install core dependencies"
	@echo "  make install-dev       Install development dependencies and pre-commit hooks"
	@echo "  make test              Run all unit and integration tests"
	@echo "  make test-unit         Run unit tests only"
	@echo "  make test-integration  Run integration tests only"
	@echo "  make lint              Run ruff linter"
	@echo "  make format            Run ruff code formatter"
	@echo "  make doctor            Run ViPym environment readiness check"
	@echo "  make demo              Run 5-minute CPU quickstart demo"
	@echo "  make clean             Remove test, build, and bytecode caches"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,all]"
	pre-commit install

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-contract:
	pytest tests/contract/ -v

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

doctor:
	vipym doctor

demo:
	vipym run recipes/quick-demo-gpt2.yaml --output results/

clean:
	python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for pat in ['**/__pycache__', '**/.pytest_cache', '**/.ruff_cache', '**/*.egg-info', 'dist', 'build', '**/.mypy_cache', 'site'] for p in pathlib.Path('.').glob(pat)]; [p.unlink(missing_ok=True) for p in pathlib.Path('.').glob('**/*.pyc')]"

