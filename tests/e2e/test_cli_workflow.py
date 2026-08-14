"""End-to-end tests exercising the Typer CLI commands."""

from pathlib import Path
import tempfile
from typer.testing import CliRunner
from vipym.cli.main import app

runner = CliRunner()


def test_cli_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "ViPym" in result.output


def test_cli_doctor():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Doctor Diagnostic Report" in result.output


def test_cli_list_commands():
    res_models = runner.invoke(app, ["list-models"])
    assert res_models.exit_code == 0
    assert "Registered Model Adapters" in res_models.output

    res_compressors = runner.invoke(app, ["list-compressors"])
    assert res_compressors.exit_code == 0
    assert "Registered Compression Methods" in res_compressors.output

    res_evaluators = runner.invoke(app, ["list-evaluators"])
    assert res_evaluators.exit_code == 0
    assert "Registered Evaluation Benchmark Suites" in res_evaluators.output


def test_cli_validate_smoke_config():
    result = runner.invoke(app, ["validate", "--config", "configs/experiments/smoke_test.yaml"])
    assert result.exit_code == 0
    assert "Configuration is valid" in result.output


def test_cli_inspect_kimi_k3():
    result = runner.invoke(app, ["inspect-model", "--model", "moonshotai/Kimi-K3"])
    assert result.exit_code == 0
    assert "2800.00B" in result.output
    assert "104.00B" in result.output
