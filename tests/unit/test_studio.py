"""Unit and integration tests for ViPym Studio REST API & Server."""

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from vipym.studio.server import start_studio_server


@pytest.fixture(scope="module")
def studio_server(tmp_path_factory: pytest.TempPathFactory):
    """Spin up an ephemeral ViPym Studio test server on an ephemeral port."""
    artifacts_dir = tmp_path_factory.mktemp("artifacts")

    # Create a mock experiment
    exp_dir = artifacts_dir / "test-exp-001"
    exp_dir.mkdir()
    (exp_dir / "manifest.json").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-08-14T12:00:00Z",
                "duration_seconds": 45.2,
                "total_cost_usd": 0.1234,
            }
        ),
        encoding="utf-8",
    )
    (exp_dir / "state.json").write_text(json.dumps({"state": "REPORT_COMPLETED"}), encoding="utf-8")
    (exp_dir / "results.json").write_text(
        json.dumps([{"config_id": "baseline", "pass_at_1": 0.85, "peak_vram_gb": 80.0}]),
        encoding="utf-8",
    )

    server = start_studio_server(host="127.0.0.1", port=18765, artifacts_dir=artifacts_dir)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.3)

    yield "http://127.0.0.1:18765"

    server.shutdown()
    server.server_close()


def test_studio_status(studio_server: str) -> None:
    """Test GET /api/status endpoint."""
    with urllib.request.urlopen(f"{studio_server}/api/status") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "online"
        assert data["vipym_version"] == "0.1.0"


def test_studio_experiments_api(studio_server: str) -> None:
    """Test GET /api/experiments endpoint."""
    with urllib.request.urlopen(f"{studio_server}/api/experiments") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert len(data) == 1
        assert data[0]["experiment_id"] == "test-exp-001"
        assert data[0]["state"] == "REPORT_COMPLETED"


def test_studio_recipes_api(studio_server: str) -> None:
    """Test GET /api/recipes endpoint."""
    with urllib.request.urlopen(f"{studio_server}/api/recipes") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert len(data) >= 4
        ids = [r["recipe_id"] for r in data]
        assert "kimi_k3_software_engineering_matrix" in ids


def test_studio_models_inspect_api(studio_server: str) -> None:
    """Test GET /api/models/inspect endpoint."""
    with urllib.request.urlopen(
        f"{studio_server}/api/models/inspect?model_id=moonshotai/Kimi-K3"
    ) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["total_parameters"] == 2800000000000
        assert data["active_parameters"] == 104000000000
        assert data["num_experts"] == 896


def test_studio_validate_api(studio_server: str) -> None:
    """Test POST /api/validate endpoint with valid YAML."""
    yaml_payload = {
        "yaml": """
experiment_id: test-valid-001
seed: 42
model:
  id: "HuggingFaceTB/SmolLM-135M"
compression_pipeline:
  - stage_id: "s1"
    method: "awq"
    scheme: "W4A16"
    dependencies: []
serving:
  backend: "hf"
evaluation:
  suites: ["humaneval"]
"""
    }
    req = urllib.request.Request(
        f"{studio_server}/api/validate",
        data=json.dumps(yaml_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["valid"] is True
        assert data["experiment_id"] == "test-valid-001"


def test_studio_spa_index(studio_server: str) -> None:
    """Test that GET / returns the HTML single-page app."""
    with urllib.request.urlopen(f"{studio_server}/") as resp:
        assert resp.status == 200
        content = resp.read().decode("utf-8")
        assert "ViPym Studio" in content
        assert "Pareto Frontier" in content
