"""Unit tests for Side-by-Side Model Battle Playground in ViPym Studio."""

from pathlib import Path

from fastapi.testclient import TestClient

from vipym.studio.app import create_studio_app
from vipym.studio.auth import SecurityConfig, TokenValidator


def test_studio_battle_endpoint_json(tmp_path: Path) -> None:
    """Verify non-streaming POST /api/inference/battle response schema and metrics."""
    sec_cfg = SecurityConfig(auth_token="battle-token", read_only=False)
    validator = TokenValidator(expected_token="battle-token")
    app = create_studio_app(
        artifacts_dir=tmp_path,
        security_config=sec_cfg,
        token_validator=validator,
    )
    client = TestClient(app)

    payload = {
        "prompt": "def fibonacci(n):",
        "baseline_model": "meta-llama/Llama-3-8B",
        "compressed_model": "meta-llama/Llama-3-8B-AWQ",
        "stream": False,
    }

    response = client.post(
        "/api/inference/battle",
        json=payload,
        headers={"Authorization": "Bearer battle-token"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["prompt"] == "def fibonacci(n):"
    assert "baseline" in data
    assert "compressed" in data
    assert "battle_comparison" in data

    # Verify baseline & compressed telemetry
    assert data["baseline"]["peak_vram_gb"] > data["compressed"]["peak_vram_gb"]
    assert data["baseline"]["total_duration_ms"] > data["compressed"]["total_duration_ms"]
    assert data["compressed"]["cost_per_1m_tokens"] < data["baseline"]["cost_per_1m_tokens"]

    # Verify comparison ratios
    cmp = data["battle_comparison"]
    assert "x" in cmp["latency_speedup"]
    assert "x" in cmp["vram_reduction"]
    assert "%" in cmp["operational_cost_savings_pct"]
    assert 0.0 <= cmp["token_similarity_score"] <= 1.0
    assert cmp["winner"] in {"compressed", "baseline"}


def test_studio_battle_endpoint_streaming(tmp_path: Path) -> None:
    """Verify dual SSE streaming in POST /api/inference/battle."""
    sec_cfg = SecurityConfig(auth_token="battle-token", read_only=False)
    validator = TokenValidator(expected_token="battle-token")
    app = create_studio_app(
        artifacts_dir=tmp_path,
        security_config=sec_cfg,
        token_validator=validator,
    )
    client = TestClient(app)

    payload = {
        "prompt": "def quicksort(arr):",
        "stream": True,
    }

    response = client.post(
        "/api/inference/battle",
        json=payload,
        headers={"Authorization": "Bearer battle-token"},
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "baseline" in response.text
    assert "compressed" in response.text
    assert "finish" in response.text


def test_studio_battle_auth_rejection(tmp_path: Path) -> None:
    """Verify 401 response on POST /api/inference/battle when token is missing or invalid."""
    sec_cfg = SecurityConfig(auth_token="secret-gate-token", read_only=False)
    validator = TokenValidator(expected_token="secret-gate-token")
    app = create_studio_app(
        artifacts_dir=tmp_path,
        security_config=sec_cfg,
        token_validator=validator,
    )
    client = TestClient(app)

    # Missing token
    resp_no_token = client.post("/api/inference/battle", json={"prompt": "test"})
    assert resp_no_token.status_code == 401

    # Invalid token
    resp_bad_token = client.post(
        "/api/inference/battle",
        json={"prompt": "test"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp_bad_token.status_code == 401


def test_studio_battle_read_only_rejection(tmp_path: Path) -> None:
    """Verify 403 Forbidden response when server is configured with read_only=True."""
    sec_cfg = SecurityConfig(auth_token="read-token", read_only=True)
    validator = TokenValidator(expected_token="read-token")
    app = create_studio_app(
        artifacts_dir=tmp_path,
        security_config=sec_cfg,
        token_validator=validator,
    )
    client = TestClient(app)

    resp = client.post(
        "/api/inference/battle",
        json={"prompt": "test"},
        headers={"Authorization": "Bearer read-token"},
    )
    assert resp.status_code == 403
    assert resp.json()["read_only"] is True
