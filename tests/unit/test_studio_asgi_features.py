"""Unit tests for ViPym Studio ASGI advanced features (DAG Visualizer, MoE Matrix, Playground Streaming)."""

from __future__ import annotations

import json
import threading
import time
import urllib.request

import pytest

from vipym.studio.server import start_studio_server


@pytest.fixture(scope="module")
def asgi_server(tmp_path_factory: pytest.TempPathFactory):
    tmp = tmp_path_factory.mktemp("asgi_studio")
    artifacts_dir = tmp / "artifacts"
    artifacts_dir.mkdir()

    # Create dummy experiment
    exp_dir = artifacts_dir / "exp_dag_moe"
    exp_dir.mkdir()
    (exp_dir / "manifest.json").write_text(
        json.dumps({"timestamp_utc": "2026-09-11T12:00:00Z"}), encoding="utf-8"
    )
    (exp_dir / "reports").mkdir()
    (exp_dir / "reports" / "report.md").write_text(
        "# Benchmark Report\n\nQuality retention 98.5%.", encoding="utf-8"
    )

    token = "test-token-asgi-123"
    server = start_studio_server(
        host="127.0.0.1",
        port=19991,
        artifacts_dir=artifacts_dir,
        token=token,
    )
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.5)

    yield {
        "url": "http://127.0.0.1:19991",
        "token": token,
        "artifacts_dir": artifacts_dir,
        "server": server,
    }

    server.shutdown()
    server.server_close()


def test_dag_graph_endpoint(asgi_server: dict):
    """Verify /api/dag/graph returns valid topological node and edge data."""
    url = f"{asgi_server['url']}/api/dag/graph"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "nodes" in data
        assert "links" in data
        assert len(data["nodes"]) >= 5
        assert len(data["links"]) >= 4


def test_moe_matrix_endpoint(asgi_server: dict):
    """Verify /api/moe/matrix returns router expert correlation data."""
    url = f"{asgi_server['url']}/api/moe/matrix"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "correlation_matrix" in data
        assert "expert_labels" in data
        assert len(data["correlation_matrix"]) == 16
        # Diagonal elements must equal 1.0 (self-correlation)
        for i in range(16):
            assert data["correlation_matrix"][i][i] == 1.0


def test_inference_playground_generate(asgi_server: dict):
    """Verify POST /api/inference/generate generates text responses."""
    url = f"{asgi_server['url']}/api/inference/generate"
    payload = json.dumps({"prompt": "def bubble_sort(arr):", "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {asgi_server['token']}",
        },
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "generated_text" in data
        assert "bubble_sort" in data["generated_text"]


def test_inference_playground_streaming_sse(asgi_server: dict):
    """Verify POST /api/inference/generate with stream=True returns SSE data."""
    url = f"{asgi_server['url']}/api/inference/generate"
    payload = json.dumps({"prompt": "def bubble_sort(arr):", "stream": True}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {asgi_server['token']}",
        },
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        assert "text/event-stream" in resp.headers.get("Content-Type", "")
        raw_sse = resp.read().decode("utf-8")
        assert "data:" in raw_sse
        assert "finish_reason" in raw_sse


def test_report_export_latex_and_markdown(asgi_server: dict):
    """Verify /api/reports/{id}/export returns LaTeX and Markdown formatted reports."""
    url_md = f"{asgi_server['url']}/api/reports/exp_dag_moe/export?format=markdown"
    req_md = urllib.request.Request(url_md)
    with urllib.request.urlopen(req_md) as resp:
        assert resp.status == 200
        content = resp.read().decode("utf-8")
        assert "Quality retention 98.5%" in content

    url_latex = f"{asgi_server['url']}/api/reports/exp_dag_moe/export?format=latex"
    req_latex = urllib.request.Request(url_latex)
    with urllib.request.urlopen(req_latex) as resp:
        assert resp.status == 200
        content = resp.read().decode("utf-8")
        assert "\\documentclass{article}" in content
