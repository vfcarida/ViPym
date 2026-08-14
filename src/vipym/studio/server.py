"""ViPym Studio HTTP & REST API Server."""

import json
import mimetypes
import shutil
import sys
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any

from vipym.config.schema import ViPymExperimentConfig
from vipym.core.logger import get_logger
from vipym.models.registry import ModelRegistry
from vipym.recipes.registry import RecipeRegistry

logger = get_logger(__name__)


class StudioAPIHandler(SimpleHTTPRequestHandler):
    """Handles static web assets and REST API requests for ViPym Studio."""

    def __init__(
        self,
        *args: Any,
        directory: str | None = None,
        artifacts_dir: Path = Path("./artifacts"),
        **kwargs: Any,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.static_dir = Path(__file__).parent / "static"
        super().__init__(*args, directory=str(self.static_dir), **kwargs)

    def do_GET(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        # REST API Routes
        if path.startswith("/api/"):
            self.handle_api_get(path, query)
            return

        # Serve SPA index for root
        if path == "/" or path == "/index.html":
            self.serve_file(self.static_dir / "index.html", "text/html")
            return

        # Serve static assets
        local_file = self.static_dir / path.lstrip("/")
        if local_file.exists() and local_file.is_file():
            mime_type, _ = mimetypes.guess_type(str(local_file))
            self.serve_file(local_file, mime_type or "application/octet-stream")
            return

        # Fallback to SPA index
        self.serve_file(self.static_dir / "index.html", "text/html")

    def do_POST(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"

        try:
            payload = json.loads(post_data) if post_data else {}
        except Exception:
            payload = {}

        if path == "/api/validate":
            self.handle_api_validate(payload)
        else:
            self.send_json_response({"error": f"Endpoint POST '{path}' not found"}, status=404)

    def handle_api_get(self, path: str, query: dict[str, list[str]]) -> None:
        if path == "/api/status":
            self.send_json_response(
                {
                    "status": "online",
                    "vipym_version": "0.1.0",
                    "artifacts_dir": str(self.artifacts_dir),
                }
            )
        elif path == "/api/experiments":
            self.send_json_response(self.list_experiments())
        elif path.startswith("/api/experiments/"):
            exp_id = path.replace("/api/experiments/", "").strip("/")
            self.send_json_response(self.get_experiment_details(exp_id))
        elif path == "/api/recipes":
            recipes = RecipeRegistry.list_recipes()
            res = [
                {
                    "recipe_id": v.recipe_id,
                    "name": v.name,
                    "model": v.target_model_family,
                    "domain": v.domain,
                    "compression_ratio": v.expected_compression_ratio,
                    "quality_retention": v.expected_quality_retention,
                    "hardware": v.hardware_target,
                    "description": v.description,
                    "tags": v.tags,
                }
                for v in recipes.values()
            ]
            self.send_json_response(res)
        elif path == "/api/models/inspect":
            model_id = query.get("model_id", ["moonshotai/Kimi-K3"])[0]
            self.send_json_response(self.inspect_model(model_id))
        elif path == "/api/doctor":
            # Return system diagnostics
            import torch

            cuda_ok = torch.cuda.is_available()
            total, used, free = shutil.disk_usage(".")
            self.send_json_response(
                {
                    "python_version": sys.version.split()[0],
                    "cuda_available": cuda_ok,
                    "gpu_count": torch.cuda.device_count() if cuda_ok else 0,
                    "gpu_devices": [
                        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
                    ]
                    if cuda_ok
                    else [],
                    "disk_free_gb": round(free / (1024**3), 1),
                    "docker_available": shutil.which("docker") is not None,
                }
            )
        else:
            self.send_json_response({"error": f"API endpoint '{path}' not found"}, status=404)

    def handle_api_validate(self, payload: dict[str, Any]) -> None:
        import yaml

        try:
            yaml_str = payload.get("yaml", "")
            data = yaml.safe_load(yaml_str)
            cfg = ViPymExperimentConfig(**data)
            self.send_json_response(
                {
                    "valid": True,
                    "experiment_id": cfg.experiment_id,
                    "stages_count": len(cfg.compression_pipeline),
                    "suites": cfg.evaluation.suites,
                    "message": "Configuration is valid!",
                }
            )
        except Exception as e:
            self.send_json_response({"valid": False, "error": str(e)}, status=400)

    def list_experiments(self) -> list[dict[str, Any]]:
        results = []
        if not self.artifacts_dir.exists():
            return results

        for exp_dir in self.artifacts_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            exp_id = exp_dir.name
            manifest_file = exp_dir / "manifest.json"
            state_file = exp_dir / "state.json"
            results_file = exp_dir / "results.json"

            state = "UNKNOWN"
            if state_file.exists():
                try:
                    with open(state_file, encoding="utf-8") as f:
                        state = json.load(f).get("state", "UNKNOWN")
                except Exception:
                    pass

            manifest_data = {}
            if manifest_file.exists():
                try:
                    with open(manifest_file, encoding="utf-8") as f:
                        manifest_data = json.load(f)
                except Exception:
                    pass

            points_count = 0
            if results_file.exists():
                try:
                    with open(results_file, encoding="utf-8") as f:
                        points_count = len(json.load(f))
                except Exception:
                    pass

            results.append(
                {
                    "experiment_id": exp_id,
                    "state": state,
                    "timestamp": manifest_data.get("timestamp_utc", ""),
                    "duration_sec": manifest_data.get("duration_seconds", 0.0),
                    "cost_usd": manifest_data.get("total_cost_usd", 0.0),
                    "pareto_points_count": points_count,
                }
            )

        return results

    def get_experiment_details(self, exp_id: str) -> dict[str, Any]:
        exp_dir = self.artifacts_dir / exp_id
        if not exp_dir.exists():
            return {"error": f"Experiment '{exp_id}' not found"}

        data: dict[str, Any] = {"experiment_id": exp_id}

        for fname in [
            "manifest.json",
            "state.json",
            "metrics.json",
            "results.json",
            "experiment.json",
            "artifacts.json",
        ]:
            fpath = exp_dir / fname
            if fpath.exists():
                try:
                    with open(fpath, encoding="utf-8") as f:
                        data[fname.replace(".json", "")] = json.load(f)
                except Exception:
                    pass

        # Load markdown report if available
        report_md = exp_dir / "reports" / "report.md"
        if report_md.exists():
            data["report_markdown"] = report_md.read_text(encoding="utf-8")

        return data

    def inspect_model(self, model_id: str) -> dict[str, Any]:
        try:
            adapter = ModelRegistry.get(model_id)
        except Exception:
            adapter = ModelRegistry.get("hf")

        meta = adapter.inspect_metadata(model_id)
        return {
            "model_id": meta.model_id,
            "total_parameters": meta.total_parameters,
            "total_parameters_b": round(meta.total_parameters / 1e9, 2),
            "active_parameters": meta.active_parameters,
            "active_parameters_b": round(meta.active_parameters / 1e9, 2),
            "architecture": str(meta.architecture_type),
            "num_layers": meta.num_layers,
            "hidden_size": meta.hidden_size,
            "num_attention_heads": meta.num_attention_heads,
            "num_experts": meta.num_experts,
            "num_selected_experts": meta.num_selected_experts,
            "context_window": meta.context_window,
            "native_dtypes": [str(d) for d in meta.native_dtypes],
        }

    def serve_file(self, file_path: Path, content_type: str) -> None:
        if not file_path.exists():
            self.send_error(404, f"File {file_path.name} not found")
            return
        content = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_json_response(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def start_studio_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    artifacts_dir: Path | str = "./artifacts",
) -> HTTPServer:
    """Instantiate and start the ViPym Studio web server."""
    artifacts_path = Path(artifacts_dir).resolve()

    def handler_factory(*args: Any, **kwargs: Any) -> StudioAPIHandler:
        return StudioAPIHandler(*args, artifacts_dir=artifacts_path, **kwargs)

    server = HTTPServer((host, port), handler_factory)
    logger.info(
        f"ViPym Studio UI & REST API running at: [bold cyan]http://{host}:{port}[/bold cyan]"
    )
    return server
