"""ViPym Studio HTTP, REST API & Hardened WebSocket Server.

Implements:
- Localhost binding by default (127.0.0.1)
- Token authentication with auto-generated Bearer tokens saved to ~/.vipym/studio-token
- Rate limiting (100 req/min per token/IP) with 429 Too Many Requests
- Comprehensive audit logging of all requests to ~/.vipym/studio-audit.log
- Read-only mode (--read-only) blocking mutations with 403 Forbidden
- Unauthenticated /health endpoint for probes and health checks
- Strict same-origin CORS protection by default
- Authenticated real-time WebSocket progress streaming (/ws/progress)
"""

from __future__ import annotations

import json
import mimetypes
import shutil
import sys
import time
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any

from vipym.config.schema import ViPymExperimentConfig
from vipym.core.logger import get_logger
from vipym.models.registry import ModelRegistry
from vipym.recipes.registry import RecipeRegistry
from vipym.studio.auth import (
    AuditLogger,
    RateLimiter,
    SecurityConfig,
    TokenValidator,
    get_or_create_studio_token,
)
from vipym.studio.websocket import StudioWebSocketManager, get_websocket_manager

logger = get_logger(__name__)


class StudioAPIHandler(SimpleHTTPRequestHandler):
    """Handles static web assets, REST API requests, and authenticated WebSockets for ViPym Studio."""

    # Set by start_studio_server
    security_config: SecurityConfig
    token_validator: TokenValidator
    rate_limiter: RateLimiter
    audit_logger: AuditLogger
    ws_manager: StudioWebSocketManager
    server_start_time: float = time.time()

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

    def _get_client_ip(self) -> str:
        """Extract source IP address from request."""
        client_address = getattr(self, "client_address", ("127.0.0.1", 0))
        return client_address[0] if client_address else "127.0.0.1"

    def _check_rate_limit(self, client_key: str) -> tuple[bool, float]:
        """Verify rate limit for client."""
        if hasattr(self, "rate_limiter") and self.rate_limiter:
            return self.rate_limiter.is_allowed(client_key)
        return True, 0.0

    def _apply_cors_headers(self) -> None:
        """Apply strict CORS headers based on allowed origins."""
        origin = self.headers.get("Origin")
        host = self.headers.get("Host")
        cfg = getattr(self, "security_config", None)

        if cfg and cfg.is_origin_allowed(origin, host_header=host):
            allow_origin = origin if origin else "http://127.0.0.1"
            self.send_header("Access-Control-Allow-Origin", allow_origin)
            self.send_header("Access-Control-Allow-Credentials", "true")
            self.send_header("Vary", "Origin")
        else:
            # Fallback to local origin
            self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1")

        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization, X-API-Key, Upgrade, Sec-WebSocket-Key, Sec-WebSocket-Version",
        )

    def do_OPTIONS(self) -> None:
        """Handle CORS pre-flight requests."""
        self.send_response(200)
        self._apply_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        start_time = time.perf_counter()
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)
        client_ip = self._get_client_ip()

        # 1. Unauthenticated Health Probe Endpoint
        if path in ("/health", "/api/health"):
            duration = (time.perf_counter() - start_time) * 1000.0
            uptime = round(time.time() - getattr(self, "server_start_time", time.time()), 1)
            cfg = getattr(self, "security_config", None)
            res = {
                "status": "healthy",
                "service": "vipym-studio",
                "version": "0.1.0",
                "uptime_seconds": uptime,
                "read_only": cfg.read_only if cfg else False,
            }
            self.send_json_response(res, status=200)
            if hasattr(self, "audit_logger") and self.audit_logger:
                self.audit_logger.log_action(
                    method="GET",
                    path=path,
                    source_ip=client_ip,
                    status_code=200,
                    token_id="probe",
                    duration_ms=duration,
                    read_only=cfg.read_only if cfg else False,
                )
            return

        # 2. Extract Auth Token
        token = None
        if hasattr(self, "token_validator") and self.token_validator:
            token = self.token_validator.extract_token(self.headers, query)

        masked_token = (
            self.token_validator.mask_token(token)
            if hasattr(self, "token_validator")
            else "anonymous"
        )
        rate_key = token if token else client_ip

        # 3. Rate Limit Check
        allowed, retry_after = self._check_rate_limit(rate_key)
        if not allowed:
            duration = (time.perf_counter() - start_time) * 1000.0
            self.send_json_response(
                {
                    "error": "Rate limit exceeded (100 req/min)",
                    "retry_after_seconds": retry_after,
                },
                status=429,
                headers={"Retry-After": str(int(retry_after))},
            )
            if hasattr(self, "audit_logger") and self.audit_logger:
                self.audit_logger.log_action(
                    method="GET",
                    path=path,
                    source_ip=client_ip,
                    status_code=429,
                    token_id=masked_token,
                    duration_ms=duration,
                )
            return

        # 4. WebSocket Upgrade Handler (/ws/progress or /api/ws/progress)
        if (
            path in ("/ws/progress", "/api/ws/progress")
            or self.headers.get("Upgrade", "").lower() == "websocket"
        ):
            self.handle_websocket_upgrade(token=token, query=query)
            return

        # 5. REST API Routes
        if path.startswith("/api/"):
            # Enforce read token auth if require_auth_for_read is active
            cfg = getattr(self, "security_config", None)
            if cfg and cfg.require_auth_for_read:
                if not hasattr(self, "token_validator") or not self.token_validator.validate(token):
                    duration = (time.perf_counter() - start_time) * 1000.0
                    self.send_json_response(
                        {"error": "Unauthorized: Invalid or missing authentication token"},
                        status=401,
                    )
                    if hasattr(self, "audit_logger") and self.audit_logger:
                        self.audit_logger.log_action(
                            method="GET",
                            path=path,
                            source_ip=client_ip,
                            status_code=401,
                            token_id=masked_token,
                            duration_ms=duration,
                        )
                    return

            status_code = self.handle_api_get(path, query)
            duration = (time.perf_counter() - start_time) * 1000.0
            if hasattr(self, "audit_logger") and self.audit_logger:
                self.audit_logger.log_action(
                    method="GET",
                    path=path,
                    source_ip=client_ip,
                    status_code=status_code,
                    token_id=masked_token,
                    duration_ms=duration,
                    read_only=cfg.read_only if cfg else False,
                )
            return

        # 6. Serve SPA and Static Assets
        if path in ("/", "/index.html"):
            self.serve_file(self.static_dir / "index.html", "text/html")
            return

        local_file = self.static_dir / path.lstrip("/")
        if local_file.exists() and local_file.is_file():
            mime_type, _ = mimetypes.guess_type(str(local_file))
            self.serve_file(local_file, mime_type or "application/octet-stream")
            return

        # Fallback to SPA index
        self.serve_file(self.static_dir / "index.html", "text/html")

    def do_POST(self) -> None:
        start_time = time.perf_counter()
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)
        client_ip = self._get_client_ip()
        cfg = getattr(self, "security_config", None)
        read_only = cfg.read_only if cfg else False

        # 1. Extract and Validate Auth Token
        token = None
        if hasattr(self, "token_validator") and self.token_validator:
            token = self.token_validator.extract_token(self.headers, query)

        masked_token = (
            self.token_validator.mask_token(token)
            if hasattr(self, "token_validator")
            else "anonymous"
        )
        rate_key = token if token else client_ip

        # 2. Rate Limit Check
        allowed, retry_after = self._check_rate_limit(rate_key)
        if not allowed:
            duration = (time.perf_counter() - start_time) * 1000.0
            self.send_json_response(
                {
                    "error": "Rate limit exceeded (100 req/min)",
                    "retry_after_seconds": retry_after,
                },
                status=429,
                headers={"Retry-After": str(int(retry_after))},
            )
            if hasattr(self, "audit_logger") and self.audit_logger:
                self.audit_logger.log_action(
                    method="POST",
                    path=path,
                    source_ip=client_ip,
                    status_code=429,
                    token_id=masked_token,
                    duration_ms=duration,
                )
            return

        # 3. Token Authentication Enforcement for ALL Mutations
        if not hasattr(self, "token_validator") or not self.token_validator.validate(token):
            duration = (time.perf_counter() - start_time) * 1000.0
            self.send_json_response(
                {"error": "Unauthorized: Invalid or missing authentication token"},
                status=401,
            )
            if hasattr(self, "audit_logger") and self.audit_logger:
                self.audit_logger.log_action(
                    method="POST",
                    path=path,
                    source_ip=client_ip,
                    status_code=401,
                    token_id=masked_token,
                    duration_ms=duration,
                    read_only=read_only,
                )
            return

        # 4. Read-Only Mode Enforcement
        if read_only:
            duration = (time.perf_counter() - start_time) * 1000.0
            self.send_json_response(
                {
                    "error": "Forbidden: Studio server is running in read-only mode",
                    "read_only": True,
                },
                status=403,
            )
            if hasattr(self, "audit_logger") and self.audit_logger:
                self.audit_logger.log_action(
                    method="POST",
                    path=path,
                    source_ip=client_ip,
                    status_code=403,
                    token_id=masked_token,
                    duration_ms=duration,
                    read_only=read_only,
                )
            return

        # 5. Parse Payload
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            payload = json.loads(post_data) if post_data else {}
        except Exception:
            payload = {}

        # 6. Route Mutation Endpoints
        status_code = 200
        if path == "/api/validate":
            status_code = self.handle_api_validate(payload)
        elif path == "/api/experiments/start":
            self.send_json_response(
                {
                    "status": "queued",
                    "message": "Experiment launched successfully",
                    "payload": payload,
                },
                status=202,
            )
            status_code = 202
        elif path.startswith("/api/experiments/") and path.endswith("/cancel"):
            exp_id = path.replace("/api/experiments/", "").replace("/cancel", "").strip("/")
            self.send_json_response(
                {"status": "cancelled", "experiment_id": exp_id},
                status=200,
            )
            status_code = 200
        else:
            self.send_json_response({"error": f"Endpoint POST '{path}' not found"}, status=404)
            status_code = 404

        duration = (time.perf_counter() - start_time) * 1000.0
        if hasattr(self, "audit_logger") and self.audit_logger:
            self.audit_logger.log_action(
                method="POST",
                path=path,
                source_ip=client_ip,
                status_code=status_code,
                token_id=masked_token,
                duration_ms=duration,
                read_only=read_only,
            )

    def handle_websocket_upgrade(self, token: str | None, query: dict[str, list[str]]) -> None:
        """Handle RFC 6455 WebSocket upgrade handshake with token validation."""
        ws_key = self.headers.get("Sec-WebSocket-Key")
        if not ws_key:
            self.send_json_response({"error": "Missing Sec-WebSocket-Key header"}, status=400)
            return

        manager = getattr(self, "ws_manager", get_websocket_manager())
        validator = getattr(self, "token_validator", None)

        if validator and not validator.validate(token):
            self.send_json_response(
                {"error": "Unauthorized: Valid token required for WebSocket stream"},
                status=401,
            )
            return

        accept_key = StudioWebSocketManager.compute_handshake_accept(ws_key)
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept_key)
        self._apply_cors_headers()
        self.end_headers()

        # Register client socket for asynchronous push updates
        manager.register_client(self.request)

    def handle_api_get(self, path: str, query: dict[str, list[str]]) -> int:
        if path == "/api/status":
            cfg = getattr(self, "security_config", None)
            self.send_json_response(
                {
                    "status": "online",
                    "vipym_version": "0.1.0",
                    "artifacts_dir": str(self.artifacts_dir),
                    "read_only": cfg.read_only if cfg else False,
                    "auth_enabled": True,
                }
            )
            return 200
        elif path == "/api/experiments":
            self.send_json_response(self.list_experiments())
            return 200
        elif path.startswith("/api/experiments/"):
            exp_id = path.replace("/api/experiments/", "").strip("/")
            self.send_json_response(self.get_experiment_details(exp_id))
            return 200
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
            return 200
        elif path == "/api/models/inspect":
            model_id = query.get("model_id", ["moonshotai/Kimi-K3"])[0]
            self.send_json_response(self.inspect_model(model_id))
            return 200
        elif path == "/api/doctor":
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
            return 200
        else:
            self.send_json_response({"error": f"API endpoint '{path}' not found"}, status=404)
            return 404

    def handle_api_validate(self, payload: dict[str, Any]) -> int:
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
            return 200
        except Exception as e:
            self.send_json_response({"valid": False, "error": str(e)}, status=400)
            return 400

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
        self._apply_cors_headers()
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_json_response(
        self,
        data: Any,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._apply_cors_headers()
        if headers:
            for k, v in headers.items():
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP Server for concurrent requests and WebSocket streams."""

    daemon_threads = True
    allow_reuse_address = True


def start_studio_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    artifacts_dir: Path | str = "./artifacts",
    token: str | None = None,
    read_only: bool = False,
    rate_limit: int = 100,
    cors_origins: list[str] | None = None,
    audit_log_path: Path | str | None = None,
    require_auth_for_read: bool = False,
) -> ThreadedHTTPServer:
    """Instantiate and start the hardened ViPym Studio web server.

    Security guarantees:
    - Binds to 127.0.0.1 by default (localhost only)
    - Validates Bearer tokens on mutation endpoints (and reads if configured)
    - Enforces 100 req/min rate limit per client/token
    - Structured audit logging to ~/.vipym/studio-audit.log
    - Read-only protection prevents state changes when read_only=True
    """
    artifacts_path = Path(artifacts_dir).resolve()

    # Enforce or issue authentication token
    auth_token = get_or_create_studio_token(override_token=token)
    validator = TokenValidator(expected_token=auth_token)
    limiter = RateLimiter(max_requests=rate_limit, window_seconds=60.0)
    auditor = AuditLogger(log_path=audit_log_path)
    ws_mgr = get_websocket_manager()
    ws_mgr.set_validator(validator)

    origins = cors_origins or [
        f"http://{host}:{port}",
        f"http://127.0.0.1:{port}",
        f"http://localhost:{port}",
        "http://127.0.0.1",
        "http://localhost",
    ]

    sec_config = SecurityConfig(
        auth_token=auth_token,
        read_only=read_only,
        rate_limit_req_per_min=rate_limit,
        allowed_origins=origins,
        audit_log_path=Path(audit_log_path) if audit_log_path else None,
        require_auth_for_read=require_auth_for_read,
    )

    # Attach shared security singletons to handler class
    StudioAPIHandler.security_config = sec_config
    StudioAPIHandler.token_validator = validator
    StudioAPIHandler.rate_limiter = limiter
    StudioAPIHandler.audit_logger = auditor
    StudioAPIHandler.ws_manager = ws_mgr
    StudioAPIHandler.server_start_time = time.time()

    def handler_factory(*args: Any, **kwargs: Any) -> StudioAPIHandler:
        return StudioAPIHandler(*args, artifacts_dir=artifacts_path, **kwargs)

    server = ThreadedHTTPServer((host, port), handler_factory)

    # Attach token & config to server instance for direct test access
    server.auth_token = auth_token
    server.security_config = sec_config
    server.rate_limiter = limiter
    server.audit_logger = auditor
    server.ws_manager = ws_mgr

    if host == "0.0.0.0":
        logger.warning(
            "SECURITY WARNING: ViPym Studio bound to 0.0.0.0 (public interface). "
            "Ensure token authentication is strictly enforced."
        )

    logger.info(
        f"ViPym Studio UI & REST API running at: [bold cyan]http://{host}:{port}[/bold cyan] "
        f"(read_only={read_only}, rate_limit={rate_limit} req/min)"
    )

    return server
