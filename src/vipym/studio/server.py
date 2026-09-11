"""ViPym Studio ASGI Server & Service Management.

Modernizes the ViPym Studio architecture to FastAPI and Uvicorn ASGI while preserving:
- Localhost binding by default (127.0.0.1)
- Token authentication with auto-generated Bearer tokens saved to ~/.vipym/studio-token
- Rate limiting (100 req/min per token/IP) with 429 Too Many Requests
- Comprehensive structured audit logging of all requests to ~/.vipym/studio-audit.log
- Read-only mode (--read-only) blocking mutations with 403 Forbidden
- Unauthenticated /health endpoint for probes and health checks
- Strict same-origin CORS protection by default
- Authenticated real-time WebSocket progress streaming (/ws/progress)
"""

from __future__ import annotations

import threading
import time
from http.server import SimpleHTTPRequestHandler
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from vipym.core.logger import get_logger
from vipym.studio.app import create_studio_app
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
    """Legacy handler stub kept for backwards compatibility."""

    security_config: SecurityConfig
    token_validator: TokenValidator
    rate_limiter: RateLimiter
    audit_logger: AuditLogger
    ws_manager: StudioWebSocketManager
    server_start_time: float = time.time()


class ThreadedHTTPServer:
    """Legacy server stub kept for backwards compatibility."""

    daemon_threads = True
    allow_reuse_address = True


class StudioASGIServer:
    """Production ASGI Server wrapper managing Uvicorn and FastAPI."""

    def __init__(
        self,
        app: FastAPI,
        host: str = "127.0.0.1",
        port: int = 8080,
        auth_token: str | None = None,
        security_config: SecurityConfig | None = None,
        rate_limiter: RateLimiter | None = None,
        audit_logger: AuditLogger | None = None,
        ws_manager: StudioWebSocketManager | None = None,
    ) -> None:
        self.app = app
        self.host = host
        self.port = port
        self.auth_token = auth_token
        self.security_config = security_config
        self.rate_limiter = rate_limiter
        self.audit_logger = audit_logger
        self.ws_manager = ws_manager
        self.server_start_time = time.time()

        self.config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
            loop="asyncio",
            lifespan="off",
            access_log=False,
        )
        self.server = uvicorn.Server(self.config)
        self._thread: threading.Thread | None = None

    def serve_forever(self) -> None:
        """Run the Uvicorn ASGI server synchronously until shutdown."""
        self.server.run()

    def shutdown(self) -> None:
        """Signal the Uvicorn server to stop accepting requests."""
        self.server.should_exit = True

    def server_close(self) -> None:
        """Clean up sockets and stop server loop."""
        self.server.should_exit = True


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
) -> StudioASGIServer:
    """Instantiate and configure the hardened ViPym Studio ASGI web server.

    Security guarantees:
    - Binds to 127.0.0.1 by default (localhost only)
    - Validates Bearer tokens on mutation endpoints (and reads if configured)
    - Enforces rate limit per client/token (default 100 req/min)
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

    app = create_studio_app(
        artifacts_dir=artifacts_path,
        security_config=sec_config,
        token_validator=validator,
        rate_limiter=limiter,
        audit_logger=auditor,
        ws_manager=ws_mgr,
    )

    server = StudioASGIServer(
        app=app,
        host=host,
        port=port,
        auth_token=auth_token,
        security_config=sec_config,
        rate_limiter=limiter,
        audit_logger=auditor,
        ws_manager=ws_mgr,
    )

    if host == "0.0.0.0":
        logger.warning(
            "SECURITY WARNING: ViPym Studio bound to 0.0.0.0 (public interface). "
            "Ensure token authentication is strictly enforced."
        )

    logger.info(
        f"ViPym Studio ASGI server initialized at: [bold cyan]http://{host}:{port}[/bold cyan] "
        f"(read_only={read_only}, rate_limit={rate_limit} req/min)"
    )

    return server
