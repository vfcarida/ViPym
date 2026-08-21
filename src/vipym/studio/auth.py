"""ViPym Studio Authentication, Security, Rate Limiting & Audit Logging."""

from __future__ import annotations

import collections
import dataclasses
import datetime
import hmac
import json
import os
import secrets
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vipym.core.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TOKEN_PATH = Path.home() / ".vipym" / "studio-token"
DEFAULT_AUDIT_LOG_PATH = Path.home() / ".vipym" / "studio-audit.log"


def get_or_create_studio_token(
    token_path: Path | str | None = None,
    override_token: str | None = None,
) -> str:
    """Retrieve existing Studio auth token or generate a secure Bearer token saved to ~/.vipym/studio-token."""
    # 1. Explicit override or environment variable takes precedence
    if override_token and override_token.strip():
        return override_token.strip()

    env_token = os.environ.get("VIPYM_STUDIO_TOKEN")
    if env_token and env_token.strip():
        return env_token.strip()

    path = Path(token_path).resolve() if token_path else DEFAULT_TOKEN_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Read existing token if available
    if path.exists() and path.is_file():
        try:
            token = path.read_text(encoding="utf-8").strip()
            if token:
                return token
        except Exception as e:
            logger.warning(f"Could not read existing studio token from {path}: {e}")

    # 3. Generate a cryptographically secure 256-bit URL-safe token
    token = secrets.token_urlsafe(32)
    try:
        path.write_text(token, encoding="utf-8")
        # Restrict permissions to owner-only on POSIX if supported
        try:
            os.chmod(path, 0o600)
        except Exception:
            pass
        logger.info(f"Generated new ViPym Studio authentication token at: {path}")
    except Exception as e:
        logger.error(f"Failed to persist studio token to {path}: {e}")

    return token


class TokenValidator:
    """Validates Bearer tokens and API keys using constant-time comparisons."""

    def __init__(self, expected_token: str) -> None:
        self.expected_token = expected_token.strip()

    def extract_token(
        self,
        headers: Mapping[str, str] | dict[str, str],
        query: dict[str, list[str]] | None = None,
    ) -> str | None:
        """Extract auth token from Authorization header, X-API-Key header, or query string."""
        # Check standard Authorization header (case-insensitive)
        auth_header = None
        for k, v in headers.items():
            if k.lower() == "authorization":
                auth_header = v
                break

        if auth_header:
            parts = auth_header.split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                return parts[1].strip()
            if len(parts) == 1:
                return parts[0].strip()

        # Check X-API-Key header
        for k, v in headers.items():
            if k.lower() == "x-api-key":
                return v.strip()

        # Check query parameters (e.g. ?token=...)
        if query and "token" in query and query["token"]:
            return query["token"][0].strip()

        return None

    def validate(self, provided_token: str | None) -> bool:
        """Perform constant-time comparison against expected token."""
        if not provided_token or not self.expected_token:
            return False
        return hmac.compare_digest(provided_token.strip(), self.expected_token)

    def mask_token(self, token: str | None) -> str:
        """Return a masked representation of a token for safe audit logging."""
        if not token:
            return "anonymous"
        if len(token) <= 8:
            return "***"
        return f"{token[:4]}...{token[-4:]}"


class RateLimiter:
    """Thread-safe sliding-window rate limiter (e.g. 100 requests / minute per client/token)."""

    def __init__(self, max_requests: int = 100, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, collections.deque[float]] = collections.defaultdict(
            collections.deque
        )
        self._lock = threading.Lock()

    def is_allowed(self, key: str) -> tuple[bool, float]:
        """Check whether request for key is allowed under the rate limit.

        Returns:
            (is_allowed, retry_after_seconds)
        """
        now = time.time()
        with self._lock:
            q = self._requests[key]
            # Remove timestamps outside the sliding window
            cutoff = now - self.window_seconds
            while q and q[0] < cutoff:
                q.popleft()

            if len(q) < self.max_requests:
                q.append(now)
                return True, 0.0

            # Rate limit exceeded; calculate wait time until oldest request rolls off
            oldest = q[0]
            retry_after = max(1.0, math_ceil(self.window_seconds - (now - oldest)))
            return False, float(retry_after)

    def reset(self) -> None:
        """Clear all recorded request timestamps."""
        with self._lock:
            self._requests.clear()


def math_ceil(v: float) -> int:
    """Integer ceiling without extra imports."""
    val = int(v)
    return val + 1 if v > val else val


class AuditLogger:
    """Logs all API requests with timestamp, action, IP, token ID, and latency."""

    def __init__(self, log_path: Path | str | None = None) -> None:
        self.log_path = Path(log_path).resolve() if log_path else DEFAULT_AUDIT_LOG_PATH
        self._lock = threading.Lock()
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def log_action(
        self,
        method: str,
        path: str,
        source_ip: str,
        status_code: int,
        token_id: str = "anonymous",
        duration_ms: float = 0.0,
        read_only: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Write structured audit log entry."""
        entry = {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "method": method.upper(),
            "path": path,
            "source_ip": source_ip,
            "status_code": status_code,
            "token_id": token_id,
            "duration_ms": round(duration_ms, 2),
            "read_only": read_only,
        }
        if extra:
            entry["extra"] = extra

        line = json.dumps(entry) + "\n"
        with self._lock:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(line)
            except Exception as e:
                logger.warning(f"Failed writing studio audit log to {self.log_path}: {e}")

        # Also emit to structlog / logger
        logger.info(
            f"AUDIT: {method} {path} status={status_code} ip={source_ip} user={token_id} ({duration_ms:.1f}ms)"
        )


@dataclasses.dataclass
class SecurityConfig:
    """Security and hardening configuration for ViPym Studio API."""

    auth_token: str
    read_only: bool = False
    rate_limit_req_per_min: int = 100
    allowed_origins: list[str] = dataclasses.field(
        default_factory=lambda: [
            "http://127.0.0.1",
            "http://localhost",
            "http://[::1]",
        ]
    )
    audit_log_path: Path | None = None
    require_auth_for_read: bool = False

    def is_origin_allowed(self, origin: str | None, host_header: str | None = None) -> bool:
        """Determine if origin is permitted under strict same-origin CORS policy."""
        if not origin:
            return True  # Same-origin direct browser requests without Origin header

        origin_clean = origin.rstrip("/")
        for allowed in self.allowed_origins:
            allowed_clean = allowed.rstrip("/")
            if origin_clean == allowed_clean or origin_clean.startswith(f"{allowed_clean}:"):
                return True

        if host_header:
            if origin_clean == f"http://{host_header}" or origin_clean == f"https://{host_header}":
                return True

        return False
