"""Unit tests for ViPym Studio API Authentication, Security, Rate Limiting, Audit Logging & WebSockets."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from vipym.studio.auth import (
    AuditLogger,
    RateLimiter,
    SecurityConfig,
    TokenValidator,
    get_or_create_studio_token,
)
from vipym.studio.server import start_studio_server
from vipym.studio.websocket import StudioWebSocketManager


class TestTokenAuthAndStorage:
    def test_token_creation_and_persistence(self, tmp_path: Path):
        """Verify token is generated and persisted to file."""
        token_file = tmp_path / "custom-studio-token"
        token = get_or_create_studio_token(token_path=token_file)

        assert token is not None
        assert len(token) >= 32
        assert token_file.exists()
        assert token_file.read_text(encoding="utf-8").strip() == token

        # Second call should load and return the same token
        token2 = get_or_create_studio_token(token_path=token_file)
        assert token2 == token

    def test_token_override_and_env_var(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """Verify override token and environment variable takes priority."""
        token_file = tmp_path / "env-studio-token"

        # Explicit override
        token_override = get_or_create_studio_token(
            token_path=token_file, override_token="explicit-secret-12345"
        )
        assert token_override == "explicit-secret-12345"

        # Environment variable
        monkeypatch.setenv("VIPYM_STUDIO_TOKEN", "env-secret-67890")
        token_env = get_or_create_studio_token(token_path=token_file)
        assert token_env == "env-secret-67890"

    def test_token_validator(self):
        """Verify constant-time token validation and header extraction."""
        validator = TokenValidator(expected_token="vipym_master_key_999")

        # Valid Bearer header
        headers = {"Authorization": "Bearer vipym_master_key_999"}
        token = validator.extract_token(headers)
        assert token == "vipym_master_key_999"
        assert validator.validate(token) is True

        # Valid X-API-Key header
        headers = {"x-api-key": "vipym_master_key_999"}
        token = validator.extract_token(headers)
        assert token == "vipym_master_key_999"
        assert validator.validate(token) is True

        # Valid Query parameter
        query = {"token": ["vipym_master_key_999"]}
        token = validator.extract_token({}, query)
        assert token == "vipym_master_key_999"
        assert validator.validate(token) is True

        # Invalid token
        assert validator.validate("wrong_token_here") is False
        assert validator.validate("") is False
        assert validator.validate(None) is False

        # Masked token
        assert validator.mask_token("vipym_master_key_999") == "vipy..._999"
        assert validator.mask_token(None) == "anonymous"


class TestRateLimiter:
    def test_rate_limiting_sliding_window(self):
        """Verify 100 requests limit within 60-second window."""
        limiter = RateLimiter(max_requests=5, window_seconds=2.0)
        key = "client-token-1"

        # 5 requests should be allowed
        for _ in range(5):
            allowed, retry = limiter.is_allowed(key)
            assert allowed is True
            assert retry == 0.0

        # 6th request must be rejected with retry_after
        allowed, retry = limiter.is_allowed(key)
        assert allowed is False
        assert retry > 0.0

        # Separate key has its own bucket
        allowed_other, _ = limiter.is_allowed("client-token-2")
        assert allowed_other is True

        # Reset
        limiter.reset()
        allowed_after_reset, _ = limiter.is_allowed(key)
        assert allowed_after_reset is True


class TestAuditLogger:
    def test_audit_log_writing(self, tmp_path: Path):
        """Verify structured JSON audit log entries."""
        log_file = tmp_path / "test-audit.log"
        auditor = AuditLogger(log_path=log_file)

        auditor.log_action(
            method="POST",
            path="/api/validate",
            source_ip="127.0.0.1",
            status_code=200,
            token_id="tok_***1234",
            duration_ms=12.5,
            read_only=False,
        )

        assert log_file.exists()
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["method"] == "POST"
        assert entry["path"] == "/api/validate"
        assert entry["source_ip"] == "127.0.0.1"
        assert entry["status_code"] == 200
        assert entry["token_id"] == "tok_***1234"
        assert entry["read_only"] is False


class TestSecurityConfigCORS:
    def test_cors_same_origin_validation(self):
        """Verify same-origin CORS rules."""
        cfg = SecurityConfig(
            auth_token="secret",
            allowed_origins=["http://127.0.0.1:8080", "http://localhost:8080"],
        )

        assert cfg.is_origin_allowed(None) is True
        assert cfg.is_origin_allowed("http://127.0.0.1:8080") is True
        assert cfg.is_origin_allowed("http://localhost:8080") is True
        assert cfg.is_origin_allowed("http://malicious-site.com") is False


class TestStudioServerSecurityIntegration:
    @pytest.fixture(scope="module")
    def secure_server(self, tmp_path_factory: pytest.TempPathFactory):
        """Launch test Studio server with auth token, audit log, and rate limit = 10."""
        tmp = tmp_path_factory.mktemp("secure_studio")
        artifacts_dir = tmp / "artifacts"
        artifacts_dir.mkdir()
        audit_file = tmp / "studio-audit.log"

        server = start_studio_server(
            host="127.0.0.1",
            port=19876,
            artifacts_dir=artifacts_dir,
            token="test-secret-token-abc123xyz",
            read_only=False,
            rate_limit=15,
            audit_log_path=audit_file,
        )
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        time.sleep(0.3)

        yield {
            "url": "http://127.0.0.1:19876",
            "token": "test-secret-token-abc123xyz",
            "audit_file": audit_file,
            "server": server,
        }

        server.shutdown()
        server.server_close()

    def test_health_endpoint_unauthenticated(self, secure_server: dict):
        """GET /health is public and does not require auth."""
        url = f"{secure_server['url']}/health"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "healthy"
            assert data["service"] == "vipym-studio"
            assert data["read_only"] is False

    def test_mutation_requires_valid_token(self, secure_server: dict):
        """POST /api/validate requires valid Bearer token."""
        url = f"{secure_server['url']}/api/validate"
        body = json.dumps({"yaml": "experiment_id: test-001\nmodel:\n  id: test"}).encode("utf-8")

        # 1. Request with NO token -> 401 Unauthorized
        req_no_auth = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req_no_auth)
        assert exc_info.value.code == 401

        # 2. Request with INVALID token -> 401 Unauthorized
        req_bad_auth = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer bad-token-999",
            },
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req_bad_auth)
        assert exc_info.value.code == 401

        # 3. Request with VALID Bearer token -> 200 OK
        req_valid = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {secure_server['token']}",
            },
        )
        with urllib.request.urlopen(req_valid) as resp:
            assert resp.status == 200
            res = json.loads(resp.read().decode("utf-8"))
            assert "valid" in res

    def test_rate_limiting_enforcement(self, secure_server: dict):
        """Sending requests exceeding rate limit returns 429 Too Many Requests."""
        url = f"{secure_server['url']}/api/status"
        token = secure_server["token"]
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})

        # Send rapid requests until rate limit is triggered
        got_429 = False
        for _ in range(25):
            try:
                with urllib.request.urlopen(req) as resp:
                    assert resp.status == 200
            except urllib.error.HTTPError as err:
                if err.code == 429:
                    got_429 = True
                    assert "Retry-After" in err.headers
                    break

        assert got_429 is True, "Expected 429 Too Many Requests after exceeding rate limit"
        # Reset limiter for subsequent tests
        secure_server["server"].rate_limiter.reset()

    def test_read_only_mode_blocks_mutations(self, tmp_path_factory: pytest.TempPathFactory):
        """Verify --read-only mode returns 403 Forbidden for mutation requests."""
        tmp = tmp_path_factory.mktemp("readonly_studio")
        server = start_studio_server(
            host="127.0.0.1",
            port=19877,
            artifacts_dir=tmp,
            token="readonly-token",
            read_only=True,
        )
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        time.sleep(0.3)

        try:
            url = "http://127.0.0.1:19877"
            token = "readonly-token"

            # GET requests work in read-only mode
            req_get = urllib.request.Request(
                f"{url}/api/status", headers={"Authorization": f"Bearer {token}"}
            )
            with urllib.request.urlopen(req_get) as resp:
                assert resp.status == 200
                data = json.loads(resp.read().decode("utf-8"))
                assert data["read_only"] is True

            # POST mutations return 403 Forbidden
            body = json.dumps({"yaml": "experiment_id: ro-001"}).encode("utf-8")
            req_post = urllib.request.Request(
                f"{url}/api/validate",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                },
            )
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(req_post)
            assert exc_info.value.code == 403
            err_body = json.loads(exc_info.value.read().decode("utf-8"))
            assert err_body["read_only"] is True
            assert "Forbidden" in err_body["error"]

        finally:
            server.shutdown()
            server.server_close()


class TestStudioWebSocketStreaming:
    def test_websocket_manager_broadcast(self):
        """Verify StudioWebSocketManager distributes events to subscribers."""
        manager = StudioWebSocketManager(validator=TokenValidator("ws-secret"))
        received_events = []

        def on_progress(event: dict):
            received_events.append(event)

        manager.subscribe(on_progress)

        manager.broadcast_progress(
            {
                "stage": "quantization",
                "progress_pct": 50.0,
                "duration_seconds": 12.4,
                "eta_seconds": 12.4,
            }
        )

        assert len(received_events) == 1
        assert received_events[0]["stage"] == "quantization"
        assert received_events[0]["progress_pct"] == 50.0
        assert "timestamp" in received_events[0]

    def test_websocket_handshake_authentication(self):
        """Verify RFC 6455 handshake accepts valid token and rejects invalid token."""
        validator = TokenValidator("ws-auth-pass")
        manager = StudioWebSocketManager(validator=validator)

        key = "dGhlIHNhbXBsZSBub25jZQ=="
        # Valid token
        ok, status, headers = manager.handle_handshake(key=key, token="ws-auth-pass")
        assert ok is True
        assert status == "101 Switching Protocols"
        assert "Sec-WebSocket-Accept" in headers
        assert headers["Sec-WebSocket-Accept"] == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="

        # Invalid token
        ok_bad, status_bad, _ = manager.handle_handshake(key=key, token="wrong-ws-pass")
        assert ok_bad is False
        assert status_bad == "401 Unauthorized"
