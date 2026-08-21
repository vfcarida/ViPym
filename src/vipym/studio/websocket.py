"""ViPym Studio Real-Time WebSocket Progress Streaming & Connection Manager."""

from __future__ import annotations

import base64
import hashlib
import json
import socket
import struct
import threading
import time
from collections.abc import Callable
from typing import Any

from vipym.core.logger import get_logger
from vipym.studio.auth import TokenValidator

logger = get_logger(__name__)

# RFC 6455 WebSocket Magic GUID
WS_MAGIC_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class StudioWebSocketManager:
    """Manages authenticated WebSocket connections and broadcasts real-time progress events."""

    def __init__(self, validator: TokenValidator | None = None) -> None:
        self.validator = validator
        self._clients: set[socket.socket] = set()
        self._lock = threading.Lock()
        self._event_history: list[dict[str, Any]] = []
        self._max_history = 100
        self._subscribers: list[Callable[[dict[str, Any]], None]] = []

    def set_validator(self, validator: TokenValidator) -> None:
        self.validator = validator

    def register_client(self, client_sock: socket.socket) -> None:
        """Register an active authenticated WebSocket client socket."""
        with self._lock:
            self._clients.add(client_sock)
            logger.info(
                f"WebSocket client connected. Total active connections: {len(self._clients)}"
            )

            # Replay recent events to newly connected client
            for event in self._event_history[-10:]:
                try:
                    self._send_frame(client_sock, json.dumps(event))
                except Exception:
                    pass

    def unregister_client(self, client_sock: socket.socket) -> None:
        """Unregister a disconnected WebSocket client socket."""
        with self._lock:
            self._clients.discard(client_sock)
            try:
                client_sock.close()
            except Exception:
                pass
            logger.info(
                f"WebSocket client disconnected. Remaining connections: {len(self._clients)}"
            )

    def subscribe(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register a callback subscriber for progress events."""
        with self._lock:
            self._subscribers.append(callback)

    def broadcast_progress(self, event_data: dict[str, Any]) -> None:
        """Broadcast a real-time progress update event to all authenticated clients."""
        payload = dict(event_data)
        if "timestamp" not in payload:
            payload["timestamp"] = time.time()

        with self._lock:
            self._event_history.append(payload)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)

            # Notify in-process subscribers
            for cb in self._subscribers:
                try:
                    cb(payload)
                except Exception as e:
                    logger.warning(f"Error in progress subscriber: {e}")

            # Send to active WebSocket sockets
            msg = json.dumps(payload)
            dead_clients = set()

            for client in self._clients:
                try:
                    self._send_frame(client, msg)
                except Exception:
                    dead_clients.add(client)

            for dead in dead_clients:
                self._clients.discard(dead)
                try:
                    dead.close()
                except Exception:
                    pass

    def _send_frame(self, client_sock: socket.socket, message: str) -> None:
        """Encode and send an RFC 6455 unmasked text WebSocket frame."""
        data = message.encode("utf-8")
        length = len(data)

        # Byte 1: FIN (1) + Opcode 0x1 (Text) = 0x81
        frame = bytearray([0x81])

        # Byte 2+: Payload length encoding
        if length <= 125:
            frame.append(length)
        elif length <= 65535:
            frame.append(126)
            frame.extend(struct.pack("!H", length))
        else:
            frame.append(127)
            frame.extend(struct.pack("!Q", length))

        frame.extend(data)
        client_sock.sendall(frame)

    @classmethod
    def compute_handshake_accept(cls, key: str) -> str:
        """Compute the Sec-WebSocket-Accept response header value for RFC 6455 handshake."""
        combined = key.strip() + WS_MAGIC_GUID
        sha1_hash = hashlib.sha1(combined.encode("utf-8")).digest()
        return base64.b64encode(sha1_hash).decode("utf-8")

    def handle_handshake(
        self,
        key: str,
        token: str | None,
    ) -> tuple[bool, str, dict[str, str]]:
        """Validate token and produce RFC 6455 handshake response headers."""
        if self.validator and not self.validator.validate(token):
            return False, "401 Unauthorized", {"Content-Type": "application/json"}

        accept_val = self.compute_handshake_accept(key)
        headers = {
            "Upgrade": "websocket",
            "Connection": "Upgrade",
            "Sec-WebSocket-Accept": accept_val,
        }
        return True, "101 Switching Protocols", headers


# Global default manager instance
_GLOBAL_WS_MANAGER = StudioWebSocketManager()


def get_websocket_manager() -> StudioWebSocketManager:
    """Return the global StudioWebSocketManager instance."""
    return _GLOBAL_WS_MANAGER
