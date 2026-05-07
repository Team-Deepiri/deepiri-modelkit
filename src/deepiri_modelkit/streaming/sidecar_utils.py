"""
Shared Sugar Glider/Synapse sidecar helpers.

These utilities are reused by multiple services (for example Cyrex and Helox)
to keep sidecar transport behavior consistent across repos.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse


def env_float(name: str, default: float, logger: Optional[Callable[[str], None]] = None) -> float:
    """Read a float env var with safe fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        if logger is not None:
            logger(f"invalid float env {name}={raw!r}; using {default}")
        return default


def resolve_grpc_addr(base_url: str, explicit_grpc_addr: Optional[str] = None) -> str:
    """
    Resolve sidecar gRPC host:port from explicit/env/base URL.

    Resolution order:
    1) explicit_grpc_addr
    2) SYNAPSE_GRPC_ADDR
    3) derive from base_url (8081 -> 50051)
    """
    env_addr = os.getenv("SYNAPSE_GRPC_ADDR")
    if explicit_grpc_addr:
        return explicit_grpc_addr
    if env_addr:
        return env_addr

    parsed = urlparse(base_url)
    if parsed.scheme in {"http", "https"}:
        host = parsed.hostname or "localhost"
        port = parsed.port
        if port is None:
            port = 443 if parsed.scheme == "https" else 80
        if port == 8081:
            port = 50051
        return f"{host}:{port}"

    if base_url:
        return base_url
    return "localhost:50051"


def sidecar_payload_from_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize sidecar event fields to a payload dict."""
    payload = fields.get("payload", {})
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except ValueError:
            payload = {}
    elif not isinstance(payload, dict):
        payload = {}

    if "event" not in payload and fields.get("event_type"):
        payload["event"] = fields.get("event_type")

    if "timestamp" not in payload and fields.get("timestamp"):
        payload["timestamp"] = fields.get("timestamp")

    if "sender" not in payload and fields.get("sender"):
        payload["sender"] = fields.get("sender")

    return payload
