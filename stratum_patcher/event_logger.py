"""JSONL event writer for stratum-patcher.

All framework patchers write through the singleton EventLogger to produce
a single, append-only JSONL stream that the harness later collects from
the container.  Thread-safe via threading.Lock.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import threading
import time
import uuid
from typing import Any


# ---------------------------------------------------------------------------
# Helper utilities (module-level so they can be imported independently)
# ---------------------------------------------------------------------------

def make_node(node_type: str, node_id: str, node_name: str) -> dict[str, str]:
    """Return a node descriptor dict suitable for event source/target fields."""
    return {"node_type": node_type, "node_id": node_id, "node_name": node_name}


def generate_node_id(
    framework: str,
    class_name: str,
    source_file: str,
    line_number: int | str,
) -> str:
    """Return a runtime node ID: ``framework:ClassName:file:line``."""
    return f"{framework}:{class_name}:{source_file}:{line_number}"


def get_caller_info(skip_frames: int = 2) -> tuple[str, int, str]:
    """Inspect the call stack and return ``(filename, lineno, func_name)``.

    *skip_frames* controls how many frames to skip; the default (2) skips
    ``get_caller_info`` itself and the immediate caller, which is usually
    the patcher wrapper.
    """
    try:
        frame = inspect.currentframe()
        for _ in range(skip_frames):
            if frame is not None:
                frame = frame.f_back
        if frame is not None:
            info = inspect.getframeinfo(frame)
            return (info.filename, info.lineno, info.function)
    except Exception:
        pass
    return ("<unknown>", 0, "<unknown>")


def hash_content(content: Any) -> str:
    """Return a hex SHA-256 digest of *content* (stringified first)."""
    try:
        raw = str(content).encode("utf-8", errors="replace")
        return hashlib.sha256(raw).hexdigest()
    except Exception:
        return ""


def get_data_shape(obj: Any) -> str:
    """Return a short string describing the *shape/type* of *obj*.

    The goal is to capture structural information (type, length, keys)
    without leaking the actual content.
    """
    try:
        if obj is None:
            return "None"
        if isinstance(obj, str):
            return f"str(len={len(obj)})"
        if isinstance(obj, bytes):
            return f"bytes(len={len(obj)})"
        if isinstance(obj, (int, float, bool)):
            return type(obj).__name__
        if isinstance(obj, dict):
            keys = sorted(obj.keys())[:10]
            suffix = ", ..." if len(obj) > 10 else ""
            return f"dict(keys=[{', '.join(repr(k) for k in keys)}{suffix}], len={len(obj)})"
        if isinstance(obj, (list, tuple)):
            inner = get_data_shape(obj[0]) if len(obj) > 0 else "empty"
            return f"{type(obj).__name__}(len={len(obj)}, inner={inner})"
        # Fallback: class name
        return type(obj).__qualname__
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Singleton EventLogger
# ---------------------------------------------------------------------------

class EventLogger:
    """Append-only JSONL event writer.  Singleton — use ``EventLogger.get()``."""

    _instance: EventLogger | None = None
    _init_lock = threading.Lock()

    def __init__(self) -> None:
        # Called only once via get()
        self._file_path: str = os.environ.get(
            "STRATUM_EVENTS_FILE", "/app/stratum_events.jsonl"
        )
        self._lock = threading.Lock()
        self._run_id: str = os.environ.get("STRATUM_RUN_ID", "unknown")
        self._repo_id: str = os.environ.get("STRATUM_REPO_ID", "unknown")
        self._framework: str = os.environ.get("STRATUM_FRAMEWORK", "unknown")
        # Ensure directory exists
        try:
            parent = os.path.dirname(self._file_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
        except Exception:
            pass

    # -- singleton accessor ---------------------------------------------------

    @classmethod
    def get(cls) -> EventLogger:
        """Return the global singleton, creating it on first call."""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # -- public API -----------------------------------------------------------

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def repo_id(self) -> str:
        return self._repo_id

    @property
    def framework(self) -> str:
        return self._framework

    def log_event(
        self,
        event_type: str,
        *,
        source_node: dict[str, str] | None = None,
        target_node: dict[str, str] | None = None,
        edge_type: str | None = None,
        payload: dict[str, Any] | None = None,
        parent_event_id: str | None = None,
        stack_depth: int = 0,
    ) -> str:
        """Write one JSON-line event.  Returns the generated ``event_id``."""
        event_id = f"evt_{uuid.uuid4().hex[:16]}"
        record: dict[str, Any] = {
            "event_id": event_id,
            "timestamp_ns": time.time_ns(),
            "run_id": self._run_id,
            "repo_id": self._repo_id,
            "framework": self._framework,
            "event_type": event_type,
        }
        if source_node is not None:
            record["source_node"] = source_node
        if target_node is not None:
            record["target_node"] = target_node
        if edge_type is not None:
            record["edge_type"] = edge_type
        if payload:
            record["payload"] = payload
        if parent_event_id is not None:
            record["parent_event_id"] = parent_event_id
        record["stack_depth"] = stack_depth

        line = json.dumps(record, default=str, ensure_ascii=False)

        with self._lock:
            try:
                with open(self._file_path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except Exception:
                # Never crash the host program because of logging failures.
                pass

        return event_id
