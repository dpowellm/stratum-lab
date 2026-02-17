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
from typing import Any, Dict, Optional


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
# Semantic content capture utilities
# ---------------------------------------------------------------------------

_CLASSIFICATION_KEYS = frozenset({
    "label", "class", "category", "classification", "type",
    "action", "decision", "next_step", "route", "intent",
    "sentiment", "priority", "severity", "status",
})


def capture_output_signature(content: Any) -> Dict[str, Any]:
    """Capture a lightweight semantic signature of agent/LLM output.

    Does NOT store full content.  Stores enough to:
      1. Detect whether same input -> same output across runs (semantic determinism)
      2. Track which output hash flows to which downstream input (lineage)
      3. Classify output type/structure (for schema mismatch detection)
    """
    if content is None:
        return {"type": "null", "hash": None, "size_bytes": 0,
                "preview": "", "structure": None, "classification_fields": None}

    content_str = json.dumps(content, default=str) if not isinstance(content, str) else content

    return {
        "type": _classify_content_type(content),
        "hash": hash_content(content_str),
        "size_bytes": len(content_str.encode("utf-8", errors="replace")),
        "preview": content_str,
        "structure": _extract_structure(content),
        "classification_fields": _extract_classification_fields(content),
    }


def _classify_content_type(content: Any) -> str:
    """Classify output: classification, routing_decision, scored_output, structured_json, text."""
    if isinstance(content, dict):
        keys = {k.lower() for k in content.keys()}
        if keys & {"label", "class", "category", "classification", "type", "intent"}:
            return "classification"
        if keys & {"action", "decision", "next_step", "route"}:
            return "routing_decision"
        if keys & {"score", "rating", "confidence", "probability"}:
            return "scored_output"
        return "structured_json"
    elif isinstance(content, str):
        return "long_text" if len(content) > 500 else "short_text"
    return "unknown"


def _extract_structure(content: Any) -> Optional[Dict]:
    """Extract structural skeleton — key names and value types, not values."""
    if isinstance(content, dict):
        return {k: type(v).__name__ for k, v in content.items()}
    elif isinstance(content, list) and content:
        return {"length": len(content), "item_type": type(content[0]).__name__}
    return None


def _extract_classification_fields(content: Any) -> Optional[Dict]:
    """If output contains classification/routing decisions, extract them."""
    if not isinstance(content, dict):
        return None
    found = {}
    for key, value in content.items():
        if key.lower() in _CLASSIFICATION_KEYS:
            found[key] = str(value)[:100]
    return found if found else None


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
        self._active_node_stack: list[str] = []
        self._error_context_stack: list[dict[str, Any]] = []
        self._edge_activations: list[dict[str, Any]] = []
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

    def push_active_node(self, node_id: str) -> None:
        """Called when a node (agent) begins execution."""
        self._active_node_stack.append(node_id)

    def pop_active_node(self) -> str:
        """Called when a node (agent) finishes execution."""
        if self._active_node_stack:
            return self._active_node_stack.pop()
        return ""

    def current_node(self) -> str:
        """Return the currently-executing node ID."""
        return self._active_node_stack[-1] if self._active_node_stack else ""

    def parent_node(self) -> str:
        """Return the parent (calling) node, or empty string."""
        return self._active_node_stack[-2] if len(self._active_node_stack) >= 2 else ""

    def record_edge_activation(self, source: str, target: str, data_hash: str = "") -> None:
        """Record that an edge was activated (delegation or data flow)."""
        self._edge_activations.append({
            "source": source,
            "target": target,
            "timestamp": time.time(),
            "data_hash": data_hash,
            "run_id": self._run_id,
        })

    def record_error_context(self, node_id: str, error_type: str, error_msg: str,
                             upstream_node: str = "", upstream_output_hash: str = "") -> None:
        """Record error with causal context for propagation tracing."""
        self._error_context_stack.append({
            "node_id": node_id,
            "error_type": error_type,
            "error_msg": error_msg[:500],
            "upstream_node": upstream_node,
            "upstream_output_hash": upstream_output_hash,
            "timestamp": time.time(),
            "active_stack": list(self._active_node_stack),
        })


def classify_error(error: BaseException) -> str:
    """Classify an error into behavioral categories."""
    error_str = str(error).lower()
    error_type = type(error).__name__

    if "timeout" in error_str or "timed out" in error_str:
        return "timeout"
    if "key" in error_str and ("missing" in error_str or "not found" in error_str):
        return "schema_mismatch"
    if "json" in error_str and ("decode" in error_str or "parse" in error_str):
        return "schema_mismatch"
    if "type" in error_str and "expected" in error_str:
        return "schema_mismatch"
    if "rate" in error_str and "limit" in error_str:
        return "rate_limit"
    if "api" in error_str or "http" in error_str or "connection" in error_str:
        return "api_error"
    if "permission" in error_str or "access" in error_str:
        return "permission_error"
    if error_type in ("KeyError", "TypeError", "ValueError", "AttributeError"):
        return "schema_mismatch"
    if error_type in ("TimeoutError", "asyncio.TimeoutError"):
        return "timeout"
    return "runtime_error"
