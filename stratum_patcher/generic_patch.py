"""Generic patches that apply regardless of AI framework.

Instruments common runtime operations to capture cross-cutting behavioral
signals:

* ``requests.get/post/put/delete/patch``  -> ``external.call`` events
* ``httpx.Client`` + ``AsyncClient``      -> ``external.call`` events
* ``sys.excepthook``                      -> ``error.occurred`` events
* ``builtins.open``                       -> ``file.read`` / ``file.write`` events
  (only for paths under ``/app/``; system files are ignored)
"""

from __future__ import annotations

import builtins
import functools
import os
import sys
import time
import traceback
from typing import Any
from urllib.parse import urlparse

from stratum_patcher.event_logger import (
    EventLogger,
    get_data_shape,
    make_node,
    generate_node_id,
)

_PATCHED = False
_FRAMEWORK = "generic"

# We only log file I/O under this prefix to avoid noise from system files.
_APP_PREFIX = "/app/"


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _domain_only(url: str) -> str:
    """Return just the domain part of a URL (privacy)."""
    try:
        parsed = urlparse(str(url))
        return parsed.hostname or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# requests wrappers
# ---------------------------------------------------------------------------

def _wrap_requests_method(original: Any, method_name: str) -> Any:
    """Wrap a ``requests.get/post/...`` function."""

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = EventLogger.get()
        url = args[0] if args else kwargs.get("url", "")
        domain = _domain_only(url)
        node_id = generate_node_id(_FRAMEWORK, f"http:{domain}", __file__, 0)
        source = make_node("external", node_id, domain)

        start_id = logger.log_event(
            "external.call",
            source_node=source,
            payload={
                "method": method_name.upper(),
                "domain": domain,
                "phase": "start",
            },
        )

        t0 = time.perf_counter()
        error: Exception | None = None
        result: Any = None
        try:
            result = original(*args, **kwargs)
            return result
        except Exception as exc:
            error = exc
            raise
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            payload: dict[str, Any] = {
                "method": method_name.upper(),
                "domain": domain,
                "latency_ms": round(latency_ms, 2),
                "phase": "end",
            }
            if error:
                payload["error"] = str(error)[:300]
                payload["error_type"] = type(error).__name__
            elif result is not None:
                try:
                    payload["status_code"] = getattr(result, "status_code", None)
                except Exception:
                    pass
            logger.log_event(
                "external.call",
                source_node=source,
                payload=payload,
                parent_event_id=start_id,
            )

    return wrapper


# ---------------------------------------------------------------------------
# httpx wrappers
# ---------------------------------------------------------------------------

def _wrap_httpx_method(original: Any, method_name: str) -> Any:
    """Wrap an ``httpx.Client.get/post/...`` method (sync)."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        logger = EventLogger.get()
        url = args[0] if args else kwargs.get("url", "")
        domain = _domain_only(url)
        node_id = generate_node_id(_FRAMEWORK, f"httpx:{domain}", __file__, 0)
        source = make_node("external", node_id, domain)

        start_id = logger.log_event(
            "external.call",
            source_node=source,
            payload={
                "method": method_name.upper(),
                "domain": domain,
                "library": "httpx",
                "phase": "start",
            },
        )

        t0 = time.perf_counter()
        error: Exception | None = None
        result: Any = None
        try:
            result = original(self, *args, **kwargs)
            return result
        except Exception as exc:
            error = exc
            raise
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            payload: dict[str, Any] = {
                "method": method_name.upper(),
                "domain": domain,
                "library": "httpx",
                "latency_ms": round(latency_ms, 2),
                "phase": "end",
            }
            if error:
                payload["error"] = str(error)[:300]
                payload["error_type"] = type(error).__name__
            elif result is not None:
                try:
                    payload["status_code"] = getattr(result, "status_code", None)
                except Exception:
                    pass
            logger.log_event(
                "external.call",
                source_node=source,
                payload=payload,
                parent_event_id=start_id,
            )

    return wrapper


def _wrap_httpx_async_method(original: Any, method_name: str) -> Any:
    """Wrap an ``httpx.AsyncClient.get/post/...`` method."""

    @functools.wraps(original)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        logger = EventLogger.get()
        url = args[0] if args else kwargs.get("url", "")
        domain = _domain_only(url)
        node_id = generate_node_id(_FRAMEWORK, f"httpx_async:{domain}", __file__, 0)
        source = make_node("external", node_id, domain)

        start_id = logger.log_event(
            "external.call",
            source_node=source,
            payload={
                "method": method_name.upper(),
                "domain": domain,
                "library": "httpx.async",
                "phase": "start",
            },
        )

        t0 = time.perf_counter()
        error: Exception | None = None
        result: Any = None
        try:
            result = await original(self, *args, **kwargs)
            return result
        except Exception as exc:
            error = exc
            raise
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            payload: dict[str, Any] = {
                "method": method_name.upper(),
                "domain": domain,
                "library": "httpx.async",
                "latency_ms": round(latency_ms, 2),
                "phase": "end",
            }
            if error:
                payload["error"] = str(error)[:300]
                payload["error_type"] = type(error).__name__
            elif result is not None:
                try:
                    payload["status_code"] = getattr(result, "status_code", None)
                except Exception:
                    pass
            logger.log_event(
                "external.call",
                source_node=source,
                payload=payload,
                parent_event_id=start_id,
            )

    return wrapper


# ---------------------------------------------------------------------------
# builtins.open wrapper
# ---------------------------------------------------------------------------

_original_open = builtins.open  # save before patching


def _wrap_open(original: Any) -> Any:
    """Wrap ``builtins.open`` to log file read/write under /app/."""

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Determine file path and mode
        file_path = str(args[0]) if args else str(kwargs.get("file", ""))
        mode = args[1] if len(args) > 1 else kwargs.get("mode", "r")
        mode_str = str(mode) if mode else "r"

        # Only log for /app/ paths (skip system, patcher, and library files)
        should_log = file_path.startswith(_APP_PREFIX)

        # Also skip our own events file to avoid recursion
        events_file = os.environ.get("STRATUM_EVENTS_FILE", "/app/stratum_events.jsonl")
        if file_path == events_file:
            should_log = False

        if should_log:
            try:
                logger = EventLogger.get()
                is_write = any(c in mode_str for c in ("w", "a", "x", "+"))
                event_type = "file.write" if is_write else "file.read"
                node_id = generate_node_id(_FRAMEWORK, "file_io", file_path, 0)
                source = make_node("data_store", node_id, file_path)
                logger.log_event(
                    event_type,
                    source_node=source,
                    payload={
                        "path": file_path,
                        "mode": mode_str,
                    },
                )
            except Exception:
                pass

        return original(*args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# sys.excepthook wrapper
# ---------------------------------------------------------------------------

def _wrap_excepthook(original: Any) -> Any:
    """Wrap ``sys.excepthook`` to log unhandled exceptions."""

    @functools.wraps(original)
    def wrapper(exc_type: type, exc_value: BaseException, exc_tb: Any) -> Any:
        try:
            logger = EventLogger.get()
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
            tb_str = "".join(tb_lines)[-2000:]  # truncate

            # Extract the file/line of the exception
            tb_file = "<unknown>"
            tb_line = 0
            if exc_tb is not None:
                try:
                    # Walk to the innermost frame
                    tb = exc_tb
                    while tb.tb_next:
                        tb = tb.tb_next
                    tb_file = tb.tb_frame.f_code.co_filename
                    tb_line = tb.tb_lineno
                except Exception:
                    pass

            node_id = generate_node_id(_FRAMEWORK, "exception", tb_file, tb_line)
            source = make_node("agent", node_id, "unhandled_exception")

            logger.log_event(
                "error.occurred",
                source_node=source,
                payload={
                    "error_type": exc_type.__name__ if exc_type else "unknown",
                    "error_message": str(exc_value)[:500],
                    "traceback_tail": tb_str[-1000:],
                    "file": tb_file,
                    "line": tb_line,
                },
            )
        except Exception:
            pass  # never let our hook crash

        return original(exc_type, exc_value, exc_tb)

    return wrapper


# ---------------------------------------------------------------------------
# Patch entry point
# ---------------------------------------------------------------------------

def patch() -> None:
    """Apply generic monkey-patches.  Idempotent."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    # ------------------------------------------------------------------
    # 1) requests library
    # ------------------------------------------------------------------
    try:
        import requests

        for method_name in ("get", "post", "put", "delete", "patch", "head", "options"):
            orig = getattr(requests, method_name, None)
            if orig is not None and not getattr(orig, "_stratum_patched", False):
                wrapped = _wrap_requests_method(orig, method_name)
                wrapped._stratum_patched = True  # type: ignore[attr-defined]
                setattr(requests, method_name, wrapped)
    except ImportError:
        pass
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 2) httpx sync client
    # ------------------------------------------------------------------
    try:
        import httpx

        for method_name in ("get", "post", "put", "delete", "patch", "head", "options"):
            # Sync client
            try:
                orig = getattr(httpx.Client, method_name, None)
                if orig is not None and not getattr(orig, "_stratum_patched", False):
                    wrapped = _wrap_httpx_method(orig, method_name)
                    wrapped._stratum_patched = True  # type: ignore[attr-defined]
                    setattr(httpx.Client, method_name, wrapped)
            except Exception:
                pass

            # Async client
            try:
                orig = getattr(httpx.AsyncClient, method_name, None)
                if orig is not None and not getattr(orig, "_stratum_patched", False):
                    wrapped = _wrap_httpx_async_method(orig, method_name)
                    wrapped._stratum_patched = True  # type: ignore[attr-defined]
                    setattr(httpx.AsyncClient, method_name, wrapped)
            except Exception:
                pass
    except ImportError:
        pass
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 3) sys.excepthook
    # ------------------------------------------------------------------
    try:
        orig_hook = sys.excepthook
        if not getattr(orig_hook, "_stratum_patched", False):
            wrapped = _wrap_excepthook(orig_hook)
            wrapped._stratum_patched = True  # type: ignore[attr-defined]
            sys.excepthook = wrapped
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 4) builtins.open
    # ------------------------------------------------------------------
    try:
        orig_open = builtins.open
        if not getattr(orig_open, "_stratum_patched", False):
            wrapped = _wrap_open(orig_open)
            wrapped._stratum_patched = True  # type: ignore[attr-defined]
            builtins.open = wrapped  # type: ignore[assignment]
    except Exception:
        pass


# Auto-patch on import
patch()
