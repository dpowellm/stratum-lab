#!/usr/bin/env python3
"""Stratum runner — entry-point wrapper for instrumented execution.

This script is the Docker ENTRYPOINT.  It:

1. Imports ``stratum_patcher`` (which triggers all monkey-patches).
2. Sets up a timeout alarm (``STRATUM_TIMEOUT_SECONDS``).
3. Runs the repo's entry point via ``runpy.run_path`` or ``exec``.
4. Captures exit code and logs ``execution.end`` on timeout or crash.
"""

from __future__ import annotations

import importlib
import os
import signal
import sys
import time
import traceback
from typing import Any


def _setup_timeout(timeout_seconds: int) -> None:
    """Install a SIGALRM handler that raises SystemExit on timeout.

    On Windows (or when SIGALRM is unavailable), fall back to a threading
    timer that calls ``os._exit`` — brutal but functional in a container.
    """
    if hasattr(signal, "SIGALRM"):
        def _alarm_handler(signum: int, frame: Any) -> None:
            raise SystemExit(f"STRATUM TIMEOUT after {timeout_seconds}s")

        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(timeout_seconds)
    else:
        import threading

        def _timeout_kill() -> None:
            # Log timeout event then hard-exit
            try:
                from stratum_patcher.event_logger import EventLogger, make_node, generate_node_id
                logger = EventLogger.get()
                logger.log_event(
                    "execution.end",
                    payload={
                        "status": "timeout",
                        "timeout_seconds": timeout_seconds,
                    },
                )
            except Exception:
                pass
            os._exit(124)  # 124 = conventional timeout exit code

        timer = threading.Timer(float(timeout_seconds), _timeout_kill)
        timer.daemon = True
        timer.start()


def main() -> None:
    """Run the target entry point with stratum instrumentation active."""

    # ---------------------------------------------------------------
    # 0) Determine the entry-point path from CLI args
    # ---------------------------------------------------------------
    if len(sys.argv) < 2:
        print("Usage: runner.py <entry_point.py> [args ...]", file=sys.stderr)
        sys.exit(1)

    entry_point = sys.argv[1]
    # Pass remaining args through to the child script
    sys.argv = sys.argv[1:]

    # ---------------------------------------------------------------
    # 1) Import stratum_patcher -> applies all monkey-patches
    # ---------------------------------------------------------------
    try:
        import stratum_patcher  # noqa: F401
    except Exception as exc:
        print(f"[stratum-runner] WARNING: failed to load patcher: {exc}", file=sys.stderr)

    # ---------------------------------------------------------------
    # 2) Set up timeout
    # ---------------------------------------------------------------
    timeout_str = os.environ.get("STRATUM_TIMEOUT_SECONDS", "600")
    try:
        timeout_seconds = int(timeout_str)
    except ValueError:
        timeout_seconds = 600

    _setup_timeout(timeout_seconds)

    # ---------------------------------------------------------------
    # 3) Log execution start
    # ---------------------------------------------------------------
    from stratum_patcher.event_logger import EventLogger, make_node, generate_node_id

    logger = EventLogger.get()
    node_id = generate_node_id("runner", "main", entry_point, 0)
    source = make_node("agent", node_id, "stratum_runner")

    start_id = logger.log_event(
        "execution.start",
        source_node=source,
        payload={
            "entry_point": entry_point,
            "run_id": logger.run_id,
            "repo_id": logger.repo_id,
            "framework": logger.framework,
            "timeout_seconds": timeout_seconds,
            "python_version": sys.version,
        },
    )

    # ---------------------------------------------------------------
    # 4) Execute the entry point
    # ---------------------------------------------------------------
    t0 = time.perf_counter()
    exit_code = 0
    error_msg: str | None = None
    status = "success"

    try:
        # Ensure the entry point's directory is on sys.path so its
        # relative imports work.
        entry_dir = os.path.dirname(os.path.abspath(entry_point))
        if entry_dir not in sys.path:
            sys.path.insert(0, entry_dir)

        import runpy
        runpy.run_path(entry_point, run_name="__main__")

    except SystemExit as exc:
        # The child called sys.exit() — honour the code.
        code = exc.code
        if isinstance(code, int):
            exit_code = code
        elif isinstance(code, str):
            # Timeout or error message
            if "TIMEOUT" in code.upper():
                status = "timeout"
                exit_code = 124
                error_msg = code
            else:
                exit_code = 1
                error_msg = code
        elif code is None:
            exit_code = 0
        else:
            exit_code = 1
            error_msg = str(code)

        if exit_code != 0 and status != "timeout":
            status = "error"

    except Exception:
        exit_code = 1
        status = "crash"
        error_msg = traceback.format_exc()[-2000:]

    finally:
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Cancel alarm if it was set
        try:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
        except Exception:
            pass

    # ---------------------------------------------------------------
    # 5) Log execution end
    # ---------------------------------------------------------------
    try:
        payload: dict[str, Any] = {
            "entry_point": entry_point,
            "exit_code": exit_code,
            "status": status,
            "latency_ms": round(latency_ms, 2),
        }
        if error_msg:
            payload["error"] = error_msg[:2000]
        logger.log_event(
            "execution.end",
            source_node=source,
            payload=payload,
            parent_event_id=start_id,
        )
    except Exception:
        pass

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
