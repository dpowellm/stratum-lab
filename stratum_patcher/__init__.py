"""Stratum Patcher — import-time monkey-patching for AI agent frameworks.

Importing this package automatically applies all available patches.
Each framework patcher is wrapped in try/except so that missing
dependencies (e.g. ``crewai`` not installed) silently skip rather than
crash the host application.

The ``generic_patch`` module is imported unconditionally because it only
depends on stdlib and ubiquitous libraries (``requests``, ``httpx``).
"""

from __future__ import annotations

_patcher_status: dict[str, str] = {}

# -- Generic patches (always applied) -----------------------------------
from stratum_patcher import generic_patch  # noqa: F401
_patcher_status["generic"] = "ok"

# -- Framework-specific patches (optional) ------------------------------
try:
    from stratum_patcher import openai_patch  # noqa: F401
    _patcher_status["openai"] = "ok"
except Exception as _e:
    _patcher_status["openai"] = f"skip:{type(_e).__name__}"

try:
    from stratum_patcher import anthropic_patch  # noqa: F401
    _patcher_status["anthropic"] = "ok"
except Exception as _e:
    _patcher_status["anthropic"] = f"skip:{type(_e).__name__}"

try:
    from stratum_patcher import crewai_patch  # noqa: F401
    _patcher_status["crewai"] = "ok"
except Exception as _e:
    _patcher_status["crewai"] = f"skip:{type(_e).__name__}"

try:
    from stratum_patcher import langgraph_patch  # noqa: F401
    _patcher_status["langgraph"] = "ok"
except Exception as _e:
    _patcher_status["langgraph"] = f"skip:{type(_e).__name__}"

try:
    from stratum_patcher import autogen_patch  # noqa: F401
    _patcher_status["autogen"] = "ok"
except Exception as _e:
    _patcher_status["autogen"] = f"skip:{type(_e).__name__}"

try:
    from stratum_patcher import litellm_patch  # noqa: F401
    _patcher_status["litellm"] = "ok"
except Exception as _e:
    _patcher_status["litellm"] = f"skip:{type(_e).__name__}"

# -- Log patcher status event -------------------------------------------
try:
    from stratum_patcher.event_logger import EventLogger
    _logger = EventLogger.get()
    _logger.log_event(
        "patcher.status",
        payload={
            "patches": _patcher_status,
            "patches_ok": sum(1 for v in _patcher_status.values() if v == "ok"),
            "patches_skipped": sum(1 for v in _patcher_status.values() if v.startswith("skip")),
        },
    )
except Exception:
    pass
