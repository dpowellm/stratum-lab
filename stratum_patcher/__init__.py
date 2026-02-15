"""Stratum Patcher — import-time monkey-patching for AI agent frameworks.

Importing this package automatically applies all available patches.
Each framework patcher is wrapped in try/except so that missing
dependencies (e.g. ``crewai`` not installed) silently skip rather than
crash the host application.

The ``generic_patch`` module is imported unconditionally because it only
depends on stdlib and ubiquitous libraries (``requests``, ``httpx``).
"""

from __future__ import annotations

# -- Generic patches (always applied) -----------------------------------
from stratum_patcher import generic_patch  # noqa: F401

# -- Framework-specific patches (optional) ------------------------------
try:
    from stratum_patcher import openai_patch  # noqa: F401
except Exception:
    pass

try:
    from stratum_patcher import anthropic_patch  # noqa: F401
except Exception:
    pass

try:
    from stratum_patcher import crewai_patch  # noqa: F401
except Exception:
    pass

try:
    from stratum_patcher import langgraph_patch  # noqa: F401
except Exception:
    pass

try:
    from stratum_patcher import autogen_patch  # noqa: F401
except Exception:
    pass
