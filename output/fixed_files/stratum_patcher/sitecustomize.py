"""Auto-load stratum patcher at Python startup.

This file is placed on ``PYTHONPATH`` (at ``/opt/stratum/``) so that
CPython imports it automatically before any user code runs.

Activation order:
  1. LLM redirect layer (patches constructors to route to vLLM)
  2. Stratum patcher (observes/instruments framework calls)

The redirect MUST happen first so that when the patcher wraps calls,
they are already pointed at the vLLM endpoint.

Ordering guarantee: sitecustomize.py runs before ANY user code imports.
This means all LLM constructors (OpenAI, ChatOpenAI, crewai.LLM, etc.)
are already patched by the time repo code creates instances.  Framework
packages themselves do NOT create singleton LLM clients at import time,
so no calls are missed.
"""

# 1. Activate LLM redirect BEFORE patcher
try:
    from stratum_patcher.llm_redirect import activate as _activate_redirect
    _activate_redirect()
except Exception:
    pass

# 2. Activate stratum patcher (observes calls)
try:
    import stratum_patcher  # noqa: F401
except Exception:
    # Never let patcher loading crash the Python interpreter.
    pass
