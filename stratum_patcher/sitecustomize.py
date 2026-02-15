"""Auto-load stratum patcher at Python startup.

This file is placed on ``PYTHONPATH`` (at ``/opt/stratum/``) so that
CPython imports it automatically before any user code runs.  The single
import triggers all monkey-patches via ``stratum_patcher.__init__``.
"""

try:
    import stratum_patcher  # noqa: F401
except Exception:
    # Never let patcher loading crash the Python interpreter.
    pass
