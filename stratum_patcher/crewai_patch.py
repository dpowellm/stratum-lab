"""Monkey-patch for crewAI internals.

Instruments the core execution path of crewAI:

* ``Crew.kickoff()``       -> ``execution.start`` / ``execution.end``
* ``Agent.execute_task()``  -> ``agent.task_start`` / ``agent.task_end``
* Tool ``_run()`` methods   -> ``tool.invoked``    / ``tool.completed``
* Delegation mechanism      -> ``delegation.initiated`` / ``delegation.completed``

All events are logged through the stratum EventLogger singleton.
"""

from __future__ import annotations

import functools
import time
import traceback
from typing import Any

from stratum_patcher.event_logger import (
    EventLogger,
    capture_output_signature,
    classify_error,
    get_data_shape,
    hash_content,
    make_node,
    generate_node_id,
)

_PATCHED = False
_FRAMEWORK = "crewai"


# ---------------------------------------------------------------------------
# Crew.kickoff wrapper
# ---------------------------------------------------------------------------

def _wrap_crew_kickoff(original: Any) -> Any:
    """Wrap ``Crew.kickoff`` to log execution start/end."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        logger = EventLogger.get()
        crew_name = getattr(self, "name", None) or type(self).__name__
        node_id = generate_node_id(_FRAMEWORK, crew_name, __file__, 0)
        source = make_node("agent", node_id, crew_name)

        # Gather metadata about the crew composition
        agents_info: list[str] = []
        tasks_info: list[str] = []
        try:
            for a in getattr(self, "agents", []):
                agents_info.append(getattr(a, "role", type(a).__name__))
            for t in getattr(self, "tasks", []):
                desc = getattr(t, "description", "")
                tasks_info.append(hash_content(desc))
        except Exception:
            pass

        start_id = logger.log_event(
            "execution.start",
            source_node=source,
            payload={
                "crew_name": crew_name,
                "agent_count": len(agents_info),
                "task_count": len(tasks_info),
                "agent_roles": agents_info,
                "task_hashes": tasks_info,
                "process_type": str(getattr(self, "process", "sequential")),
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
                "latency_ms": round(latency_ms, 2),
                "status": "error" if error else "success",
            }
            if error:
                payload["error"] = str(error)[:500]
                payload["error_type"] = type(error).__name__
            if result is not None:
                payload["result_shape"] = get_data_shape(result)
            logger.log_event(
                "execution.end",
                source_node=source,
                payload=payload,
                parent_event_id=start_id,
            )

    return wrapper


# ---------------------------------------------------------------------------
# Agent.execute_task wrapper
# ---------------------------------------------------------------------------

def _wrap_agent_execute_task(original: Any) -> Any:
    """Wrap ``Agent.execute_task`` to log task start/end per agent."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        logger = EventLogger.get()
        agent_role = getattr(self, "role", type(self).__name__)
        node_id = generate_node_id(_FRAMEWORK, agent_role, __file__, 0)
        source = make_node("agent", node_id, agent_role)
        logger.push_active_node(node_id)

        # Try to get task info from args
        task_desc_hash = ""
        task_name = "unknown"
        try:
            task = args[0] if args else kwargs.get("task")
            if task is not None:
                task_desc = getattr(task, "description", "")
                task_name = getattr(task, "name", "") or hash_content(task_desc)[:12]
                task_desc_hash = hash_content(task_desc)
        except Exception:
            pass

        # Tools available to this agent
        tool_names: list[str] = []
        try:
            for t in getattr(self, "tools", []):
                tool_names.append(getattr(t, "name", type(t).__name__))
        except Exception:
            pass

        start_id = logger.log_event(
            "agent.task_start",
            source_node=source,
            payload={
                "agent_role": agent_role,
                "task_name": task_name,
                "task_description_hash": task_desc_hash,
                "tools_available": tool_names,
                "agent_goal_hash": hash_content(getattr(self, "goal", "")),
                "node_id": node_id,
                "parent_node_id": logger.parent_node(),
                "input_source": "delegation",
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
                "agent_role": agent_role,
                "latency_ms": round(latency_ms, 2),
                "status": "error" if error else "success",
            }
            if error:
                payload["error"] = str(error)[:500]
                payload["error_type"] = type(error).__name__
            payload["node_id"] = node_id
            payload["output_data_hash"] = hash_content(result) if result is not None else None
            payload["error_type"] = classify_error(error) if error else None
            if result is not None:
                payload["result_shape"] = get_data_shape(result)
                try:
                    _sig = capture_output_signature(result)
                    payload["output_hash"] = _sig["hash"]
                    payload["output_type"] = _sig["type"]
                    payload["output_size_bytes"] = _sig["size_bytes"]
                    payload["output_preview"] = _sig["preview"]
                    payload["classification_fields"] = _sig["classification_fields"]
                except Exception:
                    pass
            logger.log_event(
                "agent.task_end",
                source_node=source,
                payload=payload,
                parent_event_id=start_id,
            )
            logger.pop_active_node()
            if error:
                logger.record_error_context(
                    node_id=node_id,
                    error_type=classify_error(error),
                    error_msg=str(error),
                    upstream_node=logger.parent_node(),
                )

    return wrapper


# ---------------------------------------------------------------------------
# Tool._run wrapper
# ---------------------------------------------------------------------------

def _wrap_tool_run(original: Any, tool_instance: Any) -> Any:
    """Wrap a tool's ``_run`` method to log invocations."""

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = EventLogger.get()
        tool_name = getattr(tool_instance, "name", type(tool_instance).__name__)
        node_id = generate_node_id(_FRAMEWORK, tool_name, __file__, 0)
        source = make_node("capability", node_id, tool_name)

        start_id = logger.log_event(
            "tool.invoked",
            source_node=source,
            payload={
                "tool_name": tool_name,
                "args_shape": get_data_shape(args),
                "kwargs_shape": get_data_shape(kwargs),
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
                "tool_name": tool_name,
                "latency_ms": round(latency_ms, 2),
                "status": "error" if error else "success",
            }
            if error:
                payload["error"] = str(error)[:500]
                payload["error_type"] = type(error).__name__
            if result is not None:
                payload["result_shape"] = get_data_shape(result)
            logger.log_event(
                "tool.completed",
                source_node=source,
                payload=payload,
                parent_event_id=start_id,
            )

    return wrapper


# ---------------------------------------------------------------------------
# Delegation wrapper
# ---------------------------------------------------------------------------

def _wrap_delegate_work(original: Any) -> Any:
    """Wrap crewAI's delegation mechanism."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        logger = EventLogger.get()
        delegator_role = getattr(self, "role", type(self).__name__)
        delegator_id = generate_node_id(_FRAMEWORK, delegator_role, __file__, 0)
        source = make_node("agent", delegator_id, delegator_role)

        # Try to determine the target agent
        target_role = "unknown"
        try:
            # crewAI delegation typically passes coworker/task_description
            coworker = kwargs.get("coworker") or (args[0] if args else None)
            if coworker is not None:
                if isinstance(coworker, str):
                    target_role = coworker
                else:
                    target_role = getattr(coworker, "role", type(coworker).__name__)
        except Exception:
            pass

        target_id = generate_node_id(_FRAMEWORK, target_role, __file__, 0)
        target = make_node("agent", target_id, target_role)

        # Capture context being passed to delegate
        _task_context = kwargs.get("task", args[1] if len(args) > 1 else "")
        _context_sig = capture_output_signature(_task_context)
        start_id = logger.log_event(
            "delegation.initiated",
            source_node=source,
            target_node=target,
            edge_type="delegates_to",
            payload={
                "delegator": delegator_role,
                "delegate": target_role,
                "task_hash": hash_content(_task_context),
                "context_hash": _context_sig["hash"],
                "context_type": _context_sig["type"],
                "context_size_bytes": _context_sig["size_bytes"],
                "context_source_node": delegator_id,
                "has_classification_dependency": _context_sig["classification_fields"] is not None,
                "source_node_id": delegator_id,
                "target_node_id": target_id,
                "delegation_type": "explicit",
                "input_data_hash": hash_content(_task_context),
            },
        )
        logger.record_edge_activation(source=delegator_id, target=target_id, data_hash=hash_content(_task_context))
        _log_routing_decision(delegator_id, target_id, "manager_delegation", "explicit")

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
                "delegator": delegator_role,
                "delegate": target_role,
                "latency_ms": round(latency_ms, 2),
                "status": "error" if error else "success",
            }
            if error:
                payload["error"] = str(error)[:500]
            if result is not None:
                payload["result_shape"] = get_data_shape(result)
            logger.log_event(
                "delegation.completed",
                source_node=source,
                target_node=target,
                edge_type="delegates_to",
                payload=payload,
                parent_event_id=start_id,
            )

    return wrapper


# ---------------------------------------------------------------------------
# Tool class patching helper
# ---------------------------------------------------------------------------

def _patch_tool_class(tool_cls: type) -> None:
    """Patch the ``_run`` method on a crewAI tool *class*."""
    try:
        orig_run = tool_cls._run  # type: ignore[attr-defined]
        if getattr(orig_run, "_stratum_patched", False):
            return

        @functools.wraps(orig_run)
        def patched_run(self: Any, *a: Any, **kw: Any) -> Any:
            # Build a per-instance wrapper on first call
            bound = _wrap_tool_run(orig_run.__get__(self, type(self)), self)
            return bound(*a, **kw)

        patched_run._stratum_patched = True  # type: ignore[attr-defined]
        tool_cls._run = patched_run  # type: ignore[attr-defined]
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Implicit interaction detection helpers
# ---------------------------------------------------------------------------

def _log_state_access(node_id: str, state_key: str, access_type: str, data_hash: str = "") -> None:
    """Log access to shared state for implicit interaction detection."""
    logger = EventLogger.get()
    logger.log_event(
        "state.access",
        payload={
            "node_id": node_id,
            "state_key": state_key,
            "access_type": access_type,
            "data_hash": data_hash,
        },
    )


def _log_routing_decision(source_node: str, target_node: str, routing_type: str,
                          decision_basis: str = "") -> None:
    """Log a routing decision for emergent edge detection."""
    logger = EventLogger.get()
    logger.log_event(
        "routing.decision",
        payload={
            "source_node": source_node,
            "target_node": target_node,
            "routing_type": routing_type,
            "decision_basis": decision_basis,
        },
    )


# ---------------------------------------------------------------------------
# Patch entry point
# ---------------------------------------------------------------------------

def patch() -> None:
    """Apply monkey-patches to crewAI internals.  Idempotent."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    try:
        import crewai  # noqa: F811
    except ImportError:
        return

    # ------------------------------------------------------------------
    # 1) Crew.kickoff
    # ------------------------------------------------------------------
    try:
        from crewai import Crew

        orig = Crew.kickoff
        if not getattr(orig, "_stratum_patched", False):
            wrapped = _wrap_crew_kickoff(orig)
            wrapped._stratum_patched = True  # type: ignore[attr-defined]
            Crew.kickoff = wrapped  # type: ignore[attr-defined]
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 2) Agent.execute_task
    # ------------------------------------------------------------------
    try:
        from crewai import Agent

        orig = Agent.execute_task
        if not getattr(orig, "_stratum_patched", False):
            wrapped = _wrap_agent_execute_task(orig)
            wrapped._stratum_patched = True  # type: ignore[attr-defined]
            Agent.execute_task = wrapped  # type: ignore[attr-defined]
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 3) Tool._run  (base class)
    # ------------------------------------------------------------------
    try:
        from crewai.tools import BaseTool as CrewBaseTool
        _patch_tool_class(CrewBaseTool)
    except Exception:
        pass

    # Also try langchain-style tools that crewAI often wraps
    try:
        from langchain_core.tools import BaseTool as LCBaseTool
        _patch_tool_class(LCBaseTool)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 4) Delegation
    # ------------------------------------------------------------------
    try:
        # crewAI >= 0.30 delegation via AgentTools or DelegateWork
        from crewai.tools.agent_tools import DelegateWorkTool
        orig = DelegateWorkTool._run
        if not getattr(orig, "_stratum_patched", False):
            @functools.wraps(orig)
            def delegate_run(self: Any, *a: Any, **kw: Any) -> Any:
                bound = _wrap_tool_run(orig.__get__(self, type(self)), self)
                # Also log delegation-specific events
                logger = EventLogger.get()
                logger.log_event(
                    "delegation.initiated",
                    payload={
                        "tool_name": "DelegateWorkTool",
                        "args_shape": get_data_shape(a),
                    },
                )
                return bound(*a, **kw)
            delegate_run._stratum_patched = True  # type: ignore[attr-defined]
            DelegateWorkTool._run = delegate_run  # type: ignore[attr-defined]
    except Exception:
        pass

    # Try the older delegation path
    try:
        from crewai import Agent as _Agent2
        if hasattr(_Agent2, "delegate_work"):
            orig = _Agent2.delegate_work
            if not getattr(orig, "_stratum_patched", False):
                wrapped = _wrap_delegate_work(orig)
                wrapped._stratum_patched = True  # type: ignore[attr-defined]
                _Agent2.delegate_work = wrapped  # type: ignore[attr-defined]
    except Exception:
        pass

    # Try execute_task delegation path used in some versions
    try:
        from crewai import Agent as _Agent3
        if hasattr(_Agent3, "_delegate_task"):
            orig = _Agent3._delegate_task
            if not getattr(orig, "_stratum_patched", False):
                wrapped = _wrap_delegate_work(orig)
                wrapped._stratum_patched = True  # type: ignore[attr-defined]
                _Agent3._delegate_task = wrapped  # type: ignore[attr-defined]
    except Exception:
        pass


# Auto-patch on import
patch()
