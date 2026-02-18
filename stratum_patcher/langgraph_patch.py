"""Monkey-patch for LangGraph.

Instruments the core execution path of LangGraph compiled graphs:

* ``CompiledGraph.invoke()``  -> ``execution.start`` / ``execution.end``
* ``CompiledGraph.stream()``  -> ``execution.start`` / ``execution.end``
* Node function execution     -> ``agent.task_start`` / ``agent.task_end``
* Edge traversal              -> ``edge.traversed`` (including conditional branch info)
"""

from __future__ import annotations

import functools
import inspect
import sys
import time
from typing import Any, Iterator

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
_FRAMEWORK = "langgraph"


def _stderr(msg: str) -> None:
    print(f"stratum_patcher: {msg}", file=sys.stderr, flush=True)


def _get_state_keys(state: Any) -> list[str]:
    """Extract key names from a state object (dict or TypedDict)."""
    try:
        if isinstance(state, dict):
            return sorted(state.keys())[:20]
        if hasattr(state, "__dict__"):
            return sorted(vars(state).keys())[:20]
    except Exception:
        pass
    return []


def _state_diff_keys(before: Any, after: Any) -> list[str]:
    """Return keys whose values changed between two state dicts."""
    try:
        if isinstance(before, dict) and isinstance(after, dict):
            changed = []
            all_keys = set(before.keys()) | set(after.keys())
            for k in sorted(all_keys):
                if before.get(k) != after.get(k):
                    changed.append(k)
            return changed[:20]
    except Exception:
        pass
    return []


def _truncate(text: Any, limit: int = 2000) -> str:
    """Truncate text to limit characters."""
    s = str(text)
    return s[:limit] if len(s) > limit else s


# ---------------------------------------------------------------------------
# CompiledGraph.invoke wrapper
# ---------------------------------------------------------------------------

def _wrap_invoke(original: Any) -> Any:
    """Wrap ``CompiledGraph.invoke`` to log execution start/end."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        logger = EventLogger.get()
        graph_name = getattr(self, "name", None) or type(self).__name__
        node_id = generate_node_id(_FRAMEWORK, graph_name, __file__, 0)
        source = make_node("agent", node_id, graph_name)

        # Gather graph metadata
        node_names: list[str] = []
        try:
            nodes = getattr(self, "nodes", {})
            if isinstance(nodes, dict):
                node_names = list(nodes.keys())
            elif hasattr(nodes, "keys"):
                node_names = list(nodes.keys())
        except Exception:
            pass

        input_shape = get_data_shape(args[0]) if args else get_data_shape(kwargs.get("input"))

        start_id = logger.log_event(
            "execution.start",
            source_node=source,
            payload={
                "graph_name": graph_name,
                "node_count": len(node_names),
                "node_names": node_names[:20],
                "input_shape": input_shape,
                "config_keys": list(kwargs.get("config", {}).keys())[:10] if isinstance(
                    kwargs.get("config"), dict
                ) else [],
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
                "graph_name": graph_name,
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
# CompiledGraph.stream wrapper
# ---------------------------------------------------------------------------

def _wrap_stream(original: Any) -> Any:
    """Wrap ``CompiledGraph.stream`` to log execution start/end + per-node events."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Iterator:
        logger = EventLogger.get()
        graph_name = getattr(self, "name", None) or type(self).__name__
        node_id = generate_node_id(_FRAMEWORK, graph_name, __file__, 0)
        source = make_node("agent", node_id, graph_name)

        input_shape = get_data_shape(args[0]) if args else get_data_shape(kwargs.get("input"))

        start_id = logger.log_event(
            "execution.start",
            source_node=source,
            payload={
                "graph_name": graph_name,
                "mode": "stream",
                "input_shape": input_shape,
            },
        )

        t0 = time.perf_counter()
        error: Exception | None = None
        chunk_count = 0
        try:
            gen = original(self, *args, **kwargs)
            for chunk in gen:
                chunk_count += 1
                # Each streamed chunk is {node_name: output_state}
                try:
                    if isinstance(chunk, dict):
                        for step_name, step_output in chunk.items():
                            step_node_id = generate_node_id(
                                _FRAMEWORK, f"node:{step_name}", __file__, 0
                            )
                            step_source = make_node("agent", step_node_id, step_name)

                            # Emit agent.task_start
                            step_start_id = logger.log_event(
                                "agent.task_start",
                                source_node=step_source,
                                payload={
                                    "agent_name": step_name,
                                    "agent_goal": step_name,
                                    "task_description": f"stream chunk {chunk_count}",
                                    "tools_available": [],
                                    "input_state_keys": [],
                                    "stream_chunk_index": chunk_count,
                                },
                                parent_event_id=start_id,
                            )

                            # Emit agent.task_end
                            output_keys = _get_state_keys(step_output)
                            logger.log_event(
                                "agent.task_end",
                                source_node=step_source,
                                payload={
                                    "agent_name": step_name,
                                    "output_text": _truncate(step_output),
                                    "output_state_keys": output_keys,
                                    "success": True,
                                    "duration_ms": 0,
                                    "stream_chunk_index": chunk_count,
                                },
                                parent_event_id=step_start_id,
                            )
                except Exception:
                    pass
                yield chunk
        except Exception as exc:
            error = exc
            raise
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            payload: dict[str, Any] = {
                "graph_name": graph_name,
                "mode": "stream",
                "latency_ms": round(latency_ms, 2),
                "chunks_yielded": chunk_count,
                "status": "error" if error else "success",
            }
            if error:
                payload["error"] = str(error)[:500]
                payload["error_type"] = type(error).__name__
            logger.log_event(
                "execution.end",
                source_node=source,
                payload=payload,
                parent_event_id=start_id,
            )

    return wrapper


# ---------------------------------------------------------------------------
# Async invoke wrapper
# ---------------------------------------------------------------------------

def _wrap_ainvoke(original: Any) -> Any:
    """Wrap ``CompiledGraph.ainvoke`` for async execution."""

    @functools.wraps(original)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        logger = EventLogger.get()
        graph_name = getattr(self, "name", None) or type(self).__name__
        node_id = generate_node_id(_FRAMEWORK, graph_name, __file__, 0)
        source = make_node("agent", node_id, graph_name)

        input_shape = get_data_shape(args[0]) if args else get_data_shape(kwargs.get("input"))

        start_id = logger.log_event(
            "execution.start",
            source_node=source,
            payload={
                "graph_name": graph_name,
                "mode": "async",
                "input_shape": input_shape,
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
                "graph_name": graph_name,
                "mode": "async",
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
# Node function wrapper (patches the graph builder's add_node)
# ---------------------------------------------------------------------------

def _instrument_node_action(node_name: str, original_action: Any) -> Any:
    """Wrap a node's callable to emit agent.task_start / agent.task_end."""

    # Extract docstring or function name for agent_goal
    agent_goal = ""
    try:
        if hasattr(original_action, "__doc__") and original_action.__doc__:
            agent_goal = original_action.__doc__.strip().split("\n")[0][:200]
        if not agent_goal:
            agent_goal = getattr(original_action, "__name__", node_name)
    except Exception:
        agent_goal = node_name

    # Get source file info
    source_file = __file__
    source_line = 0
    try:
        source_file = inspect.getfile(original_action)
        source_line = inspect.getsourcelines(original_action)[1]
    except Exception:
        pass

    @functools.wraps(original_action)
    def instrumented_action(*a: Any, **kw: Any) -> Any:
        logger = EventLogger.get()
        nid = generate_node_id(_FRAMEWORK, f"node:{node_name}", source_file, source_line)
        source = make_node("agent", nid, node_name)
        logger.push_active_node(nid)

        # Extract input state info
        input_state = a[0] if a else kw.get("state")
        input_keys = _get_state_keys(input_state)

        start_id = logger.log_event(
            "agent.task_start",
            source_node=source,
            payload={
                "agent_name": node_name,
                "agent_goal": agent_goal,
                "task_description": ", ".join(input_keys) if input_keys else node_name,
                "tools_available": [],
                "input_state_keys": input_keys,
                "node_id": nid,
                "parent_node_id": logger.parent_node(),
            },
        )

        t0 = time.perf_counter()
        error: Exception | None = None
        result: Any = None
        try:
            result = original_action(*a, **kw)
            return result
        except Exception as exc:
            error = exc
            raise
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # Compute output state diff
            output_keys = _get_state_keys(result) if result is not None else []
            output_text = ""
            if result is not None:
                try:
                    if isinstance(result, dict):
                        # Show changed/returned keys
                        output_text = _truncate(result)
                    else:
                        output_text = _truncate(result)
                except Exception:
                    output_text = str(type(result).__name__)

            payload: dict[str, Any] = {
                "agent_name": node_name,
                "output_text": output_text,
                "output_state_keys": output_keys,
                "success": error is None,
                "duration_ms": round(latency_ms, 2),
                "node_id": nid,
            }
            if error:
                payload["error"] = str(error)[:500]
                payload["error_type"] = classify_error(error)
            if result is not None:
                payload["output_shape"] = get_data_shape(result)
                try:
                    _sig = capture_output_signature(result)
                    payload["output_hash"] = _sig["hash"]
                    payload["output_type"] = _sig["type"]
                    payload["output_size_bytes"] = _sig["size_bytes"]
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
                logger.record_error_context(node_id=nid, error_type=classify_error(error), error_msg=str(error))

    instrumented_action._stratum_original = original_action  # type: ignore[attr-defined]
    return instrumented_action


def _instrument_node_action_async(node_name: str, original_action: Any) -> Any:
    """Async variant of node instrumentation."""

    agent_goal = ""
    try:
        if hasattr(original_action, "__doc__") and original_action.__doc__:
            agent_goal = original_action.__doc__.strip().split("\n")[0][:200]
        if not agent_goal:
            agent_goal = getattr(original_action, "__name__", node_name)
    except Exception:
        agent_goal = node_name

    source_file = __file__
    source_line = 0
    try:
        source_file = inspect.getfile(original_action)
        source_line = inspect.getsourcelines(original_action)[1]
    except Exception:
        pass

    @functools.wraps(original_action)
    async def instrumented_action(*a: Any, **kw: Any) -> Any:
        logger = EventLogger.get()
        nid = generate_node_id(_FRAMEWORK, f"node:{node_name}", source_file, source_line)
        source = make_node("agent", nid, node_name)
        logger.push_active_node(nid)

        input_state = a[0] if a else kw.get("state")
        input_keys = _get_state_keys(input_state)

        start_id = logger.log_event(
            "agent.task_start",
            source_node=source,
            payload={
                "agent_name": node_name,
                "agent_goal": agent_goal,
                "task_description": ", ".join(input_keys) if input_keys else node_name,
                "tools_available": [],
                "input_state_keys": input_keys,
                "node_id": nid,
                "parent_node_id": logger.parent_node(),
            },
        )

        t0 = time.perf_counter()
        error: Exception | None = None
        result: Any = None
        try:
            result = await original_action(*a, **kw)
            return result
        except Exception as exc:
            error = exc
            raise
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            output_keys = _get_state_keys(result) if result is not None else []
            output_text = _truncate(result) if result is not None else ""

            payload: dict[str, Any] = {
                "agent_name": node_name,
                "output_text": output_text,
                "output_state_keys": output_keys,
                "success": error is None,
                "duration_ms": round(latency_ms, 2),
                "node_id": nid,
            }
            if error:
                payload["error"] = str(error)[:500]
                payload["error_type"] = classify_error(error)
            logger.log_event(
                "agent.task_end",
                source_node=source,
                payload=payload,
                parent_event_id=start_id,
            )
            logger.pop_active_node()
            if error:
                logger.record_error_context(node_id=nid, error_type=classify_error(error), error_msg=str(error))

    instrumented_action._stratum_original = original_action  # type: ignore[attr-defined]
    return instrumented_action


def _wrap_add_node(original: Any) -> Any:
    """Wrap ``StateGraph.add_node`` to instrument each node function.

    Handles both calling patterns:
      - graph.add_node("name", function)
      - graph.add_node(function)  # name inferred from function.__name__
      - graph.add_node("name", RunnableLambda(...))
    """

    @functools.wraps(original)
    def wrapper(self: Any, node: Any, action: Any = None, *args: Any, **kwargs: Any) -> Any:
        # Determine actual node_name and action callable
        if isinstance(node, str):
            node_name = node
            node_action = action
        elif callable(node):
            # add_node(function) — name inferred from function
            node_name = getattr(node, "__name__", None) or getattr(node, "name", None) or type(node).__name__
            node_action = node
            # Pass through with the original positional arrangement
            # In this case, `action` was actually None and `node` is the function
        else:
            # Unknown pattern, pass through
            return original(self, node, action, *args, **kwargs)

        if node_action is not None and callable(node_action):
            # Check if this is a Runnable (has invoke method)
            if hasattr(node_action, "invoke") and not hasattr(node_action, "__call__"):
                # Runnable without __call__ — can't easily wrap, pass through
                return original(self, node, action, *args, **kwargs)

            # Skip already-instrumented actions
            if getattr(node_action, "_stratum_original", None) is not None:
                return original(self, node, action, *args, **kwargs)

            # Choose sync or async wrapper
            if inspect.iscoroutinefunction(node_action):
                instrumented = _instrument_node_action_async(node_name, node_action)
            else:
                instrumented = _instrument_node_action(node_name, node_action)

            # Call original with instrumented action in the right position
            if isinstance(node, str):
                return original(self, node_name, instrumented, *args, **kwargs)
            else:
                return original(self, instrumented, action, *args, **kwargs)

        return original(self, node, action, *args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Conditional edge wrapper
# ---------------------------------------------------------------------------

def _wrap_add_conditional_edges(original: Any) -> Any:
    """Wrap ``StateGraph.add_conditional_edges`` to log branch decisions."""

    @functools.wraps(original)
    def wrapper(
        self: Any,
        source: str,
        path: Any,
        path_map: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if callable(path):
            original_path = path

            @functools.wraps(original_path)
            def instrumented_path(*a: Any, **kw: Any) -> Any:
                logger = EventLogger.get()
                result = original_path(*a, **kw)

                src_nid = generate_node_id(_FRAMEWORK, f"node:{source}", __file__, 0)
                src_node = make_node("agent", src_nid, source)

                target_name = str(result) if result is not None else "unknown"
                tgt_nid = generate_node_id(_FRAMEWORK, f"node:{target_name}", __file__, 0)
                tgt_node = make_node("agent", tgt_nid, target_name)

                logger.log_event(
                    "edge.traversed",
                    source_node=src_node,
                    target_node=tgt_node,
                    edge_type="conditional",
                    payload={
                        "from_node": source,
                        "branch_taken": target_name,
                        "path_map_keys": list(path_map.keys()) if isinstance(
                            path_map, dict
                        ) else None,
                        "condition_function": getattr(
                            original_path, "__name__", "anonymous"
                        ),
                    },
                )
                return result

            instrumented_path._stratum_original = original_path  # type: ignore[attr-defined]
            return original(self, source, instrumented_path, path_map, *args, **kwargs)

        return original(self, source, path, path_map, *args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Patch entry point
# ---------------------------------------------------------------------------

def patch() -> None:
    """Apply monkey-patches to langgraph internals.  Idempotent."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    # ------------------------------------------------------------------
    # 1) CompiledGraph.invoke / stream / ainvoke
    # ------------------------------------------------------------------
    CompiledGraph = None
    try:
        from langgraph.graph.graph import CompiledGraph
    except ImportError:
        try:
            from langgraph.graph import CompiledGraph
        except ImportError:
            _stderr("langgraph_patch SKIP: langgraph not installed")
            return

    _stderr("langgraph_patch activating")

    try:
        orig = CompiledGraph.invoke
        if not getattr(orig, "_stratum_patched", False):
            wrapped = _wrap_invoke(orig)
            wrapped._stratum_patched = True  # type: ignore[attr-defined]
            CompiledGraph.invoke = wrapped  # type: ignore[attr-defined]
            _stderr("langgraph_patch: CompiledGraph.invoke patched")
    except Exception as e:
        _stderr(f"langgraph_patch: CompiledGraph.invoke FAILED: {e}")

    try:
        orig = CompiledGraph.stream
        if not getattr(orig, "_stratum_patched", False):
            wrapped = _wrap_stream(orig)
            wrapped._stratum_patched = True  # type: ignore[attr-defined]
            CompiledGraph.stream = wrapped  # type: ignore[attr-defined]
            _stderr("langgraph_patch: CompiledGraph.stream patched")
    except Exception as e:
        _stderr(f"langgraph_patch: CompiledGraph.stream FAILED: {e}")

    try:
        if hasattr(CompiledGraph, "ainvoke"):
            orig = CompiledGraph.ainvoke
            if not getattr(orig, "_stratum_patched", False):
                wrapped = _wrap_ainvoke(orig)
                wrapped._stratum_patched = True  # type: ignore[attr-defined]
                CompiledGraph.ainvoke = wrapped  # type: ignore[attr-defined]
                _stderr("langgraph_patch: CompiledGraph.ainvoke patched")
    except Exception as e:
        _stderr(f"langgraph_patch: CompiledGraph.ainvoke FAILED: {e}")

    # ------------------------------------------------------------------
    # 2) StateGraph.add_node (instrument node functions)
    # ------------------------------------------------------------------
    try:
        from langgraph.graph import StateGraph

        orig = StateGraph.add_node
        if not getattr(orig, "_stratum_patched", False):
            wrapped = _wrap_add_node(orig)
            wrapped._stratum_patched = True  # type: ignore[attr-defined]
            StateGraph.add_node = wrapped  # type: ignore[attr-defined]
            _stderr("langgraph_patch: StateGraph.add_node patched")
    except Exception as e:
        _stderr(f"langgraph_patch: StateGraph.add_node FAILED: {e}")

    # Also try Graph.add_node for non-typed graphs
    try:
        from langgraph.graph import Graph

        if Graph is not StateGraph:
            orig = Graph.add_node
            if not getattr(orig, "_stratum_patched", False):
                wrapped = _wrap_add_node(orig)
                wrapped._stratum_patched = True  # type: ignore[attr-defined]
                Graph.add_node = wrapped  # type: ignore[attr-defined]
                _stderr("langgraph_patch: Graph.add_node patched")
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 3) StateGraph.add_conditional_edges (instrument branch decisions)
    # ------------------------------------------------------------------
    try:
        from langgraph.graph import StateGraph as _SG2

        orig = _SG2.add_conditional_edges
        if not getattr(orig, "_stratum_patched", False):
            wrapped = _wrap_add_conditional_edges(orig)
            wrapped._stratum_patched = True  # type: ignore[attr-defined]
            _SG2.add_conditional_edges = wrapped  # type: ignore[attr-defined]
            _stderr("langgraph_patch: StateGraph.add_conditional_edges patched")
    except Exception as e:
        _stderr(f"langgraph_patch: add_conditional_edges FAILED: {e}")

    _stderr("langgraph_patch activated")


# Auto-patch on import
patch()
