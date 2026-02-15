"""Monkey-patch for LangGraph.

Instruments the core execution path of LangGraph compiled graphs:

* ``CompiledGraph.invoke()``  -> ``execution.start`` / ``execution.end``
* ``CompiledGraph.stream()``  -> ``execution.start`` / ``execution.end``
* Node function execution     -> ``agent.task_start`` / ``agent.task_end``
* Edge traversal              -> ``edge.traversed`` (including conditional branch info)
* State channel reads/writes  -> ``data.read`` / ``data.write``
"""

from __future__ import annotations

import functools
import time
from typing import Any, Iterator

from stratum_patcher.event_logger import (
    EventLogger,
    capture_output_signature,
    get_data_shape,
    hash_content,
    make_node,
    generate_node_id,
)

_PATCHED = False
_FRAMEWORK = "langgraph"


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
    """Wrap ``CompiledGraph.stream`` to log execution start/end."""

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
                # Log each streamed node execution as a step
                try:
                    if isinstance(chunk, dict):
                        for step_name in chunk:
                            step_node_id = generate_node_id(
                                _FRAMEWORK, step_name, __file__, 0
                            )
                            step_source = make_node("agent", step_node_id, step_name)
                            logger.log_event(
                                "agent.task_end",
                                source_node=step_source,
                                payload={
                                    "node_name": step_name,
                                    "output_shape": get_data_shape(chunk[step_name]),
                                    "stream_chunk_index": chunk_count,
                                },
                                parent_event_id=start_id,
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

def _wrap_add_node(original: Any) -> Any:
    """Wrap ``StateGraph.add_node`` to instrument each node function."""

    @functools.wraps(original)
    def wrapper(self: Any, node_name: str, action: Any = None, *args: Any, **kwargs: Any) -> Any:
        if action is not None and callable(action):
            original_action = action

            @functools.wraps(original_action)
            def instrumented_action(*a: Any, **kw: Any) -> Any:
                logger = EventLogger.get()
                nid = generate_node_id(_FRAMEWORK, node_name, __file__, 0)
                source = make_node("agent", nid, node_name)

                start_id = logger.log_event(
                    "agent.task_start",
                    source_node=source,
                    payload={
                        "node_name": node_name,
                        "input_shape": get_data_shape(a[0]) if a else get_data_shape(
                            kw.get("state")
                        ),
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
                    payload: dict[str, Any] = {
                        "node_name": node_name,
                        "latency_ms": round(latency_ms, 2),
                        "status": "error" if error else "success",
                    }
                    if error:
                        payload["error"] = str(error)[:500]
                        payload["error_type"] = type(error).__name__
                    if result is not None:
                        payload["output_shape"] = get_data_shape(result)
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

            instrumented_action._stratum_original = original_action  # type: ignore[attr-defined]
            return original(self, node_name, instrumented_action, *args, **kwargs)

        return original(self, node_name, action, *args, **kwargs)

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

                src_nid = generate_node_id(_FRAMEWORK, source, __file__, 0)
                src_node = make_node("agent", src_nid, source)

                target_name = str(result) if result is not None else "unknown"
                tgt_nid = generate_node_id(_FRAMEWORK, target_name, __file__, 0)
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
# State channel read/write wrappers
# ---------------------------------------------------------------------------

def _wrap_channel_write(original: Any) -> Any:
    """Wrap channel write operations to log data.write events."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        logger = EventLogger.get()
        channel_name = getattr(self, "name", type(self).__name__)
        node_id = generate_node_id(_FRAMEWORK, f"channel:{channel_name}", __file__, 0)
        source = make_node("data_store", node_id, channel_name)

        logger.log_event(
            "data.write",
            source_node=source,
            payload={
                "channel": channel_name,
                "data_shape": get_data_shape(args[0]) if args else "unknown",
            },
        )

        return original(self, *args, **kwargs)

    return wrapper


def _wrap_channel_read(original: Any) -> Any:
    """Wrap channel read operations to log data.read events."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        logger = EventLogger.get()
        channel_name = getattr(self, "name", type(self).__name__)
        node_id = generate_node_id(_FRAMEWORK, f"channel:{channel_name}", __file__, 0)
        source = make_node("data_store", node_id, channel_name)

        result = original(self, *args, **kwargs)

        logger.log_event(
            "data.read",
            source_node=source,
            payload={
                "channel": channel_name,
                "data_shape": get_data_shape(result),
            },
        )

        return result

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
    try:
        from langgraph.graph.graph import CompiledGraph
    except ImportError:
        try:
            from langgraph.graph import CompiledGraph
        except ImportError:
            return

    try:
        orig = CompiledGraph.invoke
        if not getattr(orig, "_stratum_patched", False):
            wrapped = _wrap_invoke(orig)
            wrapped._stratum_patched = True  # type: ignore[attr-defined]
            CompiledGraph.invoke = wrapped  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        orig = CompiledGraph.stream
        if not getattr(orig, "_stratum_patched", False):
            wrapped = _wrap_stream(orig)
            wrapped._stratum_patched = True  # type: ignore[attr-defined]
            CompiledGraph.stream = wrapped  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        if hasattr(CompiledGraph, "ainvoke"):
            orig = CompiledGraph.ainvoke
            if not getattr(orig, "_stratum_patched", False):
                wrapped = _wrap_ainvoke(orig)
                wrapped._stratum_patched = True  # type: ignore[attr-defined]
                CompiledGraph.ainvoke = wrapped  # type: ignore[attr-defined]
    except Exception:
        pass

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
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 4) Channel read/write (langgraph.channels)
    # ------------------------------------------------------------------
    try:
        from langgraph.channels.base import BaseChannel

        if hasattr(BaseChannel, "update"):
            orig = BaseChannel.update
            if not getattr(orig, "_stratum_patched", False):
                wrapped = _wrap_channel_write(orig)
                wrapped._stratum_patched = True  # type: ignore[attr-defined]
                BaseChannel.update = wrapped  # type: ignore[attr-defined]

        if hasattr(BaseChannel, "get"):
            orig = BaseChannel.get
            if not getattr(orig, "_stratum_patched", False):
                wrapped = _wrap_channel_read(orig)
                wrapped._stratum_patched = True  # type: ignore[attr-defined]
                BaseChannel.get = wrapped  # type: ignore[attr-defined]
    except Exception:
        pass


# Auto-patch on import
patch()
