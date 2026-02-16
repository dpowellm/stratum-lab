"""Monkey-patch for Microsoft AutoGen (pyautogen / autogen-agentchat).

Instruments the core message-passing and reply-generation loop:

* ``ConversableAgent.receive()``          -> ``message.received``
* ``ConversableAgent.generate_reply()``   -> ``reply.generated``
* ``GroupChat.select_speaker()``          -> ``speaker.selected``
* ``ConversableAgent.execute_function()`` -> ``tool.invoked`` / ``tool.completed``
"""

from __future__ import annotations

import functools
import time
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
_FRAMEWORK = "autogen"


# ---------------------------------------------------------------------------
# ConversableAgent.receive wrapper
# ---------------------------------------------------------------------------

def _wrap_receive(original: Any) -> Any:
    """Wrap ``ConversableAgent.receive`` to log incoming messages."""

    @functools.wraps(original)
    def wrapper(self: Any, message: Any, sender: Any, *args: Any, **kwargs: Any) -> Any:
        logger = EventLogger.get()
        receiver_name = getattr(self, "name", type(self).__name__)
        sender_name = getattr(sender, "name", type(sender).__name__) if sender else "unknown"

        recv_nid = generate_node_id(_FRAMEWORK, receiver_name, __file__, 0)
        recv_node = make_node("agent", recv_nid, receiver_name)

        send_nid = generate_node_id(_FRAMEWORK, sender_name, __file__, 0)
        send_node = make_node("agent", send_nid, sender_name)

        # Describe the message without capturing its content
        msg_shape = get_data_shape(message)
        msg_hash = hash_content(message)
        _context_sig = capture_output_signature(message)

        start_id = logger.log_event(
            "message.received",
            source_node=send_node,
            target_node=recv_node,
            edge_type="sends_to",
            payload={
                "sender": sender_name,
                "receiver": receiver_name,
                "message_shape": msg_shape,
                "message_hash": msg_hash,
                "context_hash": _context_sig["hash"],
                "context_type": _context_sig["type"],
                "context_size_bytes": _context_sig["size_bytes"],
                "context_source_node": send_nid,
                "has_classification_dependency": _context_sig["classification_fields"] is not None,
                "source_node_id": send_nid,
                "target_node_id": recv_nid,
                "delegation_type": "implicit",
                "input_data_hash": msg_hash,
            },
        )
        logger.record_edge_activation(source=send_nid, target=recv_nid, data_hash=msg_hash)

        t0 = time.perf_counter()
        error: Exception | None = None
        result: Any = None
        try:
            result = original(self, message, sender, *args, **kwargs)
            return result
        except Exception as exc:
            error = exc
            raise
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if error:
                logger.log_event(
                    "message.receive_error",
                    source_node=send_node,
                    target_node=recv_node,
                    payload={
                        "sender": sender_name,
                        "receiver": receiver_name,
                        "error": str(error)[:500],
                        "error_type": type(error).__name__,
                        "latency_ms": round(latency_ms, 2),
                    },
                    parent_event_id=start_id,
                )

    return wrapper


def _wrap_a_receive(original: Any) -> Any:
    """Wrap ``ConversableAgent.a_receive`` (async) to log incoming messages."""

    @functools.wraps(original)
    async def wrapper(self: Any, message: Any, sender: Any, *args: Any, **kwargs: Any) -> Any:
        logger = EventLogger.get()
        receiver_name = getattr(self, "name", type(self).__name__)
        sender_name = getattr(sender, "name", type(sender).__name__) if sender else "unknown"

        recv_nid = generate_node_id(_FRAMEWORK, receiver_name, __file__, 0)
        recv_node = make_node("agent", recv_nid, receiver_name)

        send_nid = generate_node_id(_FRAMEWORK, sender_name, __file__, 0)
        send_node = make_node("agent", send_nid, sender_name)

        msg_shape = get_data_shape(message)
        msg_hash = hash_content(message)
        _context_sig = capture_output_signature(message)

        start_id = logger.log_event(
            "message.received",
            source_node=send_node,
            target_node=recv_node,
            edge_type="sends_to",
            payload={
                "sender": sender_name,
                "receiver": receiver_name,
                "message_shape": msg_shape,
                "message_hash": msg_hash,
                "async": True,
                "context_hash": _context_sig["hash"],
                "context_type": _context_sig["type"],
                "context_size_bytes": _context_sig["size_bytes"],
                "context_source_node": send_nid,
                "has_classification_dependency": _context_sig["classification_fields"] is not None,
            },
        )

        t0 = time.perf_counter()
        error: Exception | None = None
        result: Any = None
        try:
            result = await original(self, message, sender, *args, **kwargs)
            return result
        except Exception as exc:
            error = exc
            raise
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if error:
                logger.log_event(
                    "message.receive_error",
                    source_node=send_node,
                    target_node=recv_node,
                    payload={
                        "sender": sender_name,
                        "receiver": receiver_name,
                        "error": str(error)[:500],
                        "error_type": type(error).__name__,
                        "latency_ms": round(latency_ms, 2),
                    },
                    parent_event_id=start_id,
                )

    return wrapper


# ---------------------------------------------------------------------------
# ConversableAgent.generate_reply wrapper
# ---------------------------------------------------------------------------

def _wrap_generate_reply(original: Any) -> Any:
    """Wrap ``generate_reply`` to log when an agent produces a reply."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        logger = EventLogger.get()
        agent_name = getattr(self, "name", type(self).__name__)
        nid = generate_node_id(_FRAMEWORK, agent_name, __file__, 0)
        source = make_node("agent", nid, agent_name)
        logger.push_active_node(nid)

        # Describe messages being replied to
        messages = kwargs.get("messages") or (args[0] if args else None)
        msg_count = len(messages) if isinstance(messages, list) else 0

        sender = kwargs.get("sender") or (args[1] if len(args) > 1 else None)
        sender_name = getattr(sender, "name", "unknown") if sender else "unknown"

        start_id = logger.log_event(
            "reply.generation_start",
            source_node=source,
            payload={
                "agent_name": agent_name,
                "messages_count": msg_count,
                "sender": sender_name,
                "node_id": nid,
                "parent_node_id": logger.parent_node(),
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
                "agent_name": agent_name,
                "latency_ms": round(latency_ms, 2),
                "status": "error" if error else "success",
                "error_type": classify_error(error) if error else None,
            }
            if error:
                payload["error"] = str(error)[:500]
            if result is not None:
                payload["reply_shape"] = get_data_shape(result)
                payload["reply_hash"] = hash_content(result)
                # Detect if the reply is a function/tool call
                if isinstance(result, dict):
                    if "function_call" in result or "tool_calls" in result:
                        payload["has_tool_call"] = True
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
                "reply.generated",
                source_node=source,
                payload=payload,
                parent_event_id=start_id,
            )
            logger.pop_active_node()
            if error:
                logger.record_error_context(node_id=nid, error_type=classify_error(error), error_msg=str(error))

    return wrapper


def _wrap_a_generate_reply(original: Any) -> Any:
    """Async wrapper for ``a_generate_reply``."""

    @functools.wraps(original)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        logger = EventLogger.get()
        agent_name = getattr(self, "name", type(self).__name__)
        nid = generate_node_id(_FRAMEWORK, agent_name, __file__, 0)
        source = make_node("agent", nid, agent_name)

        messages = kwargs.get("messages") or (args[0] if args else None)
        msg_count = len(messages) if isinstance(messages, list) else 0
        sender = kwargs.get("sender") or (args[1] if len(args) > 1 else None)
        sender_name = getattr(sender, "name", "unknown") if sender else "unknown"

        start_id = logger.log_event(
            "reply.generation_start",
            source_node=source,
            payload={
                "agent_name": agent_name,
                "messages_count": msg_count,
                "sender": sender_name,
                "async": True,
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
                "agent_name": agent_name,
                "latency_ms": round(latency_ms, 2),
                "status": "error" if error else "success",
            }
            if error:
                payload["error"] = str(error)[:500]
                payload["error_type"] = type(error).__name__
            if result is not None:
                payload["reply_shape"] = get_data_shape(result)
                payload["reply_hash"] = hash_content(result)
                if isinstance(result, dict):
                    if "function_call" in result or "tool_calls" in result:
                        payload["has_tool_call"] = True
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
                "reply.generated",
                source_node=source,
                payload=payload,
                parent_event_id=start_id,
            )

    return wrapper


# ---------------------------------------------------------------------------
# GroupChat.select_speaker wrapper
# ---------------------------------------------------------------------------

def _wrap_select_speaker(original: Any) -> Any:
    """Wrap ``GroupChat.select_speaker`` to log speaker selection."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        logger = EventLogger.get()
        gc_name = getattr(self, "name", None) or "GroupChat"
        nid = generate_node_id(_FRAMEWORK, gc_name, __file__, 0)
        source = make_node("agent", nid, gc_name)

        # Available agents
        agent_names: list[str] = []
        try:
            agents = getattr(self, "agents", [])
            agent_names = [getattr(a, "name", type(a).__name__) for a in agents]
        except Exception:
            pass

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
            selected_name = "unknown"
            if result is not None:
                selected_name = getattr(result, "name", str(result))

            payload: dict[str, Any] = {
                "group_chat": gc_name,
                "available_agents": agent_names,
                "selected_speaker": selected_name,
                "latency_ms": round(latency_ms, 2),
                "status": "error" if error else "success",
            }
            if error:
                payload["error"] = str(error)[:500]
            logger.log_event(
                "speaker.selected",
                source_node=source,
                payload=payload,
            )
            _log_routing_decision(nid, selected_name, "manager_delegation", "llm_output")

    return wrapper


# ---------------------------------------------------------------------------
# ConversableAgent.execute_function wrapper
# ---------------------------------------------------------------------------

def _wrap_execute_function(original: Any) -> Any:
    """Wrap ``execute_function`` to log tool invocations."""

    @functools.wraps(original)
    def wrapper(self: Any, func_call: Any, *args: Any, **kwargs: Any) -> Any:
        logger = EventLogger.get()
        agent_name = getattr(self, "name", type(self).__name__)
        nid = generate_node_id(_FRAMEWORK, agent_name, __file__, 0)
        source = make_node("agent", nid, agent_name)

        # Extract function name
        func_name = "unknown"
        try:
            if isinstance(func_call, dict):
                func_name = func_call.get("name", "unknown")
            else:
                func_name = getattr(func_call, "name", str(func_call))
        except Exception:
            pass

        tool_nid = generate_node_id(_FRAMEWORK, func_name, __file__, 0)
        tool_node = make_node("capability", tool_nid, func_name)

        start_id = logger.log_event(
            "tool.invoked",
            source_node=source,
            target_node=tool_node,
            edge_type="calls",
            payload={
                "agent_name": agent_name,
                "tool_name": func_name,
                "args_shape": get_data_shape(func_call),
            },
        )

        t0 = time.perf_counter()
        error: Exception | None = None
        result: Any = None
        try:
            result = original(self, func_call, *args, **kwargs)
            return result
        except Exception as exc:
            error = exc
            raise
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            payload: dict[str, Any] = {
                "agent_name": agent_name,
                "tool_name": func_name,
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
                target_node=tool_node,
                edge_type="calls",
                payload=payload,
                parent_event_id=start_id,
            )

    return wrapper


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
    """Apply monkey-patches to AutoGen internals.  Idempotent."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    # Try multiple import paths (pyautogen vs autogen-agentchat vs autogen)
    ConversableAgent = None
    for mod_path in (
        "autogen.agentchat.conversable_agent",
        "autogen.agentchat",
        "autogen",
        "pyautogen.agentchat.conversable_agent",
        "pyautogen",
    ):
        try:
            mod = __import__(mod_path, fromlist=["ConversableAgent"])
            ConversableAgent = getattr(mod, "ConversableAgent", None)
            if ConversableAgent is not None:
                break
        except ImportError:
            continue

    if ConversableAgent is None:
        return

    # ------------------------------------------------------------------
    # 1) ConversableAgent.receive
    # ------------------------------------------------------------------
    try:
        orig = ConversableAgent.receive
        if not getattr(orig, "_stratum_patched", False):
            wrapped = _wrap_receive(orig)
            wrapped._stratum_patched = True  # type: ignore[attr-defined]
            ConversableAgent.receive = wrapped  # type: ignore[attr-defined]
    except Exception:
        pass

    # Async variant
    try:
        if hasattr(ConversableAgent, "a_receive"):
            orig = ConversableAgent.a_receive
            if not getattr(orig, "_stratum_patched", False):
                wrapped = _wrap_a_receive(orig)
                wrapped._stratum_patched = True  # type: ignore[attr-defined]
                ConversableAgent.a_receive = wrapped  # type: ignore[attr-defined]
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 2) ConversableAgent.generate_reply
    # ------------------------------------------------------------------
    try:
        orig = ConversableAgent.generate_reply
        if not getattr(orig, "_stratum_patched", False):
            wrapped = _wrap_generate_reply(orig)
            wrapped._stratum_patched = True  # type: ignore[attr-defined]
            ConversableAgent.generate_reply = wrapped  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        if hasattr(ConversableAgent, "a_generate_reply"):
            orig = ConversableAgent.a_generate_reply
            if not getattr(orig, "_stratum_patched", False):
                wrapped = _wrap_a_generate_reply(orig)
                wrapped._stratum_patched = True  # type: ignore[attr-defined]
                ConversableAgent.a_generate_reply = wrapped  # type: ignore[attr-defined]
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 3) GroupChat.select_speaker
    # ------------------------------------------------------------------
    GroupChat = None
    for mod_path in (
        "autogen.agentchat.groupchat",
        "autogen.agentchat",
        "autogen",
        "pyautogen.agentchat.groupchat",
        "pyautogen",
    ):
        try:
            mod = __import__(mod_path, fromlist=["GroupChat"])
            GroupChat = getattr(mod, "GroupChat", None)
            if GroupChat is not None:
                break
        except ImportError:
            continue

    if GroupChat is not None:
        try:
            orig = GroupChat.select_speaker
            if not getattr(orig, "_stratum_patched", False):
                wrapped = _wrap_select_speaker(orig)
                wrapped._stratum_patched = True  # type: ignore[attr-defined]
                GroupChat.select_speaker = wrapped  # type: ignore[attr-defined]
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 4) ConversableAgent.execute_function
    # ------------------------------------------------------------------
    try:
        if hasattr(ConversableAgent, "execute_function"):
            orig = ConversableAgent.execute_function
            if not getattr(orig, "_stratum_patched", False):
                wrapped = _wrap_execute_function(orig)
                wrapped._stratum_patched = True  # type: ignore[attr-defined]
                ConversableAgent.execute_function = wrapped  # type: ignore[attr-defined]
    except Exception:
        pass

    # Also try _execute_tool_call used in some AutoGen versions
    try:
        if hasattr(ConversableAgent, "_execute_tool_call"):
            orig = ConversableAgent._execute_tool_call
            if not getattr(orig, "_stratum_patched", False):
                wrapped = _wrap_execute_function(orig)
                wrapped._stratum_patched = True  # type: ignore[attr-defined]
                ConversableAgent._execute_tool_call = wrapped  # type: ignore[attr-defined]
    except Exception:
        pass


# Auto-patch on import
patch()
