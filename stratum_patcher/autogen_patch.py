"""Monkey-patch for Microsoft AutoGen (pyautogen / autogen-agentchat).

Instruments the core message-passing and reply-generation loop using
standard event types that the export pipeline recognizes:

* ``ConversableAgent.initiate_chat()`` -> ``execution.start`` / ``execution.end``
* ``ConversableAgent.receive()``       -> ``agent.task_start``
* ``ConversableAgent.generate_reply()`` -> ``agent.task_end``
* ``GroupChat.select_speaker()``       -> delegation event (logged in payload)
* ``ConversableAgent.execute_function()`` -> ``tool.invoked`` / ``tool.completed``
"""

from __future__ import annotations

import functools
import sys
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


def _stderr(msg: str) -> None:
    print(f"stratum_patcher: {msg}", file=sys.stderr, flush=True)


def _truncate(text: Any, limit: int = 2000) -> str:
    s = str(text)
    return s[:limit] if len(s) > limit else s


# ---------------------------------------------------------------------------
# ConversableAgent.initiate_chat wrapper -> execution.start / execution.end
# ---------------------------------------------------------------------------

def _wrap_initiate_chat(original: Any) -> Any:
    """Wrap ``initiate_chat`` to log execution start/end."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        logger = EventLogger.get()
        agent_name = getattr(self, "name", type(self).__name__)
        nid = generate_node_id(_FRAMEWORK, agent_name, __file__, 0)
        source = make_node("agent", nid, agent_name)

        # Determine recipient
        recipient = args[0] if args else kwargs.get("recipient")
        recipient_name = getattr(recipient, "name", "unknown") if recipient else "unknown"
        message = kwargs.get("message", args[1] if len(args) > 1 else "")

        start_id = logger.log_event(
            "execution.start",
            source_node=source,
            payload={
                "initiator": agent_name,
                "recipient": recipient_name,
                "message_shape": get_data_shape(message),
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
                "initiator": agent_name,
                "recipient": recipient_name,
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


def _wrap_a_initiate_chat(original: Any) -> Any:
    """Async wrapper for ``a_initiate_chat``."""

    @functools.wraps(original)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        logger = EventLogger.get()
        agent_name = getattr(self, "name", type(self).__name__)
        nid = generate_node_id(_FRAMEWORK, agent_name, __file__, 0)
        source = make_node("agent", nid, agent_name)

        recipient = args[0] if args else kwargs.get("recipient")
        recipient_name = getattr(recipient, "name", "unknown") if recipient else "unknown"

        start_id = logger.log_event(
            "execution.start",
            source_node=source,
            payload={
                "initiator": agent_name,
                "recipient": recipient_name,
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
                "initiator": agent_name,
                "latency_ms": round(latency_ms, 2),
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
# ConversableAgent.receive wrapper -> agent.task_start
# ---------------------------------------------------------------------------

def _wrap_receive(original: Any) -> Any:
    """Wrap ``ConversableAgent.receive`` to emit agent.task_start."""

    @functools.wraps(original)
    def wrapper(self: Any, message: Any, sender: Any, *args: Any, **kwargs: Any) -> Any:
        logger = EventLogger.get()
        receiver_name = getattr(self, "name", type(self).__name__)
        sender_name = getattr(sender, "name", type(sender).__name__) if sender else "unknown"
        system_message = getattr(self, "system_message", "") or ""

        recv_nid = generate_node_id(_FRAMEWORK, receiver_name, __file__, 0)
        recv_node = make_node("agent", recv_nid, receiver_name)
        logger.push_active_node(recv_nid)

        # Extract message content for task_description
        msg_content = ""
        if isinstance(message, str):
            msg_content = message
        elif isinstance(message, dict):
            msg_content = str(message.get("content", ""))
        msg_shape = get_data_shape(message)

        start_id = logger.log_event(
            "agent.task_start",
            source_node=recv_node,
            payload={
                "agent_name": receiver_name,
                "agent_goal": _truncate(system_message, 500),
                "task_description": _truncate(msg_content, 500),
                "tools_available": [],
                "sender": sender_name,
                "message_shape": msg_shape,
                "node_id": recv_nid,
                "parent_node_id": logger.parent_node(),
            },
        )

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
            payload: dict[str, Any] = {
                "agent_name": receiver_name,
                "output_text": "",
                "success": error is None,
                "duration_ms": round(latency_ms, 2),
            }
            if error:
                payload["error"] = str(error)[:500]
                payload["error_type"] = classify_error(error)
            logger.log_event(
                "agent.task_end",
                source_node=recv_node,
                payload=payload,
                parent_event_id=start_id,
            )
            logger.pop_active_node()
            if error:
                logger.record_error_context(node_id=recv_nid, error_type=classify_error(error), error_msg=str(error))

    return wrapper


def _wrap_a_receive(original: Any) -> Any:
    """Wrap ``ConversableAgent.a_receive`` (async) to emit agent.task_start/end."""

    @functools.wraps(original)
    async def wrapper(self: Any, message: Any, sender: Any, *args: Any, **kwargs: Any) -> Any:
        logger = EventLogger.get()
        receiver_name = getattr(self, "name", type(self).__name__)
        sender_name = getattr(sender, "name", type(sender).__name__) if sender else "unknown"
        system_message = getattr(self, "system_message", "") or ""

        recv_nid = generate_node_id(_FRAMEWORK, receiver_name, __file__, 0)
        recv_node = make_node("agent", recv_nid, receiver_name)
        logger.push_active_node(recv_nid)

        msg_content = ""
        if isinstance(message, str):
            msg_content = message
        elif isinstance(message, dict):
            msg_content = str(message.get("content", ""))

        start_id = logger.log_event(
            "agent.task_start",
            source_node=recv_node,
            payload={
                "agent_name": receiver_name,
                "agent_goal": _truncate(system_message, 500),
                "task_description": _truncate(msg_content, 500),
                "tools_available": [],
                "sender": sender_name,
                "async": True,
                "node_id": recv_nid,
                "parent_node_id": logger.parent_node(),
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
            payload: dict[str, Any] = {
                "agent_name": receiver_name,
                "output_text": "",
                "success": error is None,
                "duration_ms": round(latency_ms, 2),
            }
            if error:
                payload["error"] = str(error)[:500]
                payload["error_type"] = classify_error(error)
            logger.log_event(
                "agent.task_end",
                source_node=recv_node,
                payload=payload,
                parent_event_id=start_id,
            )
            logger.pop_active_node()
            if error:
                logger.record_error_context(node_id=recv_nid, error_type=classify_error(error), error_msg=str(error))

    return wrapper


# ---------------------------------------------------------------------------
# GroupChat.select_speaker wrapper
# ---------------------------------------------------------------------------

def _wrap_select_speaker(original: Any) -> Any:
    """Wrap ``GroupChat.select_speaker`` to log speaker selection as delegation."""

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

            # Log as agent.task_start/end for the GroupChat orchestrator
            start_id = logger.log_event(
                "agent.task_start",
                source_node=source,
                payload={
                    "agent_name": gc_name,
                    "agent_goal": "select next speaker",
                    "task_description": f"selecting from {agent_names}",
                    "tools_available": [],
                    "available_agents": agent_names,
                },
            )
            logger.log_event(
                "agent.task_end",
                source_node=source,
                payload={
                    "agent_name": gc_name,
                    "output_text": f"selected: {selected_name}",
                    "success": error is None,
                    "duration_ms": round(latency_ms, 2),
                    "selected_speaker": selected_name,
                },
                parent_event_id=start_id,
            )

    return wrapper


# ---------------------------------------------------------------------------
# ConversableAgent.execute_function wrapper -> tool.invoked / tool.completed
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
        _stderr("autogen_patch SKIP: autogen not installed")
        return

    _stderr("autogen_patch activating")

    # ------------------------------------------------------------------
    # 1) ConversableAgent.initiate_chat -> execution.start / execution.end
    # ------------------------------------------------------------------
    try:
        if hasattr(ConversableAgent, "initiate_chat"):
            orig = ConversableAgent.initiate_chat
            if not getattr(orig, "_stratum_patched", False):
                wrapped = _wrap_initiate_chat(orig)
                wrapped._stratum_patched = True  # type: ignore[attr-defined]
                ConversableAgent.initiate_chat = wrapped  # type: ignore[attr-defined]
                _stderr("autogen_patch: initiate_chat patched")
    except Exception as e:
        _stderr(f"autogen_patch: initiate_chat FAILED: {e}")

    try:
        if hasattr(ConversableAgent, "a_initiate_chat"):
            orig = ConversableAgent.a_initiate_chat
            if not getattr(orig, "_stratum_patched", False):
                wrapped = _wrap_a_initiate_chat(orig)
                wrapped._stratum_patched = True  # type: ignore[attr-defined]
                ConversableAgent.a_initiate_chat = wrapped  # type: ignore[attr-defined]
                _stderr("autogen_patch: a_initiate_chat patched")
    except Exception as e:
        _stderr(f"autogen_patch: a_initiate_chat FAILED: {e}")

    # ------------------------------------------------------------------
    # 2) ConversableAgent.receive -> agent.task_start / agent.task_end
    # ------------------------------------------------------------------
    try:
        orig = ConversableAgent.receive
        if not getattr(orig, "_stratum_patched", False):
            wrapped = _wrap_receive(orig)
            wrapped._stratum_patched = True  # type: ignore[attr-defined]
            ConversableAgent.receive = wrapped  # type: ignore[attr-defined]
            _stderr("autogen_patch: receive patched")
    except Exception as e:
        _stderr(f"autogen_patch: receive FAILED: {e}")

    # Async variant
    try:
        if hasattr(ConversableAgent, "a_receive"):
            orig = ConversableAgent.a_receive
            if not getattr(orig, "_stratum_patched", False):
                wrapped = _wrap_a_receive(orig)
                wrapped._stratum_patched = True  # type: ignore[attr-defined]
                ConversableAgent.a_receive = wrapped  # type: ignore[attr-defined]
                _stderr("autogen_patch: a_receive patched")
    except Exception as e:
        _stderr(f"autogen_patch: a_receive FAILED: {e}")

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
                _stderr("autogen_patch: select_speaker patched")
        except Exception as e:
            _stderr(f"autogen_patch: select_speaker FAILED: {e}")

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
                _stderr("autogen_patch: execute_function patched")
    except Exception as e:
        _stderr(f"autogen_patch: execute_function FAILED: {e}")

    # Also try _execute_tool_call used in some AutoGen versions
    try:
        if hasattr(ConversableAgent, "_execute_tool_call"):
            orig = ConversableAgent._execute_tool_call
            if not getattr(orig, "_stratum_patched", False):
                wrapped = _wrap_execute_function(orig)
                wrapped._stratum_patched = True  # type: ignore[attr-defined]
                ConversableAgent._execute_tool_call = wrapped  # type: ignore[attr-defined]
                _stderr("autogen_patch: _execute_tool_call patched")
    except Exception as e:
        _stderr(f"autogen_patch: _execute_tool_call FAILED: {e}")

    _stderr("autogen_patch activated")


# Auto-patch on import
patch()
