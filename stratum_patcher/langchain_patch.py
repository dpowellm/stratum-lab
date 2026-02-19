"""LangChain chain instrumentation for stratum-patcher.

Monkey-patches:
* ``langchain.chains.base.Chain.invoke``     -> ``agent.task_start`` / ``agent.task_end``
* ``langchain.chains.base.Chain.ainvoke``     -> async variant
* ``langchain_core.language_models.chat_models.BaseChatModel._generate``
                                              -> ``llm.call_start`` / ``llm.call_end``
"""

from __future__ import annotations

import functools
import hashlib
import os
import sys
import time
from typing import Any

from stratum_patcher.event_logger import (
    EventLogger,
    capture_output_signature,
    classify_error,
    generate_node_id,
    make_node,
)

_PATCHED = False
_FRAMEWORK = "langchain"


def _stderr(msg: str) -> None:
    print(f"stratum_patcher: {msg}", file=sys.stderr, flush=True)


def patch_langchain(event_logger: Any = None) -> None:
    """Monkey-patch LangChain Chain.invoke and BaseChatModel._generate."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    # ------------------------------------------------------------------
    # 1) Chain.invoke (sync)
    # ------------------------------------------------------------------
    try:
        from langchain.chains.base import Chain
    except ImportError:
        Chain = None  # type: ignore[misc,assignment]

    if Chain is not None:
        _stderr("langchain_patch activating")
        try:
            _orig_invoke = Chain.invoke

            @functools.wraps(_orig_invoke)
            def _patched_invoke(self: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
                logger = EventLogger.get()
                chain_name = self.__class__.__name__
                node_id = generate_node_id(_FRAMEWORK, chain_name, __file__, 0)
                source = make_node("agent", node_id, chain_name)
                logger.push_active_node(node_id)

                start_id = logger.log_event(
                    "agent.task_start",
                    source_node=source,
                    payload={
                        "agent_name": chain_name,
                        "agent_goal": chain_name,
                        "task_description": str(input)[:500],
                        "tools_available": [],
                        "input_preview": str(input)[:500],
                        "node_id": node_id,
                        "parent_node_id": logger.parent_node(),
                    },
                )

                t0 = time.perf_counter()
                error: Exception | None = None
                result: Any = None
                try:
                    result = _orig_invoke(self, input, config=config, **kwargs)
                    return result
                except Exception as e:
                    error = e
                    raise
                finally:
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    payload: dict[str, Any] = {
                        "status": "error" if error else "success",
                        "latency_ms": round(latency_ms, 2),
                        "error_type": classify_error(error) if error else None,
                    }
                    if error:
                        payload["error"] = str(error)[:500]
                    if result is not None:
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
                        logger.record_error_context(node_id=node_id, error_type=classify_error(error), error_msg=str(error))

            if not getattr(Chain.invoke, "_stratum_patched", False):
                _patched_invoke._stratum_patched = True  # type: ignore[attr-defined]
                Chain.invoke = _patched_invoke  # type: ignore[attr-defined]
                _stderr("langchain_patch: Chain.invoke patched")
        except Exception as e:
            _stderr(f"langchain_patch: Chain.invoke FAILED: {e}")

        # ------------------------------------------------------------------
        # 2) Chain.ainvoke (async)
        # ------------------------------------------------------------------
        try:
            _orig_ainvoke = Chain.ainvoke

            @functools.wraps(_orig_ainvoke)
            async def _patched_ainvoke(self: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
                logger = EventLogger.get()
                node_id = generate_node_id(_FRAMEWORK, self.__class__.__name__, __file__, 0)
                source = make_node("agent", node_id, self.__class__.__name__)

                start_id = logger.log_event(
                    "agent.task_start",
                    source_node=source,
                    payload={"input_preview": str(input)[:500]},
                )

                t0 = time.perf_counter()
                error: Exception | None = None
                result: Any = None
                try:
                    result = await _orig_ainvoke(self, input, config=config, **kwargs)
                    return result
                except Exception as e:
                    error = e
                    raise
                finally:
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    payload: dict[str, Any] = {
                        "status": "error" if error else "success",
                        "latency_ms": round(latency_ms, 2),
                    }
                    if error:
                        payload["error"] = str(error)[:500]
                    logger.log_event(
                        "agent.task_end",
                        source_node=source,
                        payload=payload,
                        parent_event_id=start_id,
                    )

            if not getattr(Chain.ainvoke, "_stratum_patched", False):
                _patched_ainvoke._stratum_patched = True  # type: ignore[attr-defined]
                Chain.ainvoke = _patched_ainvoke  # type: ignore[attr-defined]
                _stderr("langchain_patch: Chain.ainvoke patched")
        except AttributeError:
            pass
        except Exception as e:
            _stderr(f"langchain_patch: Chain.ainvoke FAILED: {e}")

    # ------------------------------------------------------------------
    # 3) BaseChatModel._generate (LLM call tracking)
    # ------------------------------------------------------------------
    try:
        from langchain_core.language_models.chat_models import BaseChatModel

        _orig_generate = BaseChatModel._generate

        @functools.wraps(_orig_generate)
        def _patched_generate(
            self: Any, messages: Any, stop: Any = None,
            run_manager: Any = None, **kwargs: Any,
        ) -> Any:
            logger = EventLogger.get()
            node_id = generate_node_id(_FRAMEWORK, self.__class__.__name__, __file__, 0)
            source = make_node("capability", node_id, self.__class__.__name__)

            # Build start payload with model info
            start_payload: dict[str, Any] = {
                "model_requested": getattr(self, "model_name", None) or getattr(self, "model", "unknown"),
                "message_count": len(messages) if messages else 0,
                "has_tools": bool(kwargs.get("tools") or kwargs.get("functions")),
            }

            # Capture I/O from LangChain messages
            if os.environ.get("STRATUM_CAPTURE_PROMPTS") == "1" and messages:
                try:
                    # Extract system prompt
                    sys_msgs = [m for m in messages if getattr(m, "type", "") == "system"]
                    if sys_msgs:
                        sys_text = getattr(sys_msgs[0], "content", str(sys_msgs[0]))
                        if sys_text:
                            start_payload["system_prompt_preview"] = str(sys_text)[:500]
                            start_payload["system_prompt_hash"] = hashlib.sha256(
                                str(sys_text).encode("utf-8", errors="replace")
                            ).hexdigest()
                    # Extract last user/human message
                    human_msgs = [m for m in messages if getattr(m, "type", "") in ("human", "user")]
                    if human_msgs:
                        last_user = getattr(human_msgs[-1], "content", str(human_msgs[-1]))
                        if last_user:
                            start_payload["last_user_message_preview"] = str(last_user)[:500]
                            start_payload["last_user_message_hash"] = hashlib.sha256(
                                str(last_user).encode("utf-8", errors="replace")
                            ).hexdigest()
                except Exception:
                    pass

            start_id = logger.log_event(
                "llm.call_start",
                source_node=source,
                payload=start_payload,
            )

            t0 = time.perf_counter()
            error: Exception | None = None
            result: Any = None
            try:
                result = _orig_generate(self, messages, stop=stop,
                                        run_manager=run_manager, **kwargs)
                return result
            except Exception as e:
                error = e
                raise
            finally:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                payload: dict[str, Any] = {
                    "status": "error" if error else "success",
                    "latency_ms": round(latency_ms, 2),
                }
                if error:
                    payload["error"] = str(error)[:500]
                if result is not None:
                    try:
                        # Extract text content from ChatResult
                        _text = None
                        _gens = getattr(result, "generations", None)
                        if _gens and len(_gens) > 0:
                            _msg = getattr(_gens[0], "message", None) or getattr(_gens[0], "text", None)
                            _text = getattr(_msg, "content", None) if _msg and hasattr(_msg, "content") else _msg
                        _sig = capture_output_signature(_text)
                        payload["output_hash"] = _sig["hash"]
                        payload["output_type"] = _sig["type"]
                        payload["output_size_bytes"] = _sig["size_bytes"]
                        payload["output_preview"] = _sig["preview"]
                        payload["classification_fields"] = _sig["classification_fields"]
                    except Exception:
                        pass
                logger.log_event(
                    "llm.call_end",
                    source_node=source,
                    payload=payload,
                    parent_event_id=start_id,
                )

        if not getattr(BaseChatModel._generate, "_stratum_patched", False):
            _patched_generate._stratum_patched = True  # type: ignore[attr-defined]
            BaseChatModel._generate = _patched_generate  # type: ignore[attr-defined]
            _stderr("langchain_patch: BaseChatModel._generate patched")
    except ImportError:
        pass
    except Exception as e:
        _stderr(f"langchain_patch: BaseChatModel._generate FAILED: {e}")

    if Chain is not None:
        _stderr("langchain_patch activated")
    else:
        _stderr("langchain_patch SKIP: langchain not installed")


def patch() -> None:
    """Apply LangChain patches. Alias for patch_langchain()."""
    patch_langchain()


# Auto-patch on import
patch()
