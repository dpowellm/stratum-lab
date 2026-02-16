"""LangChain chain instrumentation for stratum-patcher.

Monkey-patches:
* ``langchain.chains.base.Chain.invoke``     -> ``agent.task_start`` / ``agent.task_end``
* ``langchain.chains.base.Chain.ainvoke``     -> async variant
* ``langchain_core.language_models.chat_models.BaseChatModel._generate``
                                              -> ``llm.call_start`` / ``llm.call_end``
"""

from __future__ import annotations

import functools
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
        try:
            _orig_invoke = Chain.invoke

            @functools.wraps(_orig_invoke)
            def _patched_invoke(self: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
                logger = EventLogger.get()
                node_id = generate_node_id(_FRAMEWORK, self.__class__.__name__, __file__, 0)
                source = make_node("agent", node_id, self.__class__.__name__)
                logger.push_active_node(node_id)

                start_id = logger.log_event(
                    "agent.task_start",
                    source_node=source,
                    payload={
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
        except Exception:
            pass

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
        except AttributeError:
            pass
        except Exception:
            pass

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

            start_id = logger.log_event(
                "llm.call_start",
                source_node=source,
                payload={"model": getattr(self, "model_name", "unknown")},
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
    except ImportError:
        pass
    except Exception:
        pass


def patch() -> None:
    """Apply LangChain patches. Alias for patch_langchain()."""
    patch_langchain()


# Auto-patch on import
patch()
