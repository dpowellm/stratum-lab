"""Monkey-patch for the OpenAI Python client.

Wraps both the legacy ``openai.ChatCompletion.create`` API and the modern
``openai.Client.chat.completions.create`` / async variant so that every
LLM call is logged to the stratum event stream.

If ``OPENAI_BASE_URL`` is set the wrapper will redirect all calls to
that URL (typically the local vLLM endpoint).
"""

from __future__ import annotations

import functools
import hashlib
import os
import time
from typing import Any

from stratum_patcher.event_logger import (
    EventLogger,
    capture_output_signature,
    classify_error,
    get_caller_info,
    get_data_shape,
    make_node,
    generate_node_id,
)

_PATCHED = False

# ---------------------------------------------------------------------------
# vLLM model mapping — translate any OpenAI/Anthropic model name to our
# vLLM-served model so requests don't fail on unknown model names.
# ---------------------------------------------------------------------------

VLLM_MODEL = os.environ.get("STRATUM_VLLM_MODEL", "")


def remap_model(model: str) -> str:
    """Remap any model name to the vLLM-served model when STRATUM_VLLM_MODEL is set."""
    if not VLLM_MODEL:
        return model  # No remapping if env var not set (local dev, tests)
    if not model:
        return model
    _lower = model.lower()
    _KNOWN_PREFIXES = (
        "gpt-", "claude-", "o1-", "o3-", "chatgpt-",
        "mistral", "llama", "gemini", "command",
        "anthropic/", "openai/", "together_ai/",
        "deepseek", "qwen", "yi-", "mixtral",
    )
    if any(_lower.startswith(p) for p in _KNOWN_PREFIXES):
        return VLLM_MODEL
    return model


def _map_model(kwargs: dict[str, Any]) -> tuple[str, str]:
    """Map the requested model name to our vLLM model.

    Returns (original_model, mapped_model).
    """
    original = kwargs.get("model", "")
    mapped = remap_model(original)
    kwargs["model"] = mapped
    return original, mapped


def _extract_tool_calls(choices: Any) -> list[dict[str, Any]]:
    """Pull tool/function call metadata from the response choices."""
    tool_calls: list[dict[str, Any]] = []
    try:
        for choice in choices:
            message = getattr(choice, "message", None) or (
                choice.get("message") if isinstance(choice, dict) else None
            )
            if message is None:
                continue
            tc_list = getattr(message, "tool_calls", None) or (
                message.get("tool_calls") if isinstance(message, dict) else None
            )
            if not tc_list:
                continue
            for tc in tc_list:
                fn = getattr(tc, "function", None) or (
                    tc.get("function") if isinstance(tc, dict) else None
                )
                if fn is None:
                    continue
                name = getattr(fn, "name", None) or (
                    fn.get("name") if isinstance(fn, dict) else None
                )
                args_raw = getattr(fn, "arguments", None) or (
                    fn.get("arguments") if isinstance(fn, dict) else None
                )
                args_valid = True
                if isinstance(args_raw, str):
                    try:
                        import json
                        json.loads(args_raw)
                    except Exception:
                        args_valid = False
                tool_calls.append({
                    "tool_name": name,
                    "arguments_valid": args_valid,
                })
    except Exception:
        pass
    return tool_calls


def _build_payload(
    kwargs: dict[str, Any],
    result: Any,
    latency_ms: float,
    error: Exception | None = None,
) -> dict[str, Any]:
    """Build the event payload dict from request kwargs and response."""
    payload: dict[str, Any] = {
        "model_requested": kwargs.get("model", "unknown"),
        "latency_ms": round(latency_ms, 2),
    }
    if error is not None:
        payload["error"] = str(error)[:500]
        return payload

    # Extract from response object (may be pydantic model or dict)
    try:
        model_actual = getattr(result, "model", None) or (
            result.get("model") if isinstance(result, dict) else None
        )
        payload["model_actual"] = model_actual or kwargs.get("model", "unknown")
    except Exception:
        payload["model_actual"] = kwargs.get("model", "unknown")

    try:
        usage = getattr(result, "usage", None) or (
            result.get("usage") if isinstance(result, dict) else None
        )
        if usage is not None:
            payload["input_tokens"] = getattr(usage, "prompt_tokens", None) or (
                usage.get("prompt_tokens") if isinstance(usage, dict) else None
            )
            payload["output_tokens"] = getattr(usage, "completion_tokens", None) or (
                usage.get("completion_tokens") if isinstance(usage, dict) else None
            )
    except Exception:
        pass

    # Finish reason & tool calls
    try:
        choices = getattr(result, "choices", None) or (
            result.get("choices") if isinstance(result, dict) else None
        )
        if choices and len(choices) > 0:
            first = choices[0]
            fr = getattr(first, "finish_reason", None) or (
                first.get("finish_reason") if isinstance(first, dict) else None
            )
            payload["finish_reason"] = fr
            tc = _extract_tool_calls(choices)
            if tc:
                payload["tool_calls_made"] = tc
    except Exception:
        pass

    return payload


# ---------------------------------------------------------------------------
# Wrapper factories
# ---------------------------------------------------------------------------

def _wrap_sync_create(original: Any) -> Any:
    """Return a sync wrapper around ``completions.create``."""

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = EventLogger.get()
        caller_file, caller_line, _ = get_caller_info(skip_frames=3)
        node_id = generate_node_id("openai", "ChatCompletion", caller_file, caller_line)
        source = make_node("capability", node_id, "openai.chat.completions.create")

        # Map model name to vLLM model
        original_model, mapped_model = _map_model(kwargs)

        start_payload = {
                "model_requested": original_model,
                "model_actual": mapped_model,
                "model_mapped": original_model != mapped_model,
                "message_count": len(kwargs.get("messages", [])),
                "has_tools": bool(kwargs.get("tools") or kwargs.get("functions")),
            }

        # Prompt content capture (gated by env var, default off)
        if os.environ.get("STRATUM_CAPTURE_PROMPTS") == "1":
            messages = kwargs.get("messages", [])
            try:
                sys_msgs = [m for m in messages if m.get("role") == "system"]
                if sys_msgs:
                    sys_text = str(sys_msgs[0].get("content", ""))[:200]
                    start_payload["system_prompt_preview"] = sys_text
                    start_payload["system_prompt_hash"] = hashlib.sha256(
                        sys_text.encode()
                    ).hexdigest()[:16]
                    trust_signals = []
                    for pattern in ["verified", "confirmed", "factual", "accurate", "trusted", "reliable"]:
                        if pattern in sys_text.lower():
                            trust_signals.append(pattern)
                    if trust_signals:
                        start_payload["prompt_trust_signals"] = trust_signals

                user_msgs = [m for m in messages if m.get("role") == "user"]
                if user_msgs:
                    last_user = str(user_msgs[-1].get("content", ""))[:200]
                    start_payload["last_user_message_preview"] = last_user
                    start_payload["last_user_message_hash"] = hashlib.sha256(
                        last_user.encode()
                    ).hexdigest()[:16]
            except Exception:
                pass

        start_id = logger.log_event(
            "llm.call_start",
            source_node=source,
            payload=start_payload,
        )
        logger.push_active_node(node_id)

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
            payload = _build_payload(kwargs, result, latency_ms, error)
            payload["error_type"] = classify_error(error) if error else None
            payload["active_node_stack"] = list(logger._active_node_stack)
            # Semantic content capture on llm.call_end
            if error is None and result is not None:
                try:
                    _content = _extract_response_content(result)
                    _sig = capture_output_signature(_content)
                    payload["output_hash"] = _sig["hash"]
                    payload["output_type"] = _sig["type"]
                    payload["output_size_bytes"] = _sig["size_bytes"]
                    payload["output_preview"] = _sig["preview"]
                    payload["output_structure"] = _sig["structure"]
                    payload["classification_fields"] = _sig["classification_fields"]
                except Exception:
                    pass
            logger.log_event(
                "llm.call_end",
                source_node=source,
                payload=payload,
                parent_event_id=start_id,
            )
            logger.pop_active_node()
            if error:
                logger.record_error_context(node_id=node_id, error_type=classify_error(error), error_msg=str(error))
            # Log tool call failures from vLLM
            if error is None and result is not None:
                try:
                    tc = payload.get("tool_calls_made", [])
                    for t in tc:
                        if not t.get("arguments_valid", True):
                            logger.log_event(
                                "tool.call_failure",
                                source_node=source,
                                payload={
                                    "tool_name": t.get("tool_name"),
                                    "reason": "invalid_arguments_json",
                                },
                                parent_event_id=start_id,
                            )
                except Exception:
                    pass

    return wrapper


def _extract_response_content(result: Any) -> Any:
    """Extract the text content from an OpenAI response object."""
    try:
        choices = getattr(result, "choices", None) or (
            result.get("choices") if isinstance(result, dict) else None
        )
        if choices and len(choices) > 0:
            first = choices[0]
            message = getattr(first, "message", None) or (
                first.get("message") if isinstance(first, dict) else None
            )
            if message is not None:
                content = getattr(message, "content", None) or (
                    message.get("content") if isinstance(message, dict) else None
                )
                if content is not None:
                    # Try to parse as JSON for richer classification
                    try:
                        import json as _json
                        return _json.loads(content)
                    except Exception:
                        pass
                    return content
    except Exception:
        pass
    return None


def _wrap_async_create(original: Any) -> Any:
    """Return an async wrapper around the async ``completions.create``."""

    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = EventLogger.get()
        caller_file, caller_line, _ = get_caller_info(skip_frames=3)
        node_id = generate_node_id("openai", "AsyncChatCompletion", caller_file, caller_line)
        source = make_node("capability", node_id, "openai.async.chat.completions.create")

        # Map model name to vLLM model
        original_model, mapped_model = _map_model(kwargs)

        start_payload = {
                "model_requested": original_model,
                "model_actual": mapped_model,
                "model_mapped": original_model != mapped_model,
                "message_count": len(kwargs.get("messages", [])),
                "has_tools": bool(kwargs.get("tools") or kwargs.get("functions")),
            }

        # Prompt content capture (gated by env var, default off)
        if os.environ.get("STRATUM_CAPTURE_PROMPTS") == "1":
            messages = kwargs.get("messages", [])
            try:
                sys_msgs = [m for m in messages if m.get("role") == "system"]
                if sys_msgs:
                    sys_text = str(sys_msgs[0].get("content", ""))[:200]
                    start_payload["system_prompt_preview"] = sys_text
                    start_payload["system_prompt_hash"] = hashlib.sha256(
                        sys_text.encode()
                    ).hexdigest()[:16]
                    trust_signals = []
                    for pattern in ["verified", "confirmed", "factual", "accurate", "trusted", "reliable"]:
                        if pattern in sys_text.lower():
                            trust_signals.append(pattern)
                    if trust_signals:
                        start_payload["prompt_trust_signals"] = trust_signals

                user_msgs = [m for m in messages if m.get("role") == "user"]
                if user_msgs:
                    last_user = str(user_msgs[-1].get("content", ""))[:200]
                    start_payload["last_user_message_preview"] = last_user
                    start_payload["last_user_message_hash"] = hashlib.sha256(
                        last_user.encode()
                    ).hexdigest()[:16]
            except Exception:
                pass

        start_id = logger.log_event(
            "llm.call_start",
            source_node=source,
            payload=start_payload,
        )
        logger.push_active_node(node_id)

        t0 = time.perf_counter()
        error: Exception | None = None
        result: Any = None
        try:
            result = await original(*args, **kwargs)
            return result
        except Exception as exc:
            error = exc
            raise
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            payload = _build_payload(kwargs, result, latency_ms, error)
            payload["error_type"] = classify_error(error) if error else None
            payload["active_node_stack"] = list(logger._active_node_stack)
            # Semantic content capture on async llm.call_end
            if error is None and result is not None:
                try:
                    _content = _extract_response_content(result)
                    _sig = capture_output_signature(_content)
                    payload["output_hash"] = _sig["hash"]
                    payload["output_type"] = _sig["type"]
                    payload["output_size_bytes"] = _sig["size_bytes"]
                    payload["output_preview"] = _sig["preview"]
                    payload["output_structure"] = _sig["structure"]
                    payload["classification_fields"] = _sig["classification_fields"]
                except Exception:
                    pass
            logger.log_event(
                "llm.call_end",
                source_node=source,
                payload=payload,
                parent_event_id=start_id,
            )
            logger.pop_active_node()
            if error:
                logger.record_error_context(node_id=node_id, error_type=classify_error(error), error_msg=str(error))
            if error is None and result is not None:
                try:
                    tc = payload.get("tool_calls_made", [])
                    for t in tc:
                        if not t.get("arguments_valid", True):
                            logger.log_event(
                                "tool.call_failure",
                                source_node=source,
                                payload={
                                    "tool_name": t.get("tool_name"),
                                    "reason": "invalid_arguments_json",
                                },
                                parent_event_id=start_id,
                            )
                except Exception:
                    pass

    return wrapper


# ---------------------------------------------------------------------------
# Patch entry point
# ---------------------------------------------------------------------------

def patch() -> None:
    """Apply monkey-patches to the openai module.  Idempotent."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    try:
        import openai  # noqa: F811
    except ImportError:
        return

    # ------------------------------------------------------------------
    # 1) Legacy API  (openai < 1.0)
    # ------------------------------------------------------------------
    try:
        chat_completion_cls = getattr(openai, "ChatCompletion", None)
        if chat_completion_cls is not None:
            orig = getattr(chat_completion_cls, "create", None)
            if orig is not None and not getattr(orig, "_stratum_patched", False):
                wrapped = _wrap_sync_create(orig)
                wrapped._stratum_patched = True  # type: ignore[attr-defined]
                chat_completion_cls.create = wrapped  # type: ignore[attr-defined]
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 2) New sync client  (openai >= 1.0)
    # ------------------------------------------------------------------
    try:
        from openai.resources.chat import completions as _comp_mod

        _Completions = getattr(_comp_mod, "Completions", None)
        if _Completions is not None:
            orig = _Completions.create
            if not getattr(orig, "_stratum_patched", False):
                wrapped = _wrap_sync_create(orig)
                wrapped._stratum_patched = True  # type: ignore[attr-defined]
                _Completions.create = wrapped  # type: ignore[attr-defined]
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 3) New async client  (openai >= 1.0)
    # ------------------------------------------------------------------
    try:
        from openai.resources.chat import completions as _comp_mod2

        _AsyncCompletions = getattr(_comp_mod2, "AsyncCompletions", None)
        if _AsyncCompletions is not None:
            orig = _AsyncCompletions.create
            if not getattr(orig, "_stratum_patched", False):
                wrapped = _wrap_async_create(orig)
                wrapped._stratum_patched = True  # type: ignore[attr-defined]
                _AsyncCompletions.create = wrapped  # type: ignore[attr-defined]
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 4) AsyncOpenAI client class (ensure async completions are patched)
    # ------------------------------------------------------------------
    try:
        _AsyncOpenAI = getattr(openai, "AsyncOpenAI", None)
        if _AsyncOpenAI is not None:
            # Patch at the class level to ensure any instantiation gets the wrapper.
            # The AsyncCompletions class patch above covers the method, but we also
            # mark the client class so tests can verify the patch was applied.
            if not getattr(_AsyncOpenAI, "_stratum_patched", False):
                _AsyncOpenAI._stratum_patched = True  # type: ignore[attr-defined]
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 5) Redirect base_url if env is set
    # ------------------------------------------------------------------
    try:
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            # Set on the module-level default for older API
            if hasattr(openai, "api_base"):
                openai.api_base = base_url  # type: ignore[attr-defined]
            # Also set OPENAI_API_BASE for libraries that read it directly
            os.environ.setdefault("OPENAI_API_BASE", base_url)
    except Exception:
        pass


# Auto-patch on import
patch()
