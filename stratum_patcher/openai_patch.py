"""Monkey-patch for the OpenAI Python client.

Wraps both the legacy ``openai.ChatCompletion.create`` API and the modern
``openai.Client.chat.completions.create`` / async variant so that every
LLM call is logged to the stratum event stream.

If ``OPENAI_BASE_URL`` is set the wrapper will redirect all calls to
that URL (typically the local vLLM endpoint).
"""

from __future__ import annotations

import functools
import os
import time
from typing import Any

from stratum_patcher.event_logger import (
    EventLogger,
    capture_output_signature,
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

VLLM_MODEL = os.environ.get("STRATUM_VLLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")

MODEL_MAP: dict[str, str] = {
    "gpt-4": VLLM_MODEL,
    "gpt-4o": VLLM_MODEL,
    "gpt-4o-mini": VLLM_MODEL,
    "gpt-4-turbo": VLLM_MODEL,
    "gpt-4-turbo-preview": VLLM_MODEL,
    "gpt-3.5-turbo": VLLM_MODEL,
    "gpt-3.5-turbo-16k": VLLM_MODEL,
    "gpt-4-1106-preview": VLLM_MODEL,
    "gpt-4-0125-preview": VLLM_MODEL,
    # Claude model names (if anthropic_patch routes through)
    "claude-3-opus-20240229": VLLM_MODEL,
    "claude-3-sonnet-20240229": VLLM_MODEL,
    "claude-3-haiku-20240307": VLLM_MODEL,
    "claude-3-5-sonnet-20241022": VLLM_MODEL,
}


def _map_model(kwargs: dict[str, Any]) -> tuple[str, str]:
    """Map the requested model name to our vLLM model.

    Returns (original_model, mapped_model).
    """
    original = kwargs.get("model", "")
    mapped = MODEL_MAP.get(original, VLLM_MODEL)
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

        start_id = logger.log_event(
            "llm.call_start",
            source_node=source,
            payload={
                "model_requested": original_model,
                "model_actual": mapped_model,
                "model_mapped": original_model != mapped_model,
                "message_count": len(kwargs.get("messages", [])),
                "has_tools": bool(kwargs.get("tools") or kwargs.get("functions")),
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
            payload = _build_payload(kwargs, result, latency_ms, error)
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

        start_id = logger.log_event(
            "llm.call_start",
            source_node=source,
            payload={
                "model_requested": original_model,
                "model_actual": mapped_model,
                "model_mapped": original_model != mapped_model,
                "message_count": len(kwargs.get("messages", [])),
                "has_tools": bool(kwargs.get("tools") or kwargs.get("functions")),
            },
        )

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
