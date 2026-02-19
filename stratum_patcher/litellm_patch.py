"""Monkey-patch for LiteLLM (used by CrewAI under the hood).

Wraps ``litellm.completion`` and ``litellm.acompletion`` so that:
1. Model names are rewritten to the vLLM-served model
2. LLM calls are logged to the stratum event stream
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
    make_node,
    generate_node_id,
)
from stratum_patcher.openai_patch import remap_model

_PATCHED = False


def _build_litellm_payload(
    kwargs: dict[str, Any],
    result: Any,
    latency_ms: float,
    error: Exception | None = None,
) -> dict[str, Any]:
    """Build event payload from litellm call kwargs and response."""
    payload: dict[str, Any] = {
        "model_requested": kwargs.get("model", "unknown"),
        "latency_ms": round(latency_ms, 2),
    }
    if error is not None:
        payload["error"] = str(error)[:500]
        return payload

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

    return payload


def _wrap_litellm_sync(original: Any) -> Any:
    """Wrap litellm.completion (sync)."""

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = EventLogger.get()
        caller_file, caller_line, _ = get_caller_info(skip_frames=3)
        node_id = generate_node_id("litellm", "Completion", caller_file, caller_line)
        source = make_node("capability", node_id, "litellm.completion")

        original_model = kwargs.get("model", "")
        mapped_model = remap_model(original_model)
        kwargs["model"] = f"openai/{mapped_model}" if os.environ.get("STRATUM_VLLM_MODEL") and not mapped_model.startswith("openai/") else mapped_model

        start_payload = {
            "model_requested": original_model,
            "model_actual": mapped_model,
            "model_mapped": original_model != mapped_model,
            "message_count": len(kwargs.get("messages", [])),
        }

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

        # Cap max_tokens to avoid exceeding vLLM model context window
        for _mt_key in ("max_tokens", "max_completion_tokens"):
            if _mt_key in kwargs and isinstance(kwargs[_mt_key], int) and kwargs[_mt_key] > 512:
                kwargs[_mt_key] = 512

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
            payload = _build_litellm_payload(kwargs, result, latency_ms, error)
            payload["error_type"] = classify_error(error) if error else None
            if error is None and result is not None:
                try:
                    _content = _extract_litellm_content(result)
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

    return wrapper


def _wrap_litellm_async(original: Any) -> Any:
    """Wrap litellm.acompletion (async)."""

    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = EventLogger.get()
        caller_file, caller_line, _ = get_caller_info(skip_frames=3)
        node_id = generate_node_id("litellm", "AsyncCompletion", caller_file, caller_line)
        source = make_node("capability", node_id, "litellm.acompletion")

        original_model = kwargs.get("model", "")
        mapped_model = remap_model(original_model)
        kwargs["model"] = f"openai/{mapped_model}" if os.environ.get("STRATUM_VLLM_MODEL") and not mapped_model.startswith("openai/") else mapped_model

        start_payload = {
            "model_requested": original_model,
            "model_actual": mapped_model,
            "model_mapped": original_model != mapped_model,
            "message_count": len(kwargs.get("messages", [])),
        }

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

        # Cap max_tokens to avoid exceeding vLLM model context window
        for _mt_key in ("max_tokens", "max_completion_tokens"):
            if _mt_key in kwargs and isinstance(kwargs[_mt_key], int) and kwargs[_mt_key] > 512:
                kwargs[_mt_key] = 512

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
            payload = _build_litellm_payload(kwargs, result, latency_ms, error)
            payload["error_type"] = classify_error(error) if error else None
            if error is None and result is not None:
                try:
                    _content = _extract_litellm_content(result)
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

    return wrapper


def _extract_litellm_content(result: Any) -> Any:
    """Extract text content from a litellm response."""
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
                return getattr(message, "content", None) or (
                    message.get("content") if isinstance(message, dict) else None
                )
    except Exception:
        pass
    return None


def patch() -> None:
    """Apply monkey-patches to litellm. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    try:
        import litellm
    except ImportError:
        return

    # Patch sync completion
    try:
        orig = litellm.completion
        if not getattr(orig, "_stratum_patched", False):
            wrapped = _wrap_litellm_sync(orig)
            wrapped._stratum_patched = True  # type: ignore[attr-defined]
            litellm.completion = wrapped  # type: ignore[attr-defined]
    except Exception:
        pass

    # Patch async completion
    try:
        orig = litellm.acompletion
        if not getattr(orig, "_stratum_patched", False):
            wrapped = _wrap_litellm_async(orig)
            wrapped._stratum_patched = True  # type: ignore[attr-defined]
            litellm.acompletion = wrapped  # type: ignore[attr-defined]
    except Exception:
        pass


# Auto-patch on import
patch()
