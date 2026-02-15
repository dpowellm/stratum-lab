"""Monkey-patch for the Anthropic Python client.

Because stratum-lab routes all inference through a local vLLM instance
(OpenAI-compatible), this patcher **translates** Anthropic API calls into
OpenAI-format requests and forwards them to the vLLM endpoint.  This
lets repos that use ``anthropic.Anthropic()`` run without a real
Anthropic key.

Translation rules
-----------------
- Anthropic ``system`` parameter  -> OpenAI ``{"role": "system", ...}``
- Anthropic ``messages`` with ``user`` / ``assistant`` roles -> same roles
- Anthropic ``tool_use`` content blocks -> OpenAI ``tool_calls``
- Anthropic ``tool_result`` content blocks -> OpenAI ``{"role": "tool", ...}``
"""

from __future__ import annotations

import functools
import json
import os
import time
from typing import Any

from stratum_patcher.event_logger import (
    EventLogger,
    get_caller_info,
    get_data_shape,
    make_node,
    generate_node_id,
)

_PATCHED = False

# ---------------------------------------------------------------------------
# Anthropic -> OpenAI translation
# ---------------------------------------------------------------------------

def _translate_messages(
    system: str | list | None,
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert Anthropic-style messages to OpenAI chat format."""
    oai_messages: list[dict[str, Any]] = []

    # System prompt
    if system:
        if isinstance(system, str):
            oai_messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            # Anthropic system can be a list of content blocks
            text_parts = []
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            if text_parts:
                oai_messages.append({"role": "system", "content": "\n".join(text_parts)})

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        if isinstance(content, str):
            oai_messages.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            # May contain text, tool_use, tool_result blocks
            text_parts: list[str] = []
            tool_calls_out: list[dict[str, Any]] = []
            tool_results_out: list[dict[str, Any]] = []

            for block in content:
                if not isinstance(block, dict):
                    text_parts.append(str(block))
                    continue
                btype = block.get("type", "text")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_calls_out.append({
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(
                                block.get("input", {}), default=str
                            ),
                        },
                    })
                elif btype == "tool_result":
                    tc_content = block.get("content", "")
                    if isinstance(tc_content, list):
                        tc_content = "\n".join(
                            b.get("text", str(b)) if isinstance(b, dict) else str(b)
                            for b in tc_content
                        )
                    tool_results_out.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": str(tc_content),
                    })

            if role == "assistant" and tool_calls_out:
                oai_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": "\n".join(text_parts) if text_parts else None,
                    "tool_calls": tool_calls_out,
                }
                oai_messages.append(oai_msg)
            elif tool_results_out:
                # tool_result blocks become individual tool messages
                if text_parts:
                    oai_messages.append({"role": role, "content": "\n".join(text_parts)})
                oai_messages.extend(tool_results_out)
            else:
                oai_messages.append({"role": role, "content": "\n".join(text_parts)})

    return oai_messages


def _translate_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Convert Anthropic tool definitions to OpenAI tool format."""
    if not tools:
        return None
    oai_tools: list[dict[str, Any]] = []
    for tool in tools:
        oai_tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        })
    return oai_tools


def _translate_response_to_anthropic(oai_response: Any, model: str) -> Any:
    """Build a minimal Anthropic-compatible response object from OpenAI response.

    We return a lightweight namespace object with the fields most Anthropic
    consumers expect: ``id``, ``type``, ``role``, ``content``, ``model``,
    ``stop_reason``, ``usage``.
    """
    # Use a simple namespace to avoid importing anthropic types (which may
    # not be fully available).

    class _Ns:
        """Tiny attribute namespace."""
        def __init__(self, **kw: Any):
            self.__dict__.update(kw)
        def __repr__(self) -> str:
            return f"_Ns({self.__dict__})"

    content_blocks: list[Any] = []
    stop_reason = "end_turn"
    finish_reason = None

    try:
        choices = getattr(oai_response, "choices", None) or (
            oai_response.get("choices") if isinstance(oai_response, dict) else []
        )
        if choices:
            first = choices[0]
            message = getattr(first, "message", None) or (
                first.get("message") if isinstance(first, dict) else {}
            )
            finish_reason = getattr(first, "finish_reason", None) or (
                first.get("finish_reason") if isinstance(first, dict) else None
            )

            # Text content
            text = getattr(message, "content", None) or (
                message.get("content") if isinstance(message, dict) else None
            )
            if text:
                content_blocks.append(_Ns(type="text", text=text))

            # Tool calls -> tool_use blocks
            tc_list = getattr(message, "tool_calls", None) or (
                message.get("tool_calls") if isinstance(message, dict) else None
            )
            if tc_list:
                stop_reason = "tool_use"
                for tc in tc_list:
                    fn = getattr(tc, "function", None) or (
                        tc.get("function") if isinstance(tc, dict) else {}
                    )
                    fn_name = getattr(fn, "name", None) or (
                        fn.get("name") if isinstance(fn, dict) else ""
                    )
                    fn_args = getattr(fn, "arguments", None) or (
                        fn.get("arguments") if isinstance(fn, dict) else "{}"
                    )
                    tc_id = getattr(tc, "id", None) or (
                        tc.get("id") if isinstance(tc, dict) else ""
                    )
                    try:
                        parsed_args = json.loads(fn_args)
                    except Exception:
                        parsed_args = {}
                    content_blocks.append(_Ns(
                        type="tool_use",
                        id=tc_id,
                        name=fn_name,
                        input=parsed_args,
                    ))
    except Exception:
        pass

    if finish_reason == "stop":
        stop_reason = "end_turn"
    elif finish_reason == "length":
        stop_reason = "max_tokens"
    elif finish_reason == "tool_calls":
        stop_reason = "tool_use"

    # Usage
    usage_in = 0
    usage_out = 0
    try:
        usage = getattr(oai_response, "usage", None) or (
            oai_response.get("usage") if isinstance(oai_response, dict) else None
        )
        if usage is not None:
            usage_in = getattr(usage, "prompt_tokens", 0) or (
                usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0
            )
            usage_out = getattr(usage, "completion_tokens", 0) or (
                usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0
            )
    except Exception:
        pass

    resp_id = getattr(oai_response, "id", "msg_stratum")
    if isinstance(oai_response, dict):
        resp_id = oai_response.get("id", "msg_stratum")

    return _Ns(
        id=resp_id,
        type="message",
        role="assistant",
        content=content_blocks,
        model=model,
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=_Ns(input_tokens=usage_in, output_tokens=usage_out),
    )


# ---------------------------------------------------------------------------
# Event logging helper
# ---------------------------------------------------------------------------

def _log_call(kwargs: dict[str, Any], result: Any, latency_ms: float,
              error: Exception | None, start_id: str, source: dict) -> None:
    """Log llm.call_end for the Anthropic -> OpenAI redirect."""
    logger = EventLogger.get()
    payload: dict[str, Any] = {
        "model_requested": kwargs.get("model", "unknown"),
        "latency_ms": round(latency_ms, 2),
        "redirected_to": "vllm_openai_compat",
    }
    if error is not None:
        payload["error"] = str(error)[:500]
    else:
        try:
            payload["model_actual"] = getattr(result, "model", kwargs.get("model", "unknown"))
            usage = getattr(result, "usage", None)
            if usage:
                payload["input_tokens"] = getattr(usage, "input_tokens", None)
                payload["output_tokens"] = getattr(usage, "output_tokens", None)
            payload["stop_reason"] = getattr(result, "stop_reason", None)
            # Check for tool_use blocks
            content = getattr(result, "content", [])
            tc = [
                {"tool_name": getattr(b, "name", ""), "arguments_valid": True}
                for b in (content or [])
                if getattr(b, "type", None) == "tool_use"
            ]
            if tc:
                payload["tool_calls_made"] = tc
        except Exception:
            pass
    logger.log_event("llm.call_end", source_node=source, payload=payload,
                     parent_event_id=start_id)


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

def _make_openai_call(kwargs: dict[str, Any]) -> Any:
    """Perform the actual call via the openai library to vLLM."""
    import openai as _openai

    base_url = os.environ.get(
        "OPENAI_BASE_URL",
        os.environ.get("OPENAI_API_BASE", "http://host.docker.internal:8000/v1"),
    )
    api_key = os.environ.get("OPENAI_API_KEY", "sk-stratum-local")

    client = _openai.OpenAI(base_url=base_url, api_key=api_key)

    model = kwargs.get("model", os.environ.get("STRATUM_VLLM_MODEL", "Qwen/Qwen2.5-72B-Instruct"))
    messages = _translate_messages(kwargs.get("system"), kwargs.get("messages", []))
    tools = _translate_tools(kwargs.get("tools"))
    max_tokens = kwargs.get("max_tokens", 4096)
    temperature = kwargs.get("temperature", 1.0)

    call_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        call_kwargs["tools"] = tools
    if kwargs.get("stop_sequences"):
        call_kwargs["stop"] = kwargs["stop_sequences"]

    return client.chat.completions.create(**call_kwargs)


async def _make_openai_call_async(kwargs: dict[str, Any]) -> Any:
    """Async variant — forwards to async OpenAI client."""
    import openai as _openai

    base_url = os.environ.get(
        "OPENAI_BASE_URL",
        os.environ.get("OPENAI_API_BASE", "http://host.docker.internal:8000/v1"),
    )
    api_key = os.environ.get("OPENAI_API_KEY", "sk-stratum-local")

    client = _openai.AsyncOpenAI(base_url=base_url, api_key=api_key)

    model = kwargs.get("model", os.environ.get("STRATUM_VLLM_MODEL", "Qwen/Qwen2.5-72B-Instruct"))
    messages = _translate_messages(kwargs.get("system"), kwargs.get("messages", []))
    tools = _translate_tools(kwargs.get("tools"))
    max_tokens = kwargs.get("max_tokens", 4096)
    temperature = kwargs.get("temperature", 1.0)

    call_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        call_kwargs["tools"] = tools
    if kwargs.get("stop_sequences"):
        call_kwargs["stop"] = kwargs["stop_sequences"]

    return await client.chat.completions.create(**call_kwargs)


def _wrap_sync_messages_create(original: Any) -> Any:
    """Wrap ``anthropic.Anthropic().messages.create`` to redirect to vLLM."""

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = EventLogger.get()
        caller_file, caller_line, _ = get_caller_info(skip_frames=3)
        node_id = generate_node_id("anthropic", "Messages", caller_file, caller_line)
        source = make_node("capability", node_id, "anthropic.messages.create")

        start_id = logger.log_event(
            "llm.call_start",
            source_node=source,
            payload={
                "model_requested": kwargs.get("model", "unknown"),
                "message_count": len(kwargs.get("messages", [])),
                "has_tools": bool(kwargs.get("tools")),
                "redirected_to": "vllm_openai_compat",
            },
        )

        t0 = time.perf_counter()
        error: Exception | None = None
        result: Any = None
        try:
            oai_response = _make_openai_call(kwargs)
            result = _translate_response_to_anthropic(
                oai_response, kwargs.get("model", "unknown")
            )
            return result
        except Exception as exc:
            error = exc
            raise
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            _log_call(kwargs, result, latency_ms, error, start_id, source)

    return wrapper


def _wrap_async_messages_create(original: Any) -> Any:
    """Wrap ``anthropic.AsyncAnthropic().messages.create``."""

    @functools.wraps(original)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = EventLogger.get()
        caller_file, caller_line, _ = get_caller_info(skip_frames=3)
        node_id = generate_node_id("anthropic", "AsyncMessages", caller_file, caller_line)
        source = make_node("capability", node_id, "anthropic.async.messages.create")

        start_id = logger.log_event(
            "llm.call_start",
            source_node=source,
            payload={
                "model_requested": kwargs.get("model", "unknown"),
                "message_count": len(kwargs.get("messages", [])),
                "has_tools": bool(kwargs.get("tools")),
                "redirected_to": "vllm_openai_compat",
            },
        )

        t0 = time.perf_counter()
        error: Exception | None = None
        result: Any = None
        try:
            oai_response = await _make_openai_call_async(kwargs)
            result = _translate_response_to_anthropic(
                oai_response, kwargs.get("model", "unknown")
            )
            return result
        except Exception as exc:
            error = exc
            raise
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            _log_call(kwargs, result, latency_ms, error, start_id, source)

    return wrapper


# ---------------------------------------------------------------------------
# Patch entry point
# ---------------------------------------------------------------------------

def patch() -> None:
    """Apply monkey-patches to the anthropic module.  Idempotent."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    try:
        import anthropic  # noqa: F811
    except ImportError:
        return

    # ------------------------------------------------------------------
    # Sync client: anthropic.resources.messages.Messages.create
    # ------------------------------------------------------------------
    try:
        from anthropic.resources import messages as _msgs_mod

        _Messages = getattr(_msgs_mod, "Messages", None)
        if _Messages is not None:
            orig = _Messages.create
            if not getattr(orig, "_stratum_patched", False):
                wrapped = _wrap_sync_messages_create(orig)
                wrapped._stratum_patched = True  # type: ignore[attr-defined]
                _Messages.create = wrapped  # type: ignore[attr-defined]
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Async client: anthropic.resources.messages.AsyncMessages.create
    # ------------------------------------------------------------------
    try:
        from anthropic.resources import messages as _msgs_mod2

        _AsyncMessages = getattr(_msgs_mod2, "AsyncMessages", None)
        if _AsyncMessages is not None:
            orig = _AsyncMessages.create
            if not getattr(orig, "_stratum_patched", False):
                wrapped = _wrap_async_messages_create(orig)
                wrapped._stratum_patched = True  # type: ignore[attr-defined]
                _AsyncMessages.create = wrapped  # type: ignore[attr-defined]
    except Exception:
        pass


# Auto-patch on import
patch()
