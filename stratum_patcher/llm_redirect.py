"""Redirect all LLM calls to the stratum vLLM instance.

Patches constructors for: OpenAI, AsyncOpenAI, ChatOpenAI, crewai.LLM,
and Anthropic clients so that every LLM call goes through the local vLLM
endpoint regardless of what the repo hardcoded.

NOTE: litellm is NOT patched here — litellm_patch.py handles model
remapping via remap_model().  Patching both would cause double-prefixing
(openai/openai/model -> 404).

MUST be called BEFORE repo code imports these libraries, and BEFORE the
stratum patcher activates (which observes calls — this module redirects them).

Activation: call ``activate()`` from sitecustomize.py.
"""

from __future__ import annotations

import os
import sys


def _stderr(msg: str) -> None:
    print(f"stratum_redirect: {msg}", file=sys.stderr, flush=True)


_MAX_TOKENS_CAP = 512  # vLLM models have limited context; cap to avoid 400 errors


def _cap_max_tokens(kwargs: dict) -> dict:
    """Cap max_tokens/max_completion_tokens to avoid exceeding model context."""
    for key in ("max_tokens", "max_completion_tokens"):
        if key in kwargs and isinstance(kwargs[key], int) and kwargs[key] > _MAX_TOKENS_CAP:
            kwargs[key] = _MAX_TOKENS_CAP
    return kwargs


def activate() -> None:
    """Patch all known LLM constructors to route through vLLM."""
    vllm_url = os.environ.get("OPENAI_BASE_URL")
    vllm_model = os.environ.get("STRATUM_VLLM_MODEL")
    api_key = os.environ.get("OPENAI_API_KEY", "sk-proj-stratum000000000000000000000000000000000000000000000000")

    if not vllm_url or not vllm_model:
        return

    _stderr(f"Redirecting LLM calls -> {vllm_url} model={vllm_model}")

    # -- Patch OpenAI client -----------------------------------------------
    try:
        import openai

        _orig_openai_init = openai.OpenAI.__init__

        def _patched_openai_init(self, *args, **kwargs):
            kwargs["base_url"] = vllm_url
            kwargs["api_key"] = api_key
            _orig_openai_init(self, *args, **kwargs)

        if not getattr(openai.OpenAI.__init__, "_stratum_redirected", False):
            openai.OpenAI.__init__ = _patched_openai_init
            openai.OpenAI.__init__._stratum_redirected = True  # type: ignore[attr-defined]
            _stderr("  patched openai.OpenAI")
    except ImportError:
        pass
    except Exception as e:
        _stderr(f"  openai.OpenAI patch failed: {e}")

    # -- Patch AsyncOpenAI client ------------------------------------------
    try:
        import openai

        _orig_async_init = openai.AsyncOpenAI.__init__

        def _patched_async_openai_init(self, *args, **kwargs):
            kwargs["base_url"] = vllm_url
            kwargs["api_key"] = api_key
            _orig_async_init(self, *args, **kwargs)

        if not getattr(openai.AsyncOpenAI.__init__, "_stratum_redirected", False):
            openai.AsyncOpenAI.__init__ = _patched_async_openai_init
            openai.AsyncOpenAI.__init__._stratum_redirected = True  # type: ignore[attr-defined]
            _stderr("  patched openai.AsyncOpenAI")
    except ImportError:
        pass
    except Exception as e:
        _stderr(f"  openai.AsyncOpenAI patch failed: {e}")

    # -- Patch ChatOpenAI (langchain_openai) -------------------------------
    try:
        from langchain_openai import ChatOpenAI

        _orig_chat_init = ChatOpenAI.__init__

        def _patched_chat_init(self, *args, **kwargs):
            kwargs["model"] = vllm_model
            kwargs["base_url"] = vllm_url
            kwargs["api_key"] = api_key
            kwargs.pop("temperature", None)
            # Cap max_tokens to avoid exceeding vLLM context window
            if kwargs.get("max_tokens") and isinstance(kwargs["max_tokens"], int) and kwargs["max_tokens"] > _MAX_TOKENS_CAP:
                kwargs["max_tokens"] = _MAX_TOKENS_CAP
            _orig_chat_init(self, *args, **kwargs)

        if not getattr(ChatOpenAI.__init__, "_stratum_redirected", False):
            ChatOpenAI.__init__ = _patched_chat_init
            ChatOpenAI.__init__._stratum_redirected = True  # type: ignore[attr-defined]
            _stderr("  patched langchain_openai.ChatOpenAI")
    except ImportError:
        pass
    except Exception as e:
        _stderr(f"  ChatOpenAI patch failed: {e}")

    # -- Patch crewai.LLM -------------------------------------------------
    try:
        from crewai import LLM

        _orig_llm_init = LLM.__init__

        def _patched_llm_init(self, *args, **kwargs):
            kwargs["model"] = f"openai/{vllm_model}"
            kwargs["base_url"] = vllm_url
            kwargs["api_key"] = api_key
            _orig_llm_init(self, *args, **kwargs)

        if not getattr(LLM.__init__, "_stratum_redirected", False):
            LLM.__init__ = _patched_llm_init
            LLM.__init__._stratum_redirected = True  # type: ignore[attr-defined]
            _stderr("  patched crewai.LLM")
    except ImportError:
        pass
    except Exception as e:
        _stderr(f"  crewai.LLM patch failed: {e}")

    # NOTE: litellm.completion and litellm.acompletion are NOT patched here.
    # The litellm_patch.py in stratum_patcher already handles model remapping
    # via remap_model(). Patching here would cause double-prefixing:
    # openai/openai/mistralai/Mistral-7B -> 404 from vLLM.

    # -- Patch Anthropic client (redirect to vLLM OpenAI-compat) -----------
    try:
        import anthropic

        _orig_anthropic_init = anthropic.Anthropic.__init__

        def _patched_anthropic_init(self, *args, **kwargs):
            kwargs["base_url"] = vllm_url
            kwargs["api_key"] = api_key
            _orig_anthropic_init(self, *args, **kwargs)

        if not getattr(anthropic.Anthropic.__init__, "_stratum_redirected", False):
            anthropic.Anthropic.__init__ = _patched_anthropic_init
            anthropic.Anthropic.__init__._stratum_redirected = True  # type: ignore[attr-defined]
            _stderr("  patched anthropic.Anthropic")
    except ImportError:
        pass
    except Exception as e:
        _stderr(f"  anthropic.Anthropic patch failed: {e}")
