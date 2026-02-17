"""Patcher smoke tests — verify each patcher module loads without crashing.

These tests import each patcher module and verify basic structure.
They do NOT require the target frameworks to be installed; they test
that the patcher code itself is well-formed.
"""
from __future__ import annotations

import importlib
import os
import sys

import pytest

PARENT_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PARENT_DIR)


# ===========================================================================
# MODULE IMPORT TESTS
# ===========================================================================

class TestPatcherImports:
    def test_event_logger_import(self):
        """event_logger module imports and has key exports."""
        from stratum_patcher import event_logger
        assert hasattr(event_logger, "EventLogger")
        assert hasattr(event_logger, "generate_node_id")
        assert hasattr(event_logger, "make_node")
        assert hasattr(event_logger, "capture_output_signature")
        assert hasattr(event_logger, "classify_error")

    def test_generic_patch_import(self):
        """generic_patch module imports."""
        from stratum_patcher import generic_patch
        assert generic_patch is not None

    def test_openai_patch_import(self):
        """openai_patch module imports and has remap_model."""
        from stratum_patcher import openai_patch
        assert hasattr(openai_patch, "remap_model")
        assert hasattr(openai_patch, "patch")
        assert hasattr(openai_patch, "_build_payload")

    def test_crewai_patch_import(self):
        """crewai_patch module imports (crewai may not be installed)."""
        try:
            from stratum_patcher import crewai_patch
            assert crewai_patch is not None
        except ImportError:
            pytest.skip("crewai not installed")

    def test_autogen_patch_import(self):
        """autogen_patch module imports (autogen may not be installed)."""
        try:
            from stratum_patcher import autogen_patch
            assert autogen_patch is not None
        except ImportError:
            pytest.skip("autogen not installed")

    def test_litellm_patch_import(self):
        """litellm_patch module imports."""
        from stratum_patcher import litellm_patch
        assert hasattr(litellm_patch, "patch")
        assert hasattr(litellm_patch, "_build_litellm_payload")


# ===========================================================================
# PATCHER STATUS TESTS
# ===========================================================================

class TestPatcherStatus:
    def test_patcher_status_dict_populated(self):
        """__init__.py populates _patcher_status dict."""
        import stratum_patcher
        assert hasattr(stratum_patcher, "_patcher_status")
        status = stratum_patcher._patcher_status
        assert isinstance(status, dict)
        assert "generic" in status
        assert status["generic"] == "ok"

    def test_patcher_status_openai(self):
        """OpenAI patcher status is recorded."""
        import stratum_patcher
        assert "openai" in stratum_patcher._patcher_status

    def test_patcher_status_litellm(self):
        """LiteLLM patcher status is recorded."""
        import stratum_patcher
        assert "litellm" in stratum_patcher._patcher_status


# ===========================================================================
# MODEL REMAPPING TESTS
# ===========================================================================

class TestModelRemapping:
    def test_remap_no_vllm_model(self):
        """Without STRATUM_VLLM_MODEL, model names pass through."""
        from stratum_patcher.openai_patch import remap_model, VLLM_MODEL
        if VLLM_MODEL:
            pytest.skip("STRATUM_VLLM_MODEL is set in this env")
        assert remap_model("gpt-4") == "gpt-4"
        assert remap_model("claude-3-opus") == "claude-3-opus"

    def test_remap_with_vllm_model(self, monkeypatch):
        """With STRATUM_VLLM_MODEL set, known models are remapped."""
        monkeypatch.setattr("stratum_patcher.openai_patch.VLLM_MODEL", "Qwen/Qwen2.5-72B")
        from stratum_patcher.openai_patch import remap_model
        assert remap_model("gpt-4") == "Qwen/Qwen2.5-72B"
        assert remap_model("claude-3-opus") == "Qwen/Qwen2.5-72B"
        assert remap_model("o1-preview") == "Qwen/Qwen2.5-72B"
        assert remap_model("mistral-large") == "Qwen/Qwen2.5-72B"
        assert remap_model("llama-3.1-70b") == "Qwen/Qwen2.5-72B"
        assert remap_model("deepseek-coder") == "Qwen/Qwen2.5-72B"

    def test_remap_unknown_model_passthrough(self, monkeypatch):
        """Unknown model names pass through even with VLLM_MODEL set."""
        monkeypatch.setattr("stratum_patcher.openai_patch.VLLM_MODEL", "Qwen/Qwen2.5-72B")
        from stratum_patcher.openai_patch import remap_model
        assert remap_model("my-custom-model") == "my-custom-model"


# ===========================================================================
# EVENT LOGGER STRUCTURE TESTS
# ===========================================================================

class TestEventLoggerStructure:
    def test_generate_node_id_format(self):
        """Node IDs follow framework:class:file:line format."""
        from stratum_patcher.event_logger import generate_node_id
        nid = generate_node_id("crewai", "MyAgent", "agents.py", 42)
        assert nid == "crewai:MyAgent:agents.py:42"

    def test_make_node_structure(self):
        """make_node returns correct dict structure."""
        from stratum_patcher.event_logger import make_node
        node = make_node("agent", "crewai:Test:test.py:1", "TestAgent")
        assert node["node_type"] == "agent"
        assert node["node_id"] == "crewai:Test:test.py:1"
        assert node["node_name"] == "TestAgent"

    def test_classify_error_categories(self):
        """classify_error returns expected categories."""
        from stratum_patcher.event_logger import classify_error
        assert classify_error(TimeoutError("timed out")) == "timeout"
        assert classify_error(PermissionError("permission denied")) == "permission_error"
        assert classify_error(RuntimeError("test")) == "runtime_error"

    def test_capture_output_signature(self):
        """capture_output_signature returns expected keys."""
        from stratum_patcher.event_logger import capture_output_signature
        sig = capture_output_signature("hello world")
        assert "hash" in sig
        assert "type" in sig
        assert "size_bytes" in sig
        assert "preview" in sig
