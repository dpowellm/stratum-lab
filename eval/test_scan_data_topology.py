"""Tests for Phase 0b static source code scanner (scan_data_topology.py).

Exercises scan_tool_registrations() and scan_state_keys() on synthetic repos.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, SCRIPT_DIR)
PARENT_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PARENT_DIR)

from scan_data_topology import scan_tool_registrations, scan_state_keys, main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_repo(files: dict[str, str]) -> str:
    tmpdir = tempfile.mkdtemp(prefix="stratum_test_datatopo_")
    for relpath, content in files.items():
        fpath = os.path.join(tmpdir, relpath)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
    return tmpdir


def _all_tool_names(result: dict) -> list[str]:
    """Extract all tool names from scan result (registered + unattributed)."""
    names = []
    for tools in result.get("tool_registrations", {}).values():
        names.extend(tools)
    names.extend(result.get("unattributed_tools", []))
    return names


# ===========================================================================
# TOOL DETECTION TESTS (6)
# ===========================================================================
class TestToolDetection:
    def test_at_tool_decorator(self):
        """@tool decorator → detects function name."""
        repo = _make_repo({"agents.py": (
            "from langchain.tools import tool\n"
            "\n"
            "@tool\n"
            "def search_documents(query: str):\n"
            "    return []\n"
        )})
        try:
            result = scan_tool_registrations(repo)
            assert "search_documents" in _all_tool_names(result)
            assert result["patterns_detected"]["decorator_tools"] >= 1
        finally:
            shutil.rmtree(repo)

    def test_tool_constructor(self):
        """Tool(name='web_search') → detects tool name."""
        repo = _make_repo({"tools.py": (
            "from langchain.tools import Tool\n"
            "\n"
            'web_tool = Tool(name="web_search", func=lambda q: q)\n'
        )})
        try:
            result = scan_tool_registrations(repo)
            assert "web_search" in _all_tool_names(result)
            assert result["patterns_detected"]["constructor_tools"] >= 1
        finally:
            shutil.rmtree(repo)

    def test_structured_tool(self):
        """StructuredTool.from_function → detects func name (regex captures func= first)."""
        repo = _make_repo({"tools.py": (
            "from langchain.tools import StructuredTool\n"
            "\n"
            "def calc(a, b): return a + b\n"
            'calculator = StructuredTool.from_function(func=calc, name="calculator")\n'
        )})
        try:
            result = scan_tool_registrations(repo)
            # Implementation regex captures func= argument (group 1) before name=
            assert "calc" in _all_tool_names(result)
            assert result["patterns_detected"]["constructor_tools"] >= 1
        finally:
            shutil.rmtree(repo)

    def test_api_pattern_requests(self):
        """requests.get() → detects http_client."""
        repo = _make_repo({"api_caller.py": (
            "import requests\n"
            "\n"
            "def fetch_data(url):\n"
            "    return requests.get(url)\n"
        )})
        try:
            result = scan_tool_registrations(repo)
            assert "http_client" in _all_tool_names(result)
            assert result["patterns_detected"]["api_patterns"] >= 1
        finally:
            shutil.rmtree(repo)

    def test_database_pattern(self):
        """cursor.execute() → detects database_query."""
        repo = _make_repo({"db.py": (
            "import sqlite3\n"
            "\n"
            "def query_db(conn):\n"
            "    cursor = conn.cursor()\n"
            '    cursor.execute("SELECT * FROM users")\n'
        )})
        try:
            result = scan_tool_registrations(repo)
            assert "database_query" in _all_tool_names(result)
            assert result["patterns_detected"]["database_patterns"] >= 1
        finally:
            shutil.rmtree(repo)

    def test_multiple_tools_in_one_file(self):
        """Multiple patterns in one file → tools_found >= 3."""
        repo = _make_repo({"multi.py": (
            "import requests\n"
            "from langchain.tools import tool, Tool\n"
            "\n"
            "@tool\n"
            "def search(query):\n"
            "    return []\n"
            "\n"
            'fetcher = Tool(name="fetcher", func=lambda: None)\n'
            "\n"
            "def call_api():\n"
            '    requests.post("http://example.com")\n'
        )})
        try:
            result = scan_tool_registrations(repo)
            assert result["tools_found"] >= 3
        finally:
            shutil.rmtree(repo)


# ===========================================================================
# STATE KEY DETECTION TESTS (3)
# ===========================================================================
class TestStateKeyDetection:
    def test_typed_dict_fields(self):
        """TypedDict fields → detected as state keys."""
        repo = _make_repo({"state.py": (
            "from typing import TypedDict\n"
            "\n"
            "class AgentState(TypedDict):\n"
            "    research_notes: str\n"
            "    final_report: str\n"
        )})
        try:
            keys = scan_state_keys(repo)
            assert "research_notes" in keys
            assert "final_report" in keys
        finally:
            shutil.rmtree(repo)

    def test_dict_access_patterns(self):
        """state['key'] → detected as state keys."""
        repo = _make_repo({"workflow.py": (
            "def process(state):\n"
            '    records = state["patient_records"]\n'
            "    info = state['salary_info']\n"
            "    return records, info\n"
        )})
        try:
            keys = scan_state_keys(repo)
            assert "patient_records" in keys
            assert "salary_info" in keys
        finally:
            shutil.rmtree(repo)

    def test_pydantic_model_fields(self):
        """Pydantic BaseModel fields → detected as state keys."""
        repo = _make_repo({"models.py": (
            "from pydantic import BaseModel\n"
            "\n"
            "class State(BaseModel):\n"
            "    treatment_plan: str\n"
        )})
        try:
            keys = scan_state_keys(repo)
            assert "treatment_plan" in keys
        finally:
            shutil.rmtree(repo)


# ===========================================================================
# EDGE CASE TESTS (3)
# ===========================================================================
class TestEdgeCases:
    def test_skips_excluded_directories(self):
        """Files in venv/ and .git/ are not scanned."""
        repo = _make_repo({
            "agents.py": (
                "@tool\n"
                "def real_tool(x):\n"
                "    pass\n"
            ),
            "venv/lib/foo.py": (
                "@tool\n"
                "def venv_tool(x):\n"
                "    pass\n"
            ),
            ".git/hooks/pre-commit.py": (
                "@tool\n"
                "def git_tool(x):\n"
                "    pass\n"
            ),
        })
        try:
            result = scan_tool_registrations(repo)
            all_names = _all_tool_names(result)
            assert "real_tool" in all_names
            assert "venv_tool" not in all_names
            assert "git_tool" not in all_names
        finally:
            shutil.rmtree(repo)

    def test_empty_repo(self):
        """Repo with no .py files → tools_found == 0, no crash."""
        repo = _make_repo({"README.md": "# Test repo\n"})
        try:
            result = scan_tool_registrations(repo)
            assert result["tools_found"] == 0
        finally:
            shutil.rmtree(repo)

    def test_binary_file_no_crash(self):
        """Binary file alongside valid Python → no crash."""
        tmpdir = tempfile.mkdtemp(prefix="stratum_test_datatopo_")
        with open(os.path.join(tmpdir, "agent.py"), "w", encoding="utf-8") as f:
            f.write("@tool\ndef valid_tool(x):\n    pass\n")
        with open(os.path.join(tmpdir, "data.py"), "wb") as f:
            f.write(b"\x80\x81\x82\xff\xfe\xfd")
        try:
            result = scan_tool_registrations(tmpdir)
            assert "valid_tool" in _all_tool_names(result)
        finally:
            shutil.rmtree(tmpdir)
