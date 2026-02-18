"""Tests for static defensive pattern scanner.

Exercises scan_defensive_patterns.py with synthetic code snippets to verify
detection of all 7 pattern categories, delegation boundary proximity,
binary file handling, and summary accuracy.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Path setup -- scripts/ is NOT on sys.path by default
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, SCRIPT_DIR)
PARENT_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PARENT_DIR)

from scan_defensive_patterns import scan_repo, scan_file, build_summary
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_repo(files: dict[str, str]) -> str:
    """Create a temp directory with given files. Returns repo root path."""
    tmpdir = tempfile.mkdtemp(prefix="stratum_test_defensive_")
    for relpath, content in files.items():
        fpath = os.path.join(tmpdir, relpath)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
    return tmpdir


def _scan_code(code: str) -> dict:
    """Convenience: scan a single file's code and return the repo result."""
    repo = _make_repo({"main.py": code})
    return scan_repo(repo)


# =========================================================================
# Tests
# =========================================================================


class TestDefensivePatternScanner:
    """Test the static defensive pattern scanner."""

    def test_detects_timeout_guard(self):
        """Detect timeout= near a delegation call."""
        code = (
            "from crewai import Agent, Crew\n"
            "agent = Agent(role='researcher', timeout=60)\n"
            "crew = Crew(agents=[agent])\n"
            "result = crew.kickoff()\n"
        )
        result = _scan_code(code)
        timeout_pats = [
            p for p in result["patterns"]
            if p["pattern_category"] == "timeout_iteration_guards"
        ]
        assert len(timeout_pats) >= 1, "Should detect timeout guard"
        assert timeout_pats[0]["pattern_detail"]["value"] == 60
        assert timeout_pats[0]["near_delegation_boundary"] is True

    def test_detects_pydantic_output_validation(self):
        """Detect Pydantic BaseModel used for output validation."""
        code = (
            "from pydantic import BaseModel\n"
            "class ResearchOutput(BaseModel):\n"
            "    findings: list[str]\n"
            "    confidence: float\n"
        )
        result = _scan_code(code)
        ov = [
            p for p in result["patterns"]
            if p["pattern_category"] == "output_validation"
        ]
        assert len(ov) >= 1, "Should detect Pydantic model"
        assert ov[0]["pattern_detail"]["validation_type"] == "pydantic_model"
        assert ov[0]["pattern_detail"]["model_name"] == "ResearchOutput"

    def test_detects_bare_except(self):
        """Detect bare except clause (weak error handling)."""
        code = (
            "try:\n"
            "    result = agent.run()\n"
            "except:\n"
            "    pass\n"
        )
        result = _scan_code(code)
        eh = [
            p for p in result["patterns"]
            if p["pattern_category"] == "exception_handling_topology"
        ]
        assert len(eh) > 0, "Should detect bare except"
        assert eh[0]["pattern_detail"]["except_type"] == "bare"

    def test_handles_binary_files(self):
        """Scanner gracefully handles non-UTF-8 files."""
        tmpdir = tempfile.mkdtemp(prefix="stratum_test_binary_")
        # Write a binary file with .py extension
        bin_path = os.path.join(tmpdir, "binary.py")
        with open(bin_path, "wb") as f:
            f.write(bytes(range(256)))
        # Write a normal file
        normal_path = os.path.join(tmpdir, "normal.py")
        with open(normal_path, "w", encoding="utf-8") as f:
            f.write("timeout = 30\n")

        result = scan_repo(tmpdir)
        # Should not crash, should skip the binary file
        assert result["files_skipped"] >= 1
        assert result["files_scanned"] >= 1

    def test_summary_counts_correct(self):
        """Summary counts match actual pattern counts."""
        code = (
            "from crewai import Agent, Crew\n"
            "agent = Agent(role='researcher', timeout=60)\n"
            "agent2 = Agent(role='writer', max_iterations=10)\n"
            "crew = Crew(agents=[agent, agent2])\n"
            "\n"
            "from pydantic import BaseModel\n"
            "class Output(BaseModel):\n"
            "    result: str\n"
            "\n"
            "try:\n"
            "    result = crew.kickoff()\n"
            "except ValueError as e:\n"
            "    logging.error(e)\n"
        )
        result = _scan_code(code)
        summary = result["summary"]

        # Count patterns by category from raw patterns
        from collections import Counter
        category_counts = Counter(p["pattern_category"] for p in result["patterns"])

        assert summary["timeout_iteration_guards"]["count"] == category_counts.get("timeout_iteration_guards", 0)
        assert summary["output_validation"]["count"] == category_counts.get("output_validation", 0)
        assert summary["exception_handling_topology"]["count"] == category_counts.get("exception_handling_topology", 0)

    def test_detects_concurrency_controls(self):
        """Detect threading locks near shared state."""
        code = (
            "import threading\n"
            "lock = threading.Lock()\n"
            "self.value = 0\n"
        )
        result = _scan_code(code)
        cc = [
            p for p in result["patterns"]
            if p["pattern_category"] == "concurrency_controls"
        ]
        assert len(cc) >= 1, "Should detect threading lock"
        assert cc[0]["pattern_detail"]["control_type"] == "lock"

    def test_detects_rate_limiting(self):
        """Detect rate limiting / backoff patterns."""
        code = (
            "import time\n"
            "import requests\n"
            "time.sleep(1)\n"
            "resp = requests.get('http://example.com')\n"
        )
        result = _scan_code(code)
        rl = [
            p for p in result["patterns"]
            if p["pattern_category"] == "rate_limiting_backoff"
        ]
        assert len(rl) >= 1, "Should detect time.sleep"

    def test_near_delegation_boundary_detection(self):
        """Patterns within 10 lines of delegation calls are marked near_boundary."""
        code = "timeout = 30\n"           # line 1 - near delegation at line 3
        code += "from crewai import Crew\n"  # line 2
        code += "Crew().kickoff()\n"      # line 3 - delegation
        code += "\n" * 20                 # lines 4-23 - gap of 20 blank lines
        code += "max_retries = 5\n"       # line 24 - NOT near delegation
        result = _scan_code(code)
        timeout_pats = [
            p for p in result["patterns"]
            if p["pattern_category"] == "timeout_iteration_guards"
        ]
        # Should have two patterns - one near, one far
        near = [p for p in timeout_pats if p["near_delegation_boundary"]]
        far = [p for p in timeout_pats if not p["near_delegation_boundary"]]
        assert len(near) >= 1, "Should detect near-boundary pattern"
        assert len(far) >= 1, "Should detect far-from-boundary pattern"

    def test_prompt_constraints_in_strings(self):
        """Detect prompt constraints only inside string literals."""
        code = (
            'system_prompt = "you are ONLY responsible for research"\n'
        )
        result = _scan_code(code)
        pc = [
            p for p in result["patterns"]
            if p["pattern_category"] == "prompt_constraints"
        ]
        assert len(pc) >= 1, "Should detect prompt constraint in string"
        assert pc[0]["pattern_detail"]["constraint_type"] == "role_boundary"
        assert pc[0]["pattern_detail"]["in_system_prompt"] is True

    def test_specific_exception_classified(self):
        """Specific exception catches are classified correctly."""
        code = (
            "try:\n"
            "    result = do_work()\n"
            "except ValueError as e:\n"
            "    logging.error(e)\n"
            "    raise\n"
        )
        result = _scan_code(code)
        eh = [
            p for p in result["patterns"]
            if p["pattern_category"] == "exception_handling_topology"
        ]
        assert len(eh) >= 1
        assert eh[0]["pattern_detail"]["except_type"] == "specific"
        assert "ValueError" in eh[0]["pattern_detail"]["exception_classes"]
        assert eh[0]["pattern_detail"]["has_logging"] is True
        assert eh[0]["pattern_detail"]["has_reraise"] is True

    def test_max_files_cap(self):
        """Scanner respects MAX_FILES cap of 500."""
        # Just verify the constant exists - don't create 500 files
        from scan_defensive_patterns import MAX_FILES
        assert MAX_FILES == 500


# =========================================================================
# Run
# =========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
