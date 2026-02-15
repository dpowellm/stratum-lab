"""Evaluation script for failure classification.

Creates mock RunResult objects simulating all 8 failure modes recognised by
Orchestrator.classify_status() and verifies the classification.

Failure modes tested:
  1. SUCCESS
  2. PARTIAL_SUCCESS (exit 0, no events)
  3. DEPENDENCY_FAILURE
  4. ENTRY_POINT_FAILURE
  5. MODEL_FAILURE
  6. TIMEOUT
  7. CRASH
  8. INSTRUMENTATION_FAILURE
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

# ---------------------------------------------------------------------------
# Path setup -- stratum_lab is NOT an installed package
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stratum_lab.harness.container import RunResult
from stratum_lab.harness.orchestrator import Orchestrator
from stratum_lab.config import ExecutionStatus

# =========================================================================
# 1.  Create a minimal Orchestrator instance
# =========================================================================
# Orchestrator.__init__ only stores paths; it doesn't do I/O until .run().
# We need a selection_file that actually exists on disk.

_tmp_sel = tempfile.NamedTemporaryFile(
    mode="w", suffix=".json", delete=False, encoding="utf-8"
)
json.dump({"selections": [], "summary": {}}, _tmp_sel)
_tmp_sel.close()

_tmp_outdir = tempfile.mkdtemp(prefix="stratum_eval_fc_")

orch = Orchestrator(
    selection_file=_tmp_sel.name,
    output_dir=_tmp_outdir,
    vllm_url="http://localhost:8000/v1",
)

# =========================================================================
# 2.  Helper: write a temp events file with N lines
# =========================================================================

def _make_events_file(n_events: int) -> str:
    """Write a temporary JSONL file with *n_events* event lines.

    Returns the file path.
    """
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for i in range(n_events):
        json.dump({"event_id": f"e{i}", "event_type": "agent.task_start"}, f)
        f.write("\n")
    f.close()
    return f.name


# =========================================================================
# 3.  Define test cases
# =========================================================================

test_cases: list[dict] = [
    # 1) SUCCESS: exit 0, events file with >0 events, clean stderr
    {
        "label": "SUCCESS",
        "expected": ExecutionStatus.SUCCESS,
        "run_result": RunResult(
            run_id="test_success",
            repo_id="repo_001",
            status=ExecutionStatus.CRASH,  # initial; classify_status will refine
            exit_code=0,
            stderr="INFO: application started\nINFO: done",
            events_file_path=_make_events_file(10),
        ),
    },
    # 2) PARTIAL_SUCCESS: exit 0 but no events
    {
        "label": "PARTIAL_SUCCESS",
        "expected": ExecutionStatus.PARTIAL_SUCCESS,
        "run_result": RunResult(
            run_id="test_partial",
            repo_id="repo_002",
            status=ExecutionStatus.CRASH,
            exit_code=0,
            stderr="",
            events_file_path=_make_events_file(0),
        ),
    },
    # 3) DEPENDENCY_FAILURE: stderr contains "ModuleNotFoundError"
    {
        "label": "DEPENDENCY_FAILURE",
        "expected": ExecutionStatus.DEPENDENCY_FAILURE,
        "run_result": RunResult(
            run_id="test_dep",
            repo_id="repo_003",
            status=ExecutionStatus.CRASH,
            exit_code=1,
            stderr=(
                "Traceback (most recent call last):\n"
                "  File \"main.py\", line 1, in <module>\n"
                "ModuleNotFoundError: No module named 'crewai'\n"
            ),
            events_file_path=None,
        ),
    },
    # 4) ENTRY_POINT_FAILURE: stderr contains "FileNotFoundError"
    {
        "label": "ENTRY_POINT_FAILURE",
        "expected": ExecutionStatus.ENTRY_POINT_FAILURE,
        "run_result": RunResult(
            run_id="test_entry",
            repo_id="repo_004",
            status=ExecutionStatus.CRASH,
            exit_code=1,
            stderr=(
                "Traceback (most recent call last):\n"
                "FileNotFoundError: [Errno 2] No such file or directory: 'run.py'\n"
            ),
            events_file_path=None,
        ),
    },
    # 5) MODEL_FAILURE: stderr contains "openai.APIConnectionError"
    {
        "label": "MODEL_FAILURE",
        "expected": ExecutionStatus.MODEL_FAILURE,
        "run_result": RunResult(
            run_id="test_model",
            repo_id="repo_005",
            status=ExecutionStatus.CRASH,
            exit_code=1,
            stderr=(
                "Traceback (most recent call last):\n"
                "openai.APIConnectionError: Connection refused\n"
            ),
            events_file_path=None,
        ),
    },
    # 6) TIMEOUT: status already set to TIMEOUT by container layer
    {
        "label": "TIMEOUT",
        "expected": ExecutionStatus.TIMEOUT,
        "run_result": RunResult(
            run_id="test_timeout",
            repo_id="repo_006",
            status=ExecutionStatus.TIMEOUT,  # pre-set by container
            exit_code=None,
            stderr="",
            events_file_path=None,
        ),
    },
    # 7) CRASH: exit 1, no recognisable pattern in stderr
    {
        "label": "CRASH",
        "expected": ExecutionStatus.CRASH,
        "run_result": RunResult(
            run_id="test_crash",
            repo_id="repo_007",
            status=ExecutionStatus.CRASH,
            exit_code=1,
            stderr=(
                "Traceback (most recent call last):\n"
                "  File \"app.py\", line 42, in <module>\n"
                "RuntimeError: something went wrong\n"
            ),
            events_file_path=None,
        ),
    },
    # 8) INSTRUMENTATION_FAILURE: stderr contains "stratum_patcher"
    #    NOTE: the stderr must NOT also contain dependency-class keywords
    #    (like "ImportError") because classify_status checks dependency
    #    patterns first.  We use a message that only triggers the
    #    instrumentation check.
    {
        "label": "INSTRUMENTATION_FAILURE",
        "expected": ExecutionStatus.INSTRUMENTATION_FAILURE,
        "run_result": RunResult(
            run_id="test_instrument",
            repo_id="repo_008",
            status=ExecutionStatus.CRASH,
            exit_code=1,
            stderr=(
                "Error: stratum_patcher failed to initialise hooks.\n"
                "The instrumentation layer could not attach.\n"
            ),
            events_file_path=None,
        ),
    },
]


# =========================================================================
# 4.  Run classification and print results
# =========================================================================

print("=" * 80)
print("FAILURE CLASSIFICATION TEST RESULTS")
print("=" * 80)

passed = 0
failed = 0

for tc in test_cases:
    actual = orch.classify_status(tc["run_result"])
    expected = tc["expected"]
    ok = actual == expected
    status_str = "PASS" if ok else "FAIL"

    if ok:
        passed += 1
    else:
        failed += 1

    print(f"\n  [{status_str}] {tc['label']}")
    print(f"    expected : {expected}")
    print(f"    actual   : {actual}")

    if not ok:
        print(f"    stderr   : {tc['run_result'].stderr[:200]}")

print("\n" + "-" * 80)
print(f"  Results: {passed} passed, {failed} failed, {len(test_cases)} total")

# ---- Additional edge case tests ----
print("\n" + "=" * 80)
print("EDGE CASE TESTS")
print("=" * 80)

# exit_code != 0 but some events captured => PARTIAL_SUCCESS
edge1 = RunResult(
    run_id="edge_partial_with_events",
    repo_id="repo_edge_1",
    status=ExecutionStatus.CRASH,
    exit_code=2,
    stderr="RuntimeWarning: something\n",
    events_file_path=_make_events_file(5),
)
result_e1 = orch.classify_status(edge1)
ok_e1 = result_e1 == ExecutionStatus.PARTIAL_SUCCESS
print(f"\n  [{'PASS' if ok_e1 else 'FAIL'}] Non-zero exit with events => PARTIAL_SUCCESS")
print(f"    expected: {ExecutionStatus.PARTIAL_SUCCESS}")
print(f"    actual  : {result_e1}")
if ok_e1:
    passed += 1
else:
    failed += 1

# Dependency failure should take priority even with exit_code 0
edge2 = RunResult(
    run_id="edge_dep_exit0",
    repo_id="repo_edge_2",
    status=ExecutionStatus.CRASH,
    exit_code=0,
    stderr="ModuleNotFoundError: No module named 'langchain'\n",
    events_file_path=_make_events_file(0),
)
result_e2 = orch.classify_status(edge2)
ok_e2 = result_e2 == ExecutionStatus.DEPENDENCY_FAILURE
print(f"\n  [{'PASS' if ok_e2 else 'FAIL'}] Dependency error with exit 0 => DEPENDENCY_FAILURE")
print(f"    expected: {ExecutionStatus.DEPENDENCY_FAILURE}")
print(f"    actual  : {result_e2}")
if ok_e2:
    passed += 1
else:
    failed += 1

print("\n" + "-" * 80)
print(f"  Final: {passed} passed, {failed} failed, {len(test_cases) + 2} total")

# ---- Clean up temp files ----
for pattern in [_tmp_sel.name]:
    try:
        os.unlink(pattern)
    except OSError:
        pass

print("\nDone.")
