"""Event schema validation for v6 graph discovery reframe.

Covers validation checks 18-20:
  Check 18: Error context in events -- error events have active_node_stack
  Check 19: State access events -- state.access event type validated
  Check 20: Routing decision events -- routing.decision event type validated

Also tests EventLogger node-stack and error-context APIs, validates synthetic
events through the parser, and confirms run record counters.

Run as a standalone script:
    cd stratum-lab
    python eval/test_event_schema.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup -- stratum_lab / stratum_patcher are NOT installed packages
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "stratum_patcher"))

from rich.console import Console

from stratum_patcher.event_logger import EventLogger, make_node, generate_node_id
from stratum_lab.collection.parser import (
    validate_event,
    build_run_record,
    STATE_ACCESS_EVENT_TYPES,
    ROUTING_DECISION_EVENT_TYPES,
)

# ---------------------------------------------------------------------------
# Output setup -- rich console writes to both terminal and file
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "event-schema-validation.txt"

_file_buffer = io.StringIO()
_file_console = Console(file=_file_buffer, width=100, force_terminal=False, no_color=True)
_term_console = Console(width=100)

SEPARATOR = "=" * 78
THIN_SEP = "-" * 78


def out(text: str = "") -> None:
    """Print to both the terminal (with color) and the file buffer (plain)."""
    _term_console.print(text, highlight=False)
    _file_console.print(text, highlight=False)


def out_pass(label: str, detail: str = "") -> None:
    suffix = f"  {detail}" if detail else ""
    _term_console.print(f"  [green][+][/green] {label}  PASS{suffix}", highlight=False)
    _file_console.print(f"  [+] {label}  PASS{suffix}", highlight=False)


def out_fail(label: str, detail: str = "") -> None:
    suffix = f"  {detail}" if detail else ""
    _term_console.print(f"  [red][X][/red] {label}  FAIL{suffix}", highlight=False)
    _file_console.print(f"  [X] {label}  FAIL{suffix}", highlight=False)


def out_detail(text: str) -> None:
    _term_console.print(f"      {text}", highlight=False)
    _file_console.print(f"      {text}", highlight=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_event(
    event_type: str,
    *,
    payload: dict[str, Any] | None = None,
    source_node: dict[str, str] | None = None,
    target_node: dict[str, str] | None = None,
    edge_type: str | None = None,
) -> dict[str, Any]:
    """Build a synthetic event dict with all required fields."""
    record: dict[str, Any] = {
        "event_id": f"evt_{uuid.uuid4().hex[:16]}",
        "timestamp_ns": time.time_ns(),
        "run_id": "schema-v6-test-run",
        "repo_id": "schema-v6-test-repo",
        "framework": "test",
        "event_type": event_type,
    }
    if source_node is not None:
        record["source_node"] = source_node
    if target_node is not None:
        record["target_node"] = target_node
    if edge_type is not None:
        record["edge_type"] = edge_type
    if payload is not None:
        record["payload"] = payload
    record["stack_depth"] = 0
    return record


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    pass_count = 0
    fail_count = 0

    out(SEPARATOR)
    out("  STRATUM EVENT SCHEMA VALIDATION -- v6 GRAPH DISCOVERY REFRAME")
    out("  Checks 18-20: error context, state access, routing decisions")
    out(SEPARATOR)
    out()

    # -----------------------------------------------------------------
    # CHECK 18: Error context in events -- active_node_stack
    # -----------------------------------------------------------------
    out(SEPARATOR)
    out("  CHECK 18: ERROR CONTEXT IN EVENTS")
    out(SEPARATOR)
    out()

    # 18a: Verify EventLogger has the required node-stack and error-context methods
    required_methods = [
        "push_active_node",
        "pop_active_node",
        "current_node",
        "record_error_context",
    ]
    for method_name in required_methods:
        label = f"EventLogger.{method_name} exists"
        if hasattr(EventLogger, method_name) and callable(getattr(EventLogger, method_name)):
            out_pass(f"{label:<50}")
            pass_count += 1
        else:
            out_fail(f"{label:<50}", "method not found on EventLogger")
            fail_count += 1

    out()

    # 18b: Exercise push/pop/current on a fresh singleton
    fd, events_file = tempfile.mkstemp(suffix=".jsonl", prefix="stratum_v6_schema_")
    os.close(fd)

    os.environ["STRATUM_EVENTS_FILE"] = events_file
    os.environ["STRATUM_RUN_ID"] = "schema-v6-test-run"
    os.environ["STRATUM_REPO_ID"] = "schema-v6-test-repo"
    os.environ["STRATUM_FRAMEWORK"] = "test"

    EventLogger._instance = None
    logger = EventLogger.get()

    node_a = "test:AgentA:main.py:10"
    node_b = "test:AgentB:main.py:20"

    logger.push_active_node(node_a)
    logger.push_active_node(node_b)

    label = "push_active_node stacks correctly"
    if logger.current_node() == node_b:
        out_pass(f"{label:<50}")
        pass_count += 1
    else:
        out_fail(f"{label:<50}", f"expected {node_b}, got {logger.current_node()}")
        fail_count += 1

    popped = logger.pop_active_node()
    label = "pop_active_node returns top node"
    if popped == node_b and logger.current_node() == node_a:
        out_pass(f"{label:<50}")
        pass_count += 1
    else:
        out_fail(f"{label:<50}", f"popped={popped}, current={logger.current_node()}")
        fail_count += 1

    # 18c: record_error_context captures active stack snapshot
    logger.push_active_node(node_b)  # stack: [node_a, node_b]
    logger.record_error_context(
        node_id=node_b,
        error_type="ValueError",
        error_msg="test error for schema validation",
        upstream_node=node_a,
        upstream_output_hash="abc123",
    )

    label = "record_error_context captures stack"
    if (
        len(logger._error_context_stack) == 1
        and logger._error_context_stack[0]["active_stack"] == [node_a, node_b]
    ):
        out_pass(f"{label:<50}")
        out_detail(f"active_stack: {logger._error_context_stack[0]['active_stack']}")
        pass_count += 1
    else:
        out_fail(f"{label:<50}", f"stack={logger._error_context_stack}")
        fail_count += 1

    # Clean up logger stack for remaining tests
    logger.pop_active_node()
    logger.pop_active_node()

    out()

    # 18d: Create a synthetic error event with active_node_stack in payload
    error_event = _make_synthetic_event(
        "error.occurred",
        source_node=make_node("agent", node_b, "AgentB"),
        payload={
            "error_type": "ValueError",
            "error_message": "invalid input format",
            "file": "main.py",
            "line": 42,
            "active_node_stack": [node_a, node_b],
        },
    )

    label = "error event passes validate_event"
    if validate_event(error_event):
        out_pass(f"{label:<50}")
        pass_count += 1
    else:
        out_fail(f"{label:<50}", "validate_event returned False")
        fail_count += 1

    label = "error event has active_node_stack"
    payload = error_event.get("payload", {})
    if "active_node_stack" in payload and isinstance(payload["active_node_stack"], list):
        out_pass(f"{label:<50}")
        out_detail(f"active_node_stack: {payload['active_node_stack']}")
        pass_count += 1
    else:
        out_fail(f"{label:<50}", "missing or wrong type for active_node_stack")
        fail_count += 1

    label = "active_node_stack has 2 entries"
    if len(payload.get("active_node_stack", [])) == 2:
        out_pass(f"{label:<50}")
        pass_count += 1
    else:
        out_fail(
            f"{label:<50}",
            f"expected 2, got {len(payload.get('active_node_stack', []))}",
        )
        fail_count += 1

    out()

    # -----------------------------------------------------------------
    # CHECK 19: State access events
    # -----------------------------------------------------------------
    out(SEPARATOR)
    out("  CHECK 19: STATE ACCESS EVENTS")
    out(SEPARATOR)
    out()

    # 19a: Verify state.access is in STATE_ACCESS_EVENT_TYPES
    label = "state.access in STATE_ACCESS_EVENT_TYPES"
    if "state.access" in STATE_ACCESS_EVENT_TYPES:
        out_pass(f"{label:<50}")
        out_detail(f"STATE_ACCESS_EVENT_TYPES = {STATE_ACCESS_EVENT_TYPES}")
        pass_count += 1
    else:
        out_fail(f"{label:<50}", f"set contents: {STATE_ACCESS_EVENT_TYPES}")
        fail_count += 1

    # 19b: Create a synthetic state.access event with required fields
    state_access_event = _make_synthetic_event(
        "state.access",
        source_node=make_node("agent", "test:AgentA:main.py:10", "AgentA"),
        payload={
            "state_key": "shared_memory.research_notes",
            "accessor_node": "test:AgentA:main.py:10",
            "access_type": "read",
        },
    )

    label = "state.access event passes validate_event"
    if validate_event(state_access_event):
        out_pass(f"{label:<50}")
        pass_count += 1
    else:
        out_fail(f"{label:<50}", "validate_event returned False")
        fail_count += 1

    # 19c: Verify payload fields
    sa_payload = state_access_event.get("payload", {})
    required_sa_fields = ["state_key", "accessor_node", "access_type"]
    missing_sa = [f for f in required_sa_fields if f not in sa_payload]
    label = "state.access payload has required fields"
    if not missing_sa:
        out_pass(f"{label:<50}")
        out_detail(f"state_key={sa_payload['state_key']}")
        out_detail(f"accessor_node={sa_payload['accessor_node']}")
        out_detail(f"access_type={sa_payload['access_type']}")
        pass_count += 1
    else:
        out_fail(f"{label:<50}", f"missing: {missing_sa}")
        fail_count += 1

    # 19d: Test a write-type state access
    state_write_event = _make_synthetic_event(
        "state.access",
        source_node=make_node("agent", "test:AgentB:main.py:20", "AgentB"),
        payload={
            "state_key": "shared_memory.draft_output",
            "accessor_node": "test:AgentB:main.py:20",
            "access_type": "write",
        },
    )

    label = "state.access (write) passes validate_event"
    if validate_event(state_write_event):
        out_pass(f"{label:<50}")
        pass_count += 1
    else:
        out_fail(f"{label:<50}", "validate_event returned False")
        fail_count += 1

    out()

    # -----------------------------------------------------------------
    # CHECK 20: Routing decision events
    # -----------------------------------------------------------------
    out(SEPARATOR)
    out("  CHECK 20: ROUTING DECISION EVENTS")
    out(SEPARATOR)
    out()

    # 20a: Verify routing.decision is in ROUTING_DECISION_EVENT_TYPES
    label = "routing.decision in ROUTING_DECISION_EVENT_TYPES"
    if "routing.decision" in ROUTING_DECISION_EVENT_TYPES:
        out_pass(f"{label:<50}")
        out_detail(f"ROUTING_DECISION_EVENT_TYPES = {ROUTING_DECISION_EVENT_TYPES}")
        pass_count += 1
    else:
        out_fail(f"{label:<50}", f"set contents: {ROUTING_DECISION_EVENT_TYPES}")
        fail_count += 1

    # 20b: Create a synthetic routing.decision event with required fields
    routing_event = _make_synthetic_event(
        "routing.decision",
        source_node=make_node("agent", "test:Router:main.py:5", "Router"),
        payload={
            "decision_type": "next_agent",
            "source_node": "test:Router:main.py:5",
            "selected_target": "test:AgentA:main.py:10",
            "candidates": [
                "test:AgentA:main.py:10",
                "test:AgentB:main.py:20",
                "test:AgentC:main.py:30",
            ],
        },
    )

    label = "routing.decision event passes validate_event"
    if validate_event(routing_event):
        out_pass(f"{label:<50}")
        pass_count += 1
    else:
        out_fail(f"{label:<50}", "validate_event returned False")
        fail_count += 1

    # 20c: Verify payload fields
    rd_payload = routing_event.get("payload", {})
    required_rd_fields = ["decision_type", "source_node", "selected_target", "candidates"]
    missing_rd = [f for f in required_rd_fields if f not in rd_payload]
    label = "routing.decision payload has required fields"
    if not missing_rd:
        out_pass(f"{label:<50}")
        out_detail(f"decision_type={rd_payload['decision_type']}")
        out_detail(f"source_node={rd_payload['source_node']}")
        out_detail(f"selected_target={rd_payload['selected_target']}")
        out_detail(f"candidates ({len(rd_payload['candidates'])}): {rd_payload['candidates']}")
        pass_count += 1
    else:
        out_fail(f"{label:<50}", f"missing: {missing_rd}")
        fail_count += 1

    # 20d: Test with a different decision_type
    routing_event_2 = _make_synthetic_event(
        "routing.decision",
        source_node=make_node("agent", "test:Orchestrator:main.py:1", "Orchestrator"),
        payload={
            "decision_type": "tool_selection",
            "source_node": "test:Orchestrator:main.py:1",
            "selected_target": "test:WebSearch:tools.py:10",
            "candidates": [
                "test:WebSearch:tools.py:10",
                "test:Calculator:tools.py:20",
            ],
        },
    )

    label = "routing.decision (tool_selection) validates"
    if validate_event(routing_event_2):
        out_pass(f"{label:<50}")
        pass_count += 1
    else:
        out_fail(f"{label:<50}", "validate_event returned False")
        fail_count += 1

    out()

    # -----------------------------------------------------------------
    # PARSER RUN RECORD INTEGRATION
    # -----------------------------------------------------------------
    out(SEPARATOR)
    out("  RUN RECORD INTEGRATION: state_access_count + routing_decision_count")
    out(SEPARATOR)
    out()

    # Build a mixed event list and pass through build_run_record
    mixed_events = [
        # Some baseline events
        _make_synthetic_event(
            "execution.start",
            source_node=make_node("agent", "test:AgentA:main.py:10", "AgentA"),
        ),
        _make_synthetic_event(
            "agent.task_start",
            source_node=make_node("agent", "test:AgentA:main.py:10", "AgentA"),
        ),
        # 2 state.access events
        state_access_event,
        state_write_event,
        # 2 routing.decision events
        routing_event,
        routing_event_2,
        # error event with active_node_stack
        error_event,
        # close out
        _make_synthetic_event(
            "agent.task_end",
            source_node=make_node("agent", "test:AgentA:main.py:10", "AgentA"),
            payload={"status": "success"},
        ),
        _make_synthetic_event(
            "execution.end",
            source_node=make_node("agent", "test:AgentA:main.py:10", "AgentA"),
        ),
    ]

    run_record = build_run_record(mixed_events)

    # Verify state_access_count
    label = "run_record.state_access_count == 2"
    actual_sa = run_record.get("state_access_count", -1)
    if actual_sa == 2:
        out_pass(f"{label:<50}")
        out_detail(f"state_access_count = {actual_sa}")
        pass_count += 1
    else:
        out_fail(f"{label:<50}", f"expected 2, got {actual_sa}")
        fail_count += 1

    # Verify routing_decision_count
    label = "run_record.routing_decision_count == 2"
    actual_rd = run_record.get("routing_decision_count", -1)
    if actual_rd == 2:
        out_pass(f"{label:<50}")
        out_detail(f"routing_decision_count = {actual_rd}")
        pass_count += 1
    else:
        out_fail(f"{label:<50}", f"expected 2, got {actual_rd}")
        fail_count += 1

    # Verify total_events count
    label = "run_record.total_events == 9"
    actual_total = run_record.get("total_events", -1)
    if actual_total == len(mixed_events):
        out_pass(f"{label:<50}")
        out_detail(f"total_events = {actual_total}")
        pass_count += 1
    else:
        out_fail(f"{label:<50}", f"expected {len(mixed_events)}, got {actual_total}")
        fail_count += 1

    # Verify event types counted include state.access and routing.decision
    type_counts = run_record.get("total_events_by_type", {})
    label = "type_counts includes state.access"
    if type_counts.get("state.access", 0) == 2:
        out_pass(f"{label:<50}")
        pass_count += 1
    else:
        out_fail(f"{label:<50}", f"got {type_counts.get('state.access', 0)}")
        fail_count += 1

    label = "type_counts includes routing.decision"
    if type_counts.get("routing.decision", 0) == 2:
        out_pass(f"{label:<50}")
        pass_count += 1
    else:
        out_fail(f"{label:<50}", f"got {type_counts.get('routing.decision', 0)}")
        fail_count += 1

    # Verify error_summary captured the error event
    error_summary = run_record.get("error_summary", {})
    label = "error_summary.total_errors == 1"
    if error_summary.get("total_errors", 0) == 1:
        out_pass(f"{label:<50}")
        pass_count += 1
    else:
        out_fail(f"{label:<50}", f"got {error_summary.get('total_errors', 0)}")
        fail_count += 1

    out()

    # -----------------------------------------------------------------
    # ZERO-COUNT EDGE CASE
    # -----------------------------------------------------------------
    out(SEPARATOR)
    out("  EDGE CASE: Run with no state/routing events")
    out(SEPARATOR)
    out()

    empty_events = [
        _make_synthetic_event(
            "execution.start",
            source_node=make_node("agent", "test:A:m.py:1", "A"),
        ),
        _make_synthetic_event(
            "execution.end",
            source_node=make_node("agent", "test:A:m.py:1", "A"),
        ),
    ]
    empty_record = build_run_record(empty_events)

    label = "state_access_count == 0 when none present"
    if empty_record.get("state_access_count", -1) == 0:
        out_pass(f"{label:<50}")
        pass_count += 1
    else:
        out_fail(f"{label:<50}", f"got {empty_record.get('state_access_count')}")
        fail_count += 1

    label = "routing_decision_count == 0 when none present"
    if empty_record.get("routing_decision_count", -1) == 0:
        out_pass(f"{label:<50}")
        pass_count += 1
    else:
        out_fail(f"{label:<50}", f"got {empty_record.get('routing_decision_count')}")
        fail_count += 1

    out()

    # -----------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------
    out(THIN_SEP)
    out(f"  Results: {pass_count} passed, {fail_count} failed, "
        f"{pass_count + fail_count} total")
    out(THIN_SEP)
    out()
    out(SEPARATOR)
    if fail_count == 0:
        _term_console.print("  [bold green]ALL CHECKS PASSED[/bold green]", highlight=False)
        _file_console.print("  ALL CHECKS PASSED", highlight=False)
    else:
        _term_console.print(
            f"  [bold red]{fail_count} CHECK(S) FAILED[/bold red]", highlight=False
        )
        _file_console.print(f"  {fail_count} CHECK(S) FAILED", highlight=False)
    out(SEPARATOR)

    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------
    EventLogger._instance = None
    for key in ("STRATUM_EVENTS_FILE", "STRATUM_RUN_ID",
                "STRATUM_REPO_ID", "STRATUM_FRAMEWORK"):
        os.environ.pop(key, None)
    try:
        os.unlink(events_file)
    except OSError:
        pass

    # Write output file
    OUTPUT_FILE.write_text(_file_buffer.getvalue(), encoding="utf-8")

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
