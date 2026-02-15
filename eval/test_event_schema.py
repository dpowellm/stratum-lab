"""Schema validation for all 19 event types emitted by stratum-patcher.

Generates synthetic JSONL events for every event type using the EventLogger
singleton, reads them back, and validates each against the required schema.

Run as a standalone script:
    cd stratum-lab
    python eval/test_event_schema.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure stratum_patcher is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "stratum_patcher"))


# ---------------------------------------------------------------------------
# Required and optional schema fields
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = {"event_id", "timestamp_ns", "run_id", "repo_id", "event_type"}

OPTIONAL_FIELDS = {
    "framework", "source_node", "target_node", "edge_type",
    "payload", "parent_event_id", "stack_depth",
}

# All 19 event types the patchers emit
ALL_EVENT_TYPES = [
    "execution.start",
    "execution.end",
    "agent.task_start",
    "agent.task_end",
    "delegation.initiated",
    "delegation.completed",
    "tool.invoked",
    "tool.completed",
    "tool.call_failure",
    "llm.call_start",
    "llm.call_end",
    "data.read",
    "data.write",
    "error.occurred",
    "error.propagated",
    "error.cascade",
    "decision.made",
    "guardrail.triggered",
    "external.call",
]

# Additional event types that appear in file I/O
FILE_EVENT_TYPES = [
    "file.read",
    "file.write",
]


def main() -> None:
    # -----------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------
    fd, events_file = tempfile.mkstemp(suffix=".jsonl", prefix="stratum_schema_test_")
    os.close(fd)

    os.environ["STRATUM_EVENTS_FILE"] = events_file
    os.environ["STRATUM_RUN_ID"] = "schema-test-run-001"
    os.environ["STRATUM_REPO_ID"] = "schema-test-repo-001"
    os.environ["STRATUM_FRAMEWORK"] = "schema-test"

    from stratum_patcher.event_logger import EventLogger, make_node, generate_node_id

    # Reset singleton
    EventLogger._instance = None
    logger = EventLogger.get()

    # -----------------------------------------------------------------
    # Generate events for all types
    # -----------------------------------------------------------------
    src_agent = make_node("agent", "test:Agent:main.py:1", "TestAgent")
    tgt_agent = make_node("agent", "test:Worker:main.py:10", "WorkerAgent")
    src_tool = make_node("capability", "test:Tool:main.py:20", "WebSearch")
    src_data = make_node("data_store", "test:Store:main.py:30", "SharedMemory")
    src_external = make_node("external", "test:HTTP:main.py:40", "api.example.com")
    src_guardrail = make_node("capability", "test:Guard:main.py:50", "ContentFilter")

    # Track parent event IDs for chaining
    parent_ids: dict[str, str] = {}

    event_configs = [
        # (event_type, kwargs_for_log_event)
        ("execution.start", dict(
            source_node=src_agent,
            payload={"crew_name": "TestCrew", "agent_count": 3, "task_count": 2},
        )),
        ("execution.end", dict(
            source_node=src_agent,
            payload={"latency_ms": 1234.5, "status": "success"},
            parent_event_id="__execution.start__",
        )),
        ("agent.task_start", dict(
            source_node=src_agent,
            payload={"agent_role": "Researcher", "task_name": "find_info"},
        )),
        ("agent.task_end", dict(
            source_node=src_agent,
            payload={"agent_role": "Researcher", "latency_ms": 567.8, "status": "success"},
            parent_event_id="__agent.task_start__",
        )),
        ("delegation.initiated", dict(
            source_node=src_agent,
            target_node=tgt_agent,
            edge_type="delegates_to",
            payload={"delegator": "TestAgent", "delegate": "WorkerAgent"},
        )),
        ("delegation.completed", dict(
            source_node=src_agent,
            target_node=tgt_agent,
            edge_type="delegates_to",
            payload={"delegator": "TestAgent", "delegate": "WorkerAgent", "status": "success"},
            parent_event_id="__delegation.initiated__",
        )),
        ("tool.invoked", dict(
            source_node=src_agent,
            target_node=src_tool,
            edge_type="calls",
            payload={"tool_name": "WebSearch", "args_shape": "dict(keys=['query'], len=1)"},
        )),
        ("tool.completed", dict(
            source_node=src_agent,
            target_node=src_tool,
            edge_type="calls",
            payload={"tool_name": "WebSearch", "latency_ms": 89.2, "status": "success"},
            parent_event_id="__tool.invoked__",
        )),
        ("tool.call_failure", dict(
            source_node=src_tool,
            payload={"tool_name": "WebSearch", "reason": "invalid_arguments_json"},
        )),
        ("llm.call_start", dict(
            source_node=make_node("capability", "test:LLM:main.py:100", "openai.chat.completions.create"),
            payload={"model_requested": "gpt-4", "message_count": 5, "has_tools": True},
        )),
        ("llm.call_end", dict(
            source_node=make_node("capability", "test:LLM:main.py:100", "openai.chat.completions.create"),
            payload={
                "model_requested": "gpt-4", "model_actual": "gpt-4",
                "latency_ms": 450.1, "input_tokens": 500, "output_tokens": 200,
                "finish_reason": "stop",
            },
            parent_event_id="__llm.call_start__",
        )),
        ("data.read", dict(
            source_node=src_data,
            payload={"channel": "SharedMemory", "data_shape": "dict(keys=['key1'], len=1)"},
        )),
        ("data.write", dict(
            source_node=src_data,
            payload={"channel": "SharedMemory", "data_shape": "str(len=42)"},
        )),
        ("error.occurred", dict(
            source_node=src_agent,
            payload={
                "error_type": "ValueError",
                "error_message": "invalid input format",
                "file": "main.py",
                "line": 42,
            },
        )),
        ("error.propagated", dict(
            source_node=src_agent,
            target_node=tgt_agent,
            edge_type="propagates_error",
            payload={
                "error_type": "ValueError",
                "original_source": "TestAgent",
                "propagated_to": "WorkerAgent",
            },
        )),
        ("error.cascade", dict(
            source_node=src_agent,
            payload={
                "error_type": "ValueError",
                "cascade_depth": 3,
                "affected_agents": ["TestAgent", "WorkerAgent", "ReviewerAgent"],
            },
        )),
        ("decision.made", dict(
            source_node=src_agent,
            payload={
                "decision_type": "routing",
                "options_considered": ["plan_a", "plan_b"],
                "selected": "plan_a",
                "confidence": 0.85,
            },
        )),
        ("guardrail.triggered", dict(
            source_node=src_guardrail,
            payload={
                "guardrail_name": "ContentFilter",
                "trigger_reason": "harmful_content_detected",
                "action_taken": "blocked",
            },
        )),
        ("external.call", dict(
            source_node=src_external,
            payload={
                "method": "GET",
                "domain": "api.example.com",
                "latency_ms": 156.3,
                "status_code": 200,
            },
        )),
    ]

    # Also add file.read and file.write (beyond the core 19, these are
    # emitted by generic_patch)
    event_configs.append(("file.read", dict(
        source_node=make_node("data_store", "test:FileIO:data.csv:0", "/app/data.csv"),
        payload={"path": "/app/data.csv", "mode": "r"},
    )))
    event_configs.append(("file.write", dict(
        source_node=make_node("data_store", "test:FileIO:output.txt:0", "/app/output.txt"),
        payload={"path": "/app/output.txt", "mode": "w"},
    )))

    # -----------------------------------------------------------------
    # Emit all events
    # -----------------------------------------------------------------
    for event_type, kwargs in event_configs:
        # Resolve parent_event_id placeholders
        if "parent_event_id" in kwargs:
            placeholder = kwargs["parent_event_id"]
            if placeholder.startswith("__") and placeholder.endswith("__"):
                ref_type = placeholder[2:-2]
                kwargs["parent_event_id"] = parent_ids.get(ref_type)

        eid = logger.log_event(event_type, **kwargs)
        parent_ids[event_type] = eid

    # -----------------------------------------------------------------
    # Read back and validate
    # -----------------------------------------------------------------
    events: list[dict] = []
    with open(events_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    print("=" * 72)
    print(f"  STRATUM EVENT SCHEMA VALIDATION")
    print(f"  Events file: {events_file}")
    print(f"  Total events emitted: {len(events)}")
    print("=" * 72)
    print()

    all_event_types = ALL_EVENT_TYPES + FILE_EVENT_TYPES
    event_by_type: dict[str, list[dict]] = {}
    for ev in events:
        et = ev.get("event_type", "UNKNOWN")
        event_by_type.setdefault(et, []).append(ev)

    pass_count = 0
    fail_count = 0
    results: list[tuple[str, bool, str]] = []

    for et in all_event_types:
        if et not in event_by_type:
            results.append((et, False, "NO EVENT GENERATED"))
            fail_count += 1
            continue

        ev = event_by_type[et][0]  # Check the first one
        errors: list[str] = []

        # Check required fields
        for field in REQUIRED_FIELDS:
            if field not in ev:
                errors.append(f"missing required field '{field}'")

        # Validate field types
        if "event_id" in ev and not ev["event_id"].startswith("evt_"):
            errors.append(f"event_id should start with 'evt_', got '{ev['event_id']}'")
        if "timestamp_ns" in ev and not isinstance(ev["timestamp_ns"], int):
            errors.append(f"timestamp_ns should be int, got {type(ev['timestamp_ns']).__name__}")
        if "run_id" in ev and not isinstance(ev["run_id"], str):
            errors.append(f"run_id should be str, got {type(ev['run_id']).__name__}")
        if "repo_id" in ev and not isinstance(ev["repo_id"], str):
            errors.append(f"repo_id should be str, got {type(ev['repo_id']).__name__}")

        # Check event_type matches
        if ev.get("event_type") != et:
            errors.append(f"event_type mismatch: expected '{et}', got '{ev.get('event_type')}'")

        # Validate optional field types when present
        if "source_node" in ev:
            sn = ev["source_node"]
            if not isinstance(sn, dict):
                errors.append("source_node should be dict")
            else:
                for k in ("node_type", "node_id", "node_name"):
                    if k not in sn:
                        errors.append(f"source_node missing '{k}'")

        if "target_node" in ev:
            tn = ev["target_node"]
            if not isinstance(tn, dict):
                errors.append("target_node should be dict")
            else:
                for k in ("node_type", "node_id", "node_name"):
                    if k not in tn:
                        errors.append(f"target_node missing '{k}'")

        if "payload" in ev and not isinstance(ev["payload"], dict):
            errors.append(f"payload should be dict, got {type(ev['payload']).__name__}")

        if "stack_depth" in ev and not isinstance(ev["stack_depth"], int):
            errors.append(f"stack_depth should be int, got {type(ev['stack_depth']).__name__}")

        if "parent_event_id" in ev:
            if ev["parent_event_id"] is not None and not isinstance(ev["parent_event_id"], str):
                errors.append(f"parent_event_id should be str or None")

        if errors:
            results.append((et, False, "; ".join(errors)))
            fail_count += 1
        else:
            results.append((et, True, "all checks passed"))
            pass_count += 1

    # -----------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------
    max_type_len = max(len(et) for et, _, _ in results)

    for et, passed, msg in results:
        status = "PASS" if passed else "FAIL"
        indicator = "[+]" if passed else "[X]"
        print(f"  {indicator} {et:<{max_type_len}}  {status}  {msg}")

    print()
    print("-" * 72)
    print(f"  Results: {pass_count} passed, {fail_count} failed, "
          f"{pass_count + fail_count} total")
    print("-" * 72)

    # -----------------------------------------------------------------
    # Additional cross-event validations
    # -----------------------------------------------------------------
    print()
    print("  CROSS-EVENT VALIDATIONS:")
    print()

    # 1) All event IDs are unique
    all_ids = [ev["event_id"] for ev in events]
    unique_ids = set(all_ids)
    if len(all_ids) == len(unique_ids):
        print("  [+] All event_id values are unique")
    else:
        print(f"  [X] Duplicate event_id values found ({len(all_ids) - len(unique_ids)} dupes)")
        fail_count += 1

    # 2) All timestamps are positive integers
    all_ts = [ev["timestamp_ns"] for ev in events]
    if all(isinstance(ts, int) and ts > 0 for ts in all_ts):
        print("  [+] All timestamp_ns values are positive integers")
    else:
        print("  [X] Some timestamp_ns values are invalid")
        fail_count += 1

    # 3) Timestamps are monotonically non-decreasing
    if all(all_ts[i] <= all_ts[i + 1] for i in range(len(all_ts) - 1)):
        print("  [+] Timestamps are monotonically non-decreasing")
    else:
        print("  [X] Timestamps are NOT monotonically non-decreasing")
        # This is a soft warning, not a failure (logging is lock-based, not ordered)

    # 4) Parent event IDs reference existing events
    parent_refs = [ev["parent_event_id"] for ev in events
                   if "parent_event_id" in ev and ev["parent_event_id"] is not None]
    bad_refs = [pid for pid in parent_refs if pid not in unique_ids]
    if not bad_refs:
        print(f"  [+] All parent_event_id references are valid ({len(parent_refs)} refs)")
    else:
        print(f"  [X] {len(bad_refs)} parent_event_id references point to non-existent events")
        fail_count += 1

    # 5) run_id and repo_id are consistent
    run_ids = {ev["run_id"] for ev in events}
    repo_ids = {ev["repo_id"] for ev in events}
    if len(run_ids) == 1:
        print(f"  [+] Consistent run_id across all events: {run_ids.pop()}")
    else:
        print(f"  [X] Multiple run_ids found: {run_ids}")
        fail_count += 1

    if len(repo_ids) == 1:
        print(f"  [+] Consistent repo_id across all events: {repo_ids.pop()}")
    else:
        print(f"  [X] Multiple repo_ids found: {repo_ids}")
        fail_count += 1

    print()
    print("=" * 72)
    if fail_count == 0:
        print("  ALL CHECKS PASSED")
    else:
        print(f"  {fail_count} CHECK(S) FAILED")
    print("=" * 72)

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

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
