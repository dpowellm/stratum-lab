"""Eval script for patcher unit tests (v6 graph discovery reframe).

Validation check 21: Patcher unit tests pass (all existing + new tests pass).

Tests:
  1. EventLogger basic functionality (singleton, log_event, event_id format)
  2. EventLogger v6 features:
     - push_active_node / pop_active_node / current_node / parent_node
     - record_edge_activation
     - record_error_context
     - classify_error standalone function
  3. Helper functions: make_node, generate_node_id, get_caller_info,
     hash_content, get_data_shape, capture_output_signature
  4. All patcher modules are importable
  5. Each patcher has _PATCHED idempotency flag
  6. New event types emitted correctly through the logger

Run as a standalone script:
    cd stratum-lab
    python eval/test_patchers.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup -- stratum_patcher is NOT an installed package
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "stratum_patcher"))

# ---------------------------------------------------------------------------
# Rich console setup with tee to output file
# ---------------------------------------------------------------------------
from rich.console import Console

_OUTPUT_PATH = Path(__file__).parent / "outputs" / "patcher-unit-tests.txt"
_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
_output_buffer = io.StringIO()

console = Console(record=True)

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

_pass_count = 0
_fail_count = 0
_test_results: list[tuple[str, bool, str]] = []


def _record(name: str, passed: bool, detail: str = "") -> None:
    global _pass_count, _fail_count
    if passed:
        _pass_count += 1
    else:
        _fail_count += 1
    _test_results.append((name, passed, detail))
    status = "[bold green]PASS[/bold green]" if passed else "[bold red]FAIL[/bold red]"
    msg = f"  {status}  {name}"
    if detail:
        msg += f"  --  {detail}"
    console.print(msg)


def _reset_logger() -> None:
    """Reset the EventLogger singleton so each test group starts fresh."""
    from stratum_patcher.event_logger import EventLogger
    EventLogger._instance = None


def _make_temp_events_file() -> str:
    """Create a temp file and set STRATUM_EVENTS_FILE to it."""
    fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="stratum_test_")
    os.close(fd)
    os.environ["STRATUM_EVENTS_FILE"] = path
    return path


def _read_events(path: str) -> list[dict]:
    """Read JSONL events back from file."""
    events = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _cleanup_env() -> None:
    """Remove stratum env vars."""
    for key in ("STRATUM_EVENTS_FILE", "STRATUM_RUN_ID",
                "STRATUM_REPO_ID", "STRATUM_FRAMEWORK"):
        os.environ.pop(key, None)


# ===========================================================================
# TEST GROUP 1: EventLogger basic functionality
# ===========================================================================

def test_eventlogger_basics() -> None:
    console.print("\n[bold cyan]--- 1. EventLogger Basic Functionality ---[/bold cyan]")

    _reset_logger()
    events_file = _make_temp_events_file()
    os.environ["STRATUM_RUN_ID"] = "test-run-001"
    os.environ["STRATUM_REPO_ID"] = "test-repo-001"
    os.environ["STRATUM_FRAMEWORK"] = "test-framework"

    from stratum_patcher.event_logger import EventLogger

    # 1a: Singleton pattern
    logger1 = EventLogger.get()
    logger2 = EventLogger.get()
    _record("EventLogger.get() returns singleton", logger1 is logger2)

    # 1b: run_id / repo_id / framework from env
    _record("run_id from env", logger1.run_id == "test-run-001",
            f"got {logger1.run_id!r}")
    _record("repo_id from env", logger1.repo_id == "test-repo-001",
            f"got {logger1.repo_id!r}")
    _record("framework from env", logger1.framework == "test-framework",
            f"got {logger1.framework!r}")

    # 1c: log_event returns event_id starting with "evt_"
    eid = logger1.log_event("test.basic", payload={"key": "value"})
    _record("log_event returns evt_ prefixed id", eid.startswith("evt_"),
            f"got {eid!r}")

    # 1d: Event is actually written to file
    events = _read_events(events_file)
    _record("Event written to JSONL file", len(events) == 1,
            f"found {len(events)} events")

    # 1e: Event has required fields
    if events:
        ev = events[0]
        required = {"event_id", "timestamp_ns", "run_id", "repo_id", "event_type"}
        missing = required - set(ev.keys())
        _record("Event has all required fields", len(missing) == 0,
                f"missing: {missing}" if missing else "all present")

        _record("event_id matches returned value", ev.get("event_id") == eid)
        _record("event_type is correct", ev.get("event_type") == "test.basic")
        _record("timestamp_ns is positive int",
                isinstance(ev.get("timestamp_ns"), int) and ev["timestamp_ns"] > 0)
        _record("run_id matches env", ev.get("run_id") == "test-run-001")
        _record("repo_id matches env", ev.get("repo_id") == "test-repo-001")
        _record("payload preserved", ev.get("payload") == {"key": "value"})

    # 1f: source_node and target_node round-trip
    from stratum_patcher.event_logger import make_node
    src = make_node("agent", "test:Agent:main.py:1", "TestAgent")
    tgt = make_node("capability", "test:Tool:main.py:10", "WebSearch")
    eid2 = logger1.log_event(
        "tool.invoked",
        source_node=src,
        target_node=tgt,
        edge_type="calls",
        parent_event_id=eid,
        stack_depth=1,
    )
    events2 = _read_events(events_file)
    ev2 = events2[-1]
    _record("source_node round-trips", ev2.get("source_node") == src)
    _record("target_node round-trips", ev2.get("target_node") == tgt)
    _record("edge_type round-trips", ev2.get("edge_type") == "calls")
    _record("parent_event_id round-trips", ev2.get("parent_event_id") == eid)
    _record("stack_depth round-trips", ev2.get("stack_depth") == 1)

    # 1g: Multiple events get unique IDs
    ids = {logger1.log_event("test.unique") for _ in range(20)}
    _record("20 events have 20 unique IDs", len(ids) == 20,
            f"got {len(ids)} unique")

    # Cleanup
    _reset_logger()
    _cleanup_env()
    try:
        os.unlink(events_file)
    except OSError:
        pass


# ===========================================================================
# TEST GROUP 2: EventLogger v6 features
# ===========================================================================

def test_eventlogger_v6_features() -> None:
    console.print("\n[bold cyan]--- 2. EventLogger v6 Features ---[/bold cyan]")

    _reset_logger()
    events_file = _make_temp_events_file()
    os.environ["STRATUM_RUN_ID"] = "v6-test-run"
    os.environ["STRATUM_REPO_ID"] = "v6-test-repo"
    os.environ["STRATUM_FRAMEWORK"] = "crewai"

    from stratum_patcher.event_logger import EventLogger, classify_error

    logger = EventLogger.get()

    # ---- 2a: Active node stack management ----
    console.print("  [dim]Active node stack management[/dim]")

    # Empty stack
    _record("current_node empty on fresh logger", logger.current_node() == "",
            f"got {logger.current_node()!r}")
    _record("parent_node empty on fresh logger", logger.parent_node() == "",
            f"got {logger.parent_node()!r}")

    # Push one node
    logger.push_active_node("crewai:Researcher:agents.py:10")
    _record("current_node after push",
            logger.current_node() == "crewai:Researcher:agents.py:10")
    _record("parent_node with one node is empty",
            logger.parent_node() == "")

    # Push second node
    logger.push_active_node("crewai:Writer:agents.py:30")
    _record("current_node after second push",
            logger.current_node() == "crewai:Writer:agents.py:30")
    _record("parent_node with two nodes",
            logger.parent_node() == "crewai:Researcher:agents.py:10")

    # Push third node
    logger.push_active_node("crewai:Reviewer:agents.py:50")
    _record("current_node after third push",
            logger.current_node() == "crewai:Reviewer:agents.py:50")
    _record("parent_node with three nodes",
            logger.parent_node() == "crewai:Writer:agents.py:30")

    # Pop returns the popped node
    popped = logger.pop_active_node()
    _record("pop_active_node returns popped node",
            popped == "crewai:Reviewer:agents.py:50",
            f"got {popped!r}")
    _record("current_node after pop",
            logger.current_node() == "crewai:Writer:agents.py:30")

    # Pop remaining
    popped2 = logger.pop_active_node()
    _record("second pop returns correct node",
            popped2 == "crewai:Writer:agents.py:30")
    popped3 = logger.pop_active_node()
    _record("third pop returns correct node",
            popped3 == "crewai:Researcher:agents.py:10")

    # Pop on empty stack
    popped_empty = logger.pop_active_node()
    _record("pop on empty stack returns empty string",
            popped_empty == "",
            f"got {popped_empty!r}")

    # ---- 2b: record_edge_activation ----
    console.print("  [dim]Edge activation recording[/dim]")

    logger.record_edge_activation(
        source="crewai:Researcher:agents.py:10",
        target="crewai:Writer:agents.py:30",
        data_hash="abc123",
    )
    logger.record_edge_activation(
        source="crewai:Writer:agents.py:30",
        target="crewai:Reviewer:agents.py:50",
    )

    edges = logger._edge_activations
    _record("edge_activations has 2 entries", len(edges) == 2,
            f"got {len(edges)}")

    if len(edges) >= 2:
        e0 = edges[0]
        _record("edge[0] source correct",
                e0["source"] == "crewai:Researcher:agents.py:10")
        _record("edge[0] target correct",
                e0["target"] == "crewai:Writer:agents.py:30")
        _record("edge[0] data_hash correct", e0["data_hash"] == "abc123")
        _record("edge[0] has timestamp", isinstance(e0["timestamp"], float))
        _record("edge[0] has run_id", e0["run_id"] == "v6-test-run")

        e1 = edges[1]
        _record("edge[1] data_hash defaults to empty",
                e1["data_hash"] == "",
                f"got {e1['data_hash']!r}")

    # ---- 2c: record_error_context ----
    console.print("  [dim]Error context recording[/dim]")

    # Push some nodes so the stack is captured
    logger.push_active_node("crewai:Researcher:agents.py:10")
    logger.push_active_node("crewai:Writer:agents.py:30")

    logger.record_error_context(
        node_id="crewai:Writer:agents.py:30",
        error_type="ValueError",
        error_msg="invalid JSON in delegation payload",
        upstream_node="crewai:Researcher:agents.py:10",
        upstream_output_hash="def456",
    )

    errs = logger._error_context_stack
    _record("error_context_stack has 1 entry", len(errs) == 1,
            f"got {len(errs)}")

    if errs:
        ec = errs[0]
        _record("error_context node_id correct",
                ec["node_id"] == "crewai:Writer:agents.py:30")
        _record("error_context error_type correct",
                ec["error_type"] == "ValueError")
        _record("error_context error_msg truncated to 500",
                len(ec["error_msg"]) <= 500)
        _record("error_context upstream_node correct",
                ec["upstream_node"] == "crewai:Researcher:agents.py:10")
        _record("error_context upstream_output_hash correct",
                ec["upstream_output_hash"] == "def456")
        _record("error_context has timestamp",
                isinstance(ec["timestamp"], float))
        _record("error_context active_stack captured",
                ec["active_stack"] == [
                    "crewai:Researcher:agents.py:10",
                    "crewai:Writer:agents.py:30",
                ])

    # Test error_msg truncation
    long_msg = "x" * 1000
    logger.record_error_context(
        node_id="test", error_type="RuntimeError", error_msg=long_msg,
    )
    ec2 = logger._error_context_stack[-1]
    _record("error_msg truncated at 500 chars",
            len(ec2["error_msg"]) == 500,
            f"got length {len(ec2['error_msg'])}")

    # Clean up stack
    logger.pop_active_node()
    logger.pop_active_node()

    # ---- 2d: classify_error standalone function ----
    console.print("  [dim]Error classification[/dim]")

    _record("classify timeout error",
            classify_error(TimeoutError("connection timed out")) == "timeout")
    _record("classify missing key error",
            classify_error(KeyError("key not found")) == "schema_mismatch")
    _record("classify json decode error",
            classify_error(ValueError("json decode error")) == "schema_mismatch")
    _record("classify type expected error",
            classify_error(TypeError("type int expected")) == "schema_mismatch")
    _record("classify rate limit error",
            classify_error(Exception("rate limit exceeded")) == "rate_limit")
    _record("classify api error",
            classify_error(Exception("api connection refused")) == "api_error")
    _record("classify http error",
            classify_error(Exception("http 500 server error")) == "api_error")
    _record("classify connection error",
            classify_error(Exception("connection reset by peer")) == "api_error")
    _record("classify permission error",
            classify_error(Exception("permission denied")) == "permission_error")
    _record("classify access error",
            classify_error(Exception("access forbidden")) == "permission_error")
    _record("classify KeyError type",
            classify_error(KeyError("something")) == "schema_mismatch")
    _record("classify TypeError type",
            classify_error(TypeError("bad type")) == "schema_mismatch")
    _record("classify ValueError type",
            classify_error(ValueError("bad value")) == "schema_mismatch")
    _record("classify AttributeError type",
            classify_error(AttributeError("no attr")) == "schema_mismatch")
    _record("classify generic runtime error",
            classify_error(RuntimeError("something broke")) == "runtime_error")
    _record("classify unknown exception fallback",
            classify_error(Exception("totally unknown")) == "runtime_error")

    # Cleanup
    _reset_logger()
    _cleanup_env()
    try:
        os.unlink(events_file)
    except OSError:
        pass


# ===========================================================================
# TEST GROUP 3: Helper functions
# ===========================================================================

def test_helper_functions() -> None:
    console.print("\n[bold cyan]--- 3. Helper Functions ---[/bold cyan]")

    from stratum_patcher.event_logger import (
        make_node,
        generate_node_id,
        get_caller_info,
        hash_content,
        get_data_shape,
        capture_output_signature,
    )

    # ---- 3a: make_node ----
    console.print("  [dim]make_node[/dim]")
    node = make_node("agent", "crewai:Agent:main.py:1", "TestAgent")
    _record("make_node returns dict", isinstance(node, dict))
    _record("make_node has node_type", node.get("node_type") == "agent")
    _record("make_node has node_id", node.get("node_id") == "crewai:Agent:main.py:1")
    _record("make_node has node_name", node.get("node_name") == "TestAgent")
    _record("make_node has exactly 3 keys", len(node) == 3)

    # ---- 3b: generate_node_id ----
    console.print("  [dim]generate_node_id[/dim]")
    nid = generate_node_id("crewai", "Researcher", "agents.py", 10)
    _record("generate_node_id format",
            nid == "crewai:Researcher:agents.py:10",
            f"got {nid!r}")

    # With string line number
    nid2 = generate_node_id("langchain", "Chain", "chain.py", "42")
    _record("generate_node_id with str line",
            nid2 == "langchain:Chain:chain.py:42",
            f"got {nid2!r}")

    # ---- 3c: get_caller_info ----
    console.print("  [dim]get_caller_info[/dim]")
    filename, lineno, funcname = get_caller_info(skip_frames=1)
    _record("get_caller_info returns 3-tuple",
            isinstance(filename, str) and isinstance(lineno, int) and isinstance(funcname, str))
    _record("get_caller_info filename not unknown",
            filename != "<unknown>",
            f"got {filename!r}")
    _record("get_caller_info lineno positive", lineno > 0, f"got {lineno}")
    _record("get_caller_info funcname is this function",
            funcname == "test_helper_functions",
            f"got {funcname!r}")

    # Deep skip returns unknown gracefully
    fn_deep, ln_deep, func_deep = get_caller_info(skip_frames=999)
    _record("get_caller_info deep skip returns unknown",
            fn_deep == "<unknown>" and ln_deep == 0 and func_deep == "<unknown>")

    # ---- 3d: hash_content ----
    console.print("  [dim]hash_content[/dim]")
    h1 = hash_content("hello world")
    _record("hash_content returns hex string",
            isinstance(h1, str) and len(h1) == 64,
            f"len={len(h1)}")
    h2 = hash_content("hello world")
    _record("hash_content is deterministic", h1 == h2)
    h3 = hash_content("different content")
    _record("hash_content differs for different input", h1 != h3)

    # Dict input
    h4 = hash_content({"key": "value"})
    _record("hash_content handles dict", isinstance(h4, str) and len(h4) == 64)

    # None input
    h5 = hash_content(None)
    _record("hash_content handles None", isinstance(h5, str) and len(h5) == 64)

    # ---- 3e: get_data_shape ----
    console.print("  [dim]get_data_shape[/dim]")
    _record("get_data_shape None", get_data_shape(None) == "None")
    _record("get_data_shape str", get_data_shape("hello") == "str(len=5)",
            f"got {get_data_shape('hello')!r}")
    _record("get_data_shape bytes", get_data_shape(b"abc") == "bytes(len=3)",
            f"got {get_data_shape(b'abc')!r}")
    _record("get_data_shape int", get_data_shape(42) == "int")
    _record("get_data_shape float", get_data_shape(3.14) == "float")
    _record("get_data_shape bool", get_data_shape(True) == "bool")

    dict_shape = get_data_shape({"a": 1, "b": 2})
    _record("get_data_shape dict",
            "dict" in dict_shape and "len=2" in dict_shape,
            f"got {dict_shape!r}")

    list_shape = get_data_shape([1, 2, 3])
    _record("get_data_shape list",
            "list" in list_shape and "len=3" in list_shape,
            f"got {list_shape!r}")

    tuple_shape = get_data_shape((1, 2))
    _record("get_data_shape tuple",
            "tuple" in tuple_shape and "len=2" in tuple_shape,
            f"got {tuple_shape!r}")

    # Custom object falls back to qualname (includes enclosing scope)
    class MyWidget:
        pass
    widget_shape = get_data_shape(MyWidget())
    _record("get_data_shape custom class",
            "MyWidget" in widget_shape,
            f"got {widget_shape!r}")

    # Large dict (>10 keys) truncates keys display
    big_dict = {f"key_{i}": i for i in range(15)}
    big_shape = get_data_shape(big_dict)
    _record("get_data_shape big dict has ellipsis",
            "..." in big_shape and "len=15" in big_shape,
            f"got {big_shape!r}")

    # Empty list
    empty_shape = get_data_shape([])
    _record("get_data_shape empty list",
            "list" in empty_shape and "len=0" in empty_shape and "empty" in empty_shape,
            f"got {empty_shape!r}")

    # ---- 3f: capture_output_signature ----
    console.print("  [dim]capture_output_signature[/dim]")

    # None input
    sig_none = capture_output_signature(None)
    _record("capture_output_signature None type",
            sig_none["type"] == "null")
    _record("capture_output_signature None hash is None",
            sig_none["hash"] is None)
    _record("capture_output_signature None size_bytes 0",
            sig_none["size_bytes"] == 0)

    # String input (short)
    sig_short = capture_output_signature("hello world")
    _record("capture_output_signature short string type",
            sig_short["type"] == "short_text",
            f"got {sig_short['type']!r}")
    _record("capture_output_signature has hash",
            isinstance(sig_short["hash"], str) and len(sig_short["hash"]) == 64)
    _record("capture_output_signature has size_bytes",
            sig_short["size_bytes"] > 0)
    _record("capture_output_signature has preview",
            sig_short["preview"] == "hello world")

    # Long string
    long_str = "a" * 600
    sig_long = capture_output_signature(long_str)
    _record("capture_output_signature long string type",
            sig_long["type"] == "long_text",
            f"got {sig_long['type']!r}")
    _record("capture_output_signature preview truncated to 200",
            len(sig_long["preview"]) == 200)

    # Classification dict
    sig_class = capture_output_signature({"label": "positive", "confidence": 0.95})
    _record("capture_output_signature classification type",
            sig_class["type"] == "classification",
            f"got {sig_class['type']!r}")
    _record("capture_output_signature classification_fields extracted",
            sig_class["classification_fields"] is not None)
    if sig_class["classification_fields"]:
        _record("capture_output_signature classification field value",
                sig_class["classification_fields"].get("label") == "positive")

    # Routing decision dict
    sig_route = capture_output_signature({"action": "delegate", "next_step": "review"})
    _record("capture_output_signature routing type",
            sig_route["type"] == "routing_decision",
            f"got {sig_route['type']!r}")

    # Scored output dict
    sig_scored = capture_output_signature({"score": 0.8, "explanation": "good"})
    _record("capture_output_signature scored type",
            sig_scored["type"] == "scored_output",
            f"got {sig_scored['type']!r}")

    # Structured JSON (no classification keys)
    sig_struct = capture_output_signature({"name": "Alice", "age": 30})
    _record("capture_output_signature structured_json type",
            sig_struct["type"] == "structured_json",
            f"got {sig_struct['type']!r}")
    _record("capture_output_signature structure has key types",
            sig_struct["structure"] is not None and sig_struct["structure"].get("name") == "str")

    # Determinism: same input -> same hash
    sig_a = capture_output_signature({"key": "value"})
    sig_b = capture_output_signature({"key": "value"})
    _record("capture_output_signature deterministic hash",
            sig_a["hash"] == sig_b["hash"])

    # Different input -> different hash
    sig_c = capture_output_signature({"key": "other"})
    _record("capture_output_signature different input different hash",
            sig_a["hash"] != sig_c["hash"])


# ===========================================================================
# TEST GROUP 4: Patcher module importability
# ===========================================================================

def test_patcher_imports() -> None:
    console.print("\n[bold cyan]--- 4. Patcher Module Importability ---[/bold cyan]")

    patcher_modules = [
        "stratum_patcher.event_logger",
        "stratum_patcher.crewai_patch",
        "stratum_patcher.openai_patch",
        "stratum_patcher.anthropic_patch",
        "stratum_patcher.langchain_patch",
        "stratum_patcher.langgraph_patch",
        "stratum_patcher.autogen_patch",
        "stratum_patcher.generic_patch",
    ]

    for mod_name in patcher_modules:
        try:
            __import__(mod_name)
            _record(f"import {mod_name}", True)
        except Exception as exc:
            _record(f"import {mod_name}", False, f"{type(exc).__name__}: {exc}")


# ===========================================================================
# TEST GROUP 5: _PATCHED idempotency flags
# ===========================================================================

def test_patched_flags() -> None:
    console.print("\n[bold cyan]--- 5. _PATCHED Idempotency Flags ---[/bold cyan]")

    patcher_modules = {
        "crewai_patch": "stratum_patcher.crewai_patch",
        "openai_patch": "stratum_patcher.openai_patch",
        "anthropic_patch": "stratum_patcher.anthropic_patch",
        "langchain_patch": "stratum_patcher.langchain_patch",
        "langgraph_patch": "stratum_patcher.langgraph_patch",
        "autogen_patch": "stratum_patcher.autogen_patch",
        "generic_patch": "stratum_patcher.generic_patch",
    }

    for short_name, full_name in patcher_modules.items():
        try:
            mod = __import__(full_name, fromlist=[short_name])
            has_flag = hasattr(mod, "_PATCHED")
            _record(f"{short_name} has _PATCHED flag", has_flag)
            if has_flag:
                flag_val = getattr(mod, "_PATCHED")
                _record(f"{short_name}._PATCHED is bool",
                        isinstance(flag_val, bool),
                        f"type={type(flag_val).__name__}, value={flag_val}")
        except Exception as exc:
            _record(f"{short_name} _PATCHED check", False,
                    f"import failed: {type(exc).__name__}: {exc}")


# ===========================================================================
# TEST GROUP 6: New event types emitted through the logger
# ===========================================================================

def test_event_types_emitted() -> None:
    console.print("\n[bold cyan]--- 6. Event Types Emitted Through Logger ---[/bold cyan]")

    _reset_logger()
    events_file = _make_temp_events_file()
    os.environ["STRATUM_RUN_ID"] = "event-type-test"
    os.environ["STRATUM_REPO_ID"] = "event-type-repo"
    os.environ["STRATUM_FRAMEWORK"] = "crewai"

    from stratum_patcher.event_logger import EventLogger, make_node, generate_node_id

    logger = EventLogger.get()

    src_agent = make_node("agent", generate_node_id("crewai", "Researcher", "agents.py", 10), "Researcher")
    tgt_agent = make_node("agent", generate_node_id("crewai", "Writer", "agents.py", 30), "Writer")
    src_tool = make_node("capability", generate_node_id("crewai", "WebSearch", "tools.py", 5), "WebSearch")
    src_llm = make_node("capability", generate_node_id("crewai", "LLM", "llm.py", 1), "openai.chat")
    src_data = make_node("data_store", "ds:shared_mem:0", "SharedMemory")
    src_ext = make_node("external", "ext:api:0", "api.example.com")
    src_guard = make_node("capability", "guard:filter:0", "ContentFilter")

    # All 19 core event types
    event_specs = [
        ("execution.start", dict(source_node=src_agent, payload={"crew_name": "Test"})),
        ("execution.end", dict(source_node=src_agent, payload={"latency_ms": 100, "status": "success"})),
        ("agent.task_start", dict(source_node=src_agent, payload={"agent_role": "Researcher"})),
        ("agent.task_end", dict(source_node=src_agent, payload={"status": "success", "output_hash": "abc"})),
        ("delegation.initiated", dict(source_node=src_agent, target_node=tgt_agent, edge_type="delegates_to",
                                      payload={"delegator": "Researcher", "delegate": "Writer"})),
        ("delegation.completed", dict(source_node=src_agent, target_node=tgt_agent, edge_type="delegates_to",
                                      payload={"status": "success"})),
        ("tool.invoked", dict(source_node=src_agent, target_node=src_tool, edge_type="calls",
                              payload={"tool_name": "WebSearch"})),
        ("tool.completed", dict(source_node=src_agent, target_node=src_tool, edge_type="calls",
                                payload={"tool_name": "WebSearch", "status": "success"})),
        ("tool.call_failure", dict(source_node=src_tool,
                                   payload={"tool_name": "WebSearch", "reason": "bad_args"})),
        ("llm.call_start", dict(source_node=src_llm,
                                payload={"model_requested": "gpt-4", "message_count": 3})),
        ("llm.call_end", dict(source_node=src_llm,
                              payload={"model_requested": "gpt-4", "latency_ms": 200,
                                       "output_hash": "xyz", "output_type": "short_text"})),
        ("data.read", dict(source_node=src_data, payload={"channel": "SharedMemory"})),
        ("data.write", dict(source_node=src_data, payload={"channel": "SharedMemory"})),
        ("error.occurred", dict(source_node=src_agent,
                                payload={"error_type": "ValueError", "error_message": "bad input"})),
        ("error.propagated", dict(source_node=src_agent, target_node=tgt_agent,
                                  edge_type="propagates_error",
                                  payload={"error_type": "ValueError"})),
        ("error.cascade", dict(source_node=src_agent,
                               payload={"cascade_depth": 2, "affected_agents": ["A", "B"]})),
        ("decision.made", dict(source_node=src_agent,
                               payload={"decision_type": "routing", "selected": "plan_a"})),
        ("guardrail.triggered", dict(source_node=src_guard,
                                     payload={"guardrail_name": "ContentFilter", "action_taken": "blocked"})),
        ("external.call", dict(source_node=src_ext,
                               payload={"method": "GET", "domain": "api.example.com", "status_code": 200})),
    ]

    emitted_ids: dict[str, str] = {}
    for event_type, kwargs in event_specs:
        eid = logger.log_event(event_type, **kwargs)
        emitted_ids[event_type] = eid

    # Read back all events
    events = _read_events(events_file)
    event_by_type: dict[str, dict] = {}
    for ev in events:
        et = ev.get("event_type", "")
        if et not in event_by_type:
            event_by_type[et] = ev

    _record("All 19 event types emitted",
            len(events) == 19,
            f"got {len(events)} events")

    # Check each event type is present and well-formed
    for event_type, _ in event_specs:
        present = event_type in event_by_type
        _record(f"event type {event_type} present", present)
        if present:
            ev = event_by_type[event_type]
            # Verify event_id matches what was returned
            _record(f"event type {event_type} has correct event_id",
                    ev["event_id"] == emitted_ids[event_type])

    # Verify all events share consistent run_id and repo_id
    run_ids = {ev["run_id"] for ev in events}
    repo_ids = {ev["repo_id"] for ev in events}
    _record("Consistent run_id across all events", len(run_ids) == 1,
            f"found {run_ids}")
    _record("Consistent repo_id across all events", len(repo_ids) == 1,
            f"found {repo_ids}")

    # Verify all event_ids are unique
    all_ids = [ev["event_id"] for ev in events]
    _record("All event_ids unique", len(set(all_ids)) == len(all_ids),
            f"{len(all_ids)} total, {len(set(all_ids))} unique")

    # Verify timestamps are non-decreasing
    timestamps = [ev["timestamp_ns"] for ev in events]
    monotonic = all(timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1))
    _record("Timestamps non-decreasing", monotonic)

    # Verify edge_type present where specified
    delegation_ev = event_by_type.get("delegation.initiated", {})
    _record("delegation.initiated has edge_type",
            delegation_ev.get("edge_type") == "delegates_to")

    tool_ev = event_by_type.get("tool.invoked", {})
    _record("tool.invoked has edge_type",
            tool_ev.get("edge_type") == "calls")

    # Verify v6 semantic fields on relevant event types
    task_end_ev = event_by_type.get("agent.task_end", {})
    task_payload = task_end_ev.get("payload", {})
    _record("agent.task_end payload has output_hash",
            "output_hash" in task_payload,
            f"keys: {list(task_payload.keys())}")

    llm_end_ev = event_by_type.get("llm.call_end", {})
    llm_payload = llm_end_ev.get("payload", {})
    _record("llm.call_end payload has output_hash",
            "output_hash" in llm_payload)
    _record("llm.call_end payload has output_type",
            "output_type" in llm_payload)

    # Cleanup
    _reset_logger()
    _cleanup_env()
    try:
        os.unlink(events_file)
    except OSError:
        pass


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    console.print("[bold white on blue] STRATUM PATCHER UNIT TESTS (v6) [/bold white on blue]")
    console.print(f"Validation check 21: Patcher unit tests pass")
    console.print(f"Output: {_OUTPUT_PATH}\n")

    test_eventlogger_basics()
    test_eventlogger_v6_features()
    test_helper_functions()
    test_patcher_imports()
    test_patched_flags()
    test_event_types_emitted()

    # ---- Summary ----
    console.print("\n" + "=" * 72)
    console.print(f"[bold]  RESULTS: {_pass_count} passed, {_fail_count} failed, "
                  f"{_pass_count + _fail_count} total[/bold]")
    console.print("=" * 72)

    if _fail_count > 0:
        console.print("\n[bold red]  FAILED TESTS:[/bold red]")
        for name, passed, detail in _test_results:
            if not passed:
                console.print(f"    [red]FAIL[/red]  {name}  {detail}")

    if _fail_count == 0:
        console.print("[bold green]  ALL CHECKS PASSED[/bold green]")
    else:
        console.print(f"\n[bold red]  {_fail_count} CHECK(S) FAILED[/bold red]")

    # ---- Write output file ----
    output_text = console.export_text()
    _OUTPUT_PATH.write_text(output_text, encoding="utf-8")
    console.print(f"\nOutput written to {_OUTPUT_PATH}")

    sys.exit(0 if _fail_count == 0 else 1)


if __name__ == "__main__":
    main()
