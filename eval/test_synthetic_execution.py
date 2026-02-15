"""Synthetic execution simulation for stratum-patcher event generation.

Simulates a mock agent system WITHOUT Docker/vLLM by using the EventLogger
directly.  Creates:
  - 3 agents: Researcher, Writer, Reviewer
  - 2 tools: WebSearch, FileWriter
  - 1 data store: SharedMemory
  - 1 delegation chain: Researcher -> Writer -> Reviewer

Emits realistic events for 3 runs with different inputs, collects JSONL
files, and prints analysis.

Run as a standalone script:
    cd stratum-lab
    python eval/test_synthetic_execution.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure stratum_patcher is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "stratum_patcher"))


def _reset_logger():
    from stratum_patcher.event_logger import EventLogger
    EventLogger._instance = None


def _read_events(path: str) -> list[dict]:
    events = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def simulate_run(
    run_id: str,
    repo_id: str,
    events_file: str,
    input_query: str,
) -> tuple[list[dict], dict[str, str]]:
    """Simulate a full agent execution run and return the events list.

    The simulation follows this execution flow:
      1. execution.start  (crew kickoff)
      2. agent.task_start (Researcher begins)
      3. llm.call_start   (Researcher calls LLM)
      4. llm.call_end     (LLM responds)
      5. tool.invoked      (Researcher uses WebSearch)
      6. external.call     (WebSearch makes HTTP request)
      7. tool.completed    (WebSearch returns)
      8. data.write        (Researcher writes to SharedMemory)
      9. agent.task_end    (Researcher finishes)
     10. delegation.initiated (Researcher -> Writer)
     11. agent.task_start  (Writer begins)
     12. data.read         (Writer reads SharedMemory)
     13. llm.call_start    (Writer calls LLM)
     14. llm.call_end      (LLM responds)
     15. tool.invoked      (Writer uses FileWriter)
     16. file.write        (FileWriter writes output)
     17. tool.completed    (FileWriter returns)
     18. agent.task_end    (Writer finishes)
     19. delegation.completed (Researcher -> Writer)
     20. delegation.initiated (Writer -> Reviewer)
     21. agent.task_start  (Reviewer begins)
     22. data.read         (Reviewer reads SharedMemory)
     23. llm.call_start    (Reviewer calls LLM)
     24. decision.made     (Reviewer decides: approve/revise)
     25. llm.call_end      (LLM responds)
     26. guardrail.triggered (Content quality check)
     27. agent.task_end    (Reviewer finishes)
     28. delegation.completed (Writer -> Reviewer)
     29. execution.end     (crew finishes)
    """
    _reset_logger()
    os.environ["STRATUM_EVENTS_FILE"] = events_file
    os.environ["STRATUM_RUN_ID"] = run_id
    os.environ["STRATUM_REPO_ID"] = repo_id
    os.environ["STRATUM_FRAMEWORK"] = "crewai"

    from stratum_patcher.event_logger import EventLogger, make_node, generate_node_id

    logger = EventLogger.get()

    # -----------------------------------------------------------------
    # Define nodes (deterministic node IDs)
    # -----------------------------------------------------------------
    researcher_nid = generate_node_id("crewai", "Researcher", "agents.py", 10)
    writer_nid = generate_node_id("crewai", "Writer", "agents.py", 30)
    reviewer_nid = generate_node_id("crewai", "Reviewer", "agents.py", 50)
    websearch_nid = generate_node_id("crewai", "WebSearch", "tools.py", 5)
    filewriter_nid = generate_node_id("crewai", "FileWriter", "tools.py", 25)
    memory_nid = generate_node_id("crewai", "SharedMemory", "memory.py", 1)
    crew_nid = generate_node_id("crewai", "ResearchCrew", "crew.py", 1)
    llm_nid = generate_node_id("openai", "ChatCompletion", "llm.py", 1)
    http_nid = generate_node_id("generic", "http:api.search.com", "external.py", 1)
    guardrail_nid = generate_node_id("crewai", "QualityGuard", "guardrails.py", 1)

    crew_node = make_node("agent", crew_nid, "ResearchCrew")
    researcher_node = make_node("agent", researcher_nid, "Researcher")
    writer_node = make_node("agent", writer_nid, "Writer")
    reviewer_node = make_node("agent", reviewer_nid, "Reviewer")
    websearch_node = make_node("capability", websearch_nid, "WebSearch")
    filewriter_node = make_node("capability", filewriter_nid, "FileWriter")
    memory_node = make_node("data_store", memory_nid, "SharedMemory")
    llm_node = make_node("capability", llm_nid, "openai.chat.completions.create")
    http_node = make_node("external", http_nid, "api.search.com")
    guardrail_node = make_node("capability", guardrail_nid, "QualityGuard")

    # Store all generated node IDs for later verification
    node_ids = {
        "Researcher": researcher_nid,
        "Writer": writer_nid,
        "Reviewer": reviewer_nid,
        "WebSearch": websearch_nid,
        "FileWriter": filewriter_nid,
        "SharedMemory": memory_nid,
        "ResearchCrew": crew_nid,
        "LLM": llm_nid,
        "HTTP": http_nid,
        "QualityGuard": guardrail_nid,
    }

    # Small delay helper to ensure distinct timestamps
    def tick():
        time.sleep(0.001)

    # -----------------------------------------------------------------
    # 1. execution.start
    # -----------------------------------------------------------------
    exec_start_id = logger.log_event(
        "execution.start",
        source_node=crew_node,
        payload={
            "crew_name": "ResearchCrew",
            "agent_count": 3,
            "task_count": 3,
            "agent_roles": ["Researcher", "Writer", "Reviewer"],
            "input_query": f"hash:{hash(input_query) & 0xFFFFFFFF:08x}",
        },
    )
    tick()

    # -----------------------------------------------------------------
    # 2. agent.task_start (Researcher)
    # -----------------------------------------------------------------
    researcher_task_id = logger.log_event(
        "agent.task_start",
        source_node=researcher_node,
        payload={
            "agent_role": "Researcher",
            "task_name": "research_topic",
            "tools_available": ["WebSearch"],
        },
        parent_event_id=exec_start_id,
    )
    tick()

    # -----------------------------------------------------------------
    # 3-4. llm.call_start / llm.call_end (Researcher thinks)
    # -----------------------------------------------------------------
    llm_start_1 = logger.log_event(
        "llm.call_start",
        source_node=llm_node,
        payload={
            "model_requested": "Qwen/Qwen2.5-72B-Instruct",
            "message_count": 3,
            "has_tools": True,
        },
        parent_event_id=researcher_task_id,
    )
    tick()
    logger.log_event(
        "llm.call_end",
        source_node=llm_node,
        payload={
            "model_requested": "Qwen/Qwen2.5-72B-Instruct",
            "model_actual": "Qwen/Qwen2.5-72B-Instruct",
            "latency_ms": 234.5,
            "input_tokens": 450,
            "output_tokens": 180,
            "finish_reason": "stop",
        },
        parent_event_id=llm_start_1,
    )
    tick()

    # -----------------------------------------------------------------
    # 5. tool.invoked (WebSearch)
    # -----------------------------------------------------------------
    tool_start_1 = logger.log_event(
        "tool.invoked",
        source_node=researcher_node,
        target_node=websearch_node,
        edge_type="calls",
        payload={
            "tool_name": "WebSearch",
            "args_shape": "dict(keys=['query'], len=1)",
        },
        parent_event_id=researcher_task_id,
    )
    tick()

    # -----------------------------------------------------------------
    # 6. external.call (HTTP request)
    # -----------------------------------------------------------------
    logger.log_event(
        "external.call",
        source_node=http_node,
        payload={
            "method": "GET",
            "domain": "api.search.com",
            "latency_ms": 156.3,
            "status_code": 200,
            "phase": "end",
        },
        parent_event_id=tool_start_1,
    )
    tick()

    # -----------------------------------------------------------------
    # 7. tool.completed (WebSearch)
    # -----------------------------------------------------------------
    logger.log_event(
        "tool.completed",
        source_node=researcher_node,
        target_node=websearch_node,
        edge_type="calls",
        payload={
            "tool_name": "WebSearch",
            "latency_ms": 189.7,
            "status": "success",
            "result_shape": "dict(keys=['results', 'count'], len=2)",
        },
        parent_event_id=tool_start_1,
    )
    tick()

    # -----------------------------------------------------------------
    # 8. data.write (Researcher writes to SharedMemory)
    # -----------------------------------------------------------------
    logger.log_event(
        "data.write",
        source_node=memory_node,
        payload={
            "channel": "SharedMemory",
            "data_shape": "dict(keys=['research_findings'], len=1)",
            "writer": "Researcher",
        },
        parent_event_id=researcher_task_id,
    )
    tick()

    # -----------------------------------------------------------------
    # 9. agent.task_end (Researcher finishes)
    # -----------------------------------------------------------------
    logger.log_event(
        "agent.task_end",
        source_node=researcher_node,
        payload={
            "agent_role": "Researcher",
            "latency_ms": 890.2,
            "status": "success",
            "result_shape": "str(len=1500)",
        },
        parent_event_id=researcher_task_id,
    )
    tick()

    # -----------------------------------------------------------------
    # 10. delegation.initiated (Researcher -> Writer)
    # -----------------------------------------------------------------
    deleg_1_id = logger.log_event(
        "delegation.initiated",
        source_node=researcher_node,
        target_node=writer_node,
        edge_type="delegates_to",
        payload={
            "delegator": "Researcher",
            "delegate": "Writer",
            "task_hash": f"hash:{hash('write_article') & 0xFFFFFFFF:08x}",
        },
        parent_event_id=exec_start_id,
    )
    tick()

    # -----------------------------------------------------------------
    # 11. agent.task_start (Writer)
    # -----------------------------------------------------------------
    writer_task_id = logger.log_event(
        "agent.task_start",
        source_node=writer_node,
        payload={
            "agent_role": "Writer",
            "task_name": "write_article",
            "tools_available": ["FileWriter"],
        },
        parent_event_id=deleg_1_id,
    )
    tick()

    # -----------------------------------------------------------------
    # 12. data.read (Writer reads SharedMemory)
    # -----------------------------------------------------------------
    logger.log_event(
        "data.read",
        source_node=memory_node,
        payload={
            "channel": "SharedMemory",
            "data_shape": "dict(keys=['research_findings'], len=1)",
            "reader": "Writer",
        },
        parent_event_id=writer_task_id,
    )
    tick()

    # -----------------------------------------------------------------
    # 13-14. llm.call_start / llm.call_end (Writer thinks)
    # -----------------------------------------------------------------
    llm_start_2 = logger.log_event(
        "llm.call_start",
        source_node=llm_node,
        payload={
            "model_requested": "Qwen/Qwen2.5-72B-Instruct",
            "message_count": 5,
            "has_tools": True,
        },
        parent_event_id=writer_task_id,
    )
    tick()
    logger.log_event(
        "llm.call_end",
        source_node=llm_node,
        payload={
            "model_requested": "Qwen/Qwen2.5-72B-Instruct",
            "model_actual": "Qwen/Qwen2.5-72B-Instruct",
            "latency_ms": 567.8,
            "input_tokens": 1200,
            "output_tokens": 800,
            "finish_reason": "stop",
        },
        parent_event_id=llm_start_2,
    )
    tick()

    # -----------------------------------------------------------------
    # 15. tool.invoked (FileWriter)
    # -----------------------------------------------------------------
    tool_start_2 = logger.log_event(
        "tool.invoked",
        source_node=writer_node,
        target_node=filewriter_node,
        edge_type="calls",
        payload={
            "tool_name": "FileWriter",
            "args_shape": "dict(keys=['content', 'filename'], len=2)",
        },
        parent_event_id=writer_task_id,
    )
    tick()

    # -----------------------------------------------------------------
    # 16. file.write (FileWriter writes output)
    # -----------------------------------------------------------------
    logger.log_event(
        "file.write",
        source_node=make_node("data_store",
                              generate_node_id("generic", "file_io", "/app/output/article.md", 0),
                              "/app/output/article.md"),
        payload={
            "path": "/app/output/article.md",
            "mode": "w",
        },
        parent_event_id=tool_start_2,
    )
    tick()

    # -----------------------------------------------------------------
    # 17. tool.completed (FileWriter)
    # -----------------------------------------------------------------
    logger.log_event(
        "tool.completed",
        source_node=writer_node,
        target_node=filewriter_node,
        edge_type="calls",
        payload={
            "tool_name": "FileWriter",
            "latency_ms": 12.3,
            "status": "success",
        },
        parent_event_id=tool_start_2,
    )
    tick()

    # -----------------------------------------------------------------
    # 18. agent.task_end (Writer finishes)
    # -----------------------------------------------------------------
    logger.log_event(
        "agent.task_end",
        source_node=writer_node,
        payload={
            "agent_role": "Writer",
            "latency_ms": 1234.5,
            "status": "success",
            "result_shape": "str(len=3200)",
        },
        parent_event_id=writer_task_id,
    )
    tick()

    # -----------------------------------------------------------------
    # 19. delegation.completed (Researcher -> Writer)
    # -----------------------------------------------------------------
    logger.log_event(
        "delegation.completed",
        source_node=researcher_node,
        target_node=writer_node,
        edge_type="delegates_to",
        payload={
            "delegator": "Researcher",
            "delegate": "Writer",
            "latency_ms": 1890.0,
            "status": "success",
        },
        parent_event_id=deleg_1_id,
    )
    tick()

    # -----------------------------------------------------------------
    # 20. delegation.initiated (Writer -> Reviewer)
    # -----------------------------------------------------------------
    deleg_2_id = logger.log_event(
        "delegation.initiated",
        source_node=writer_node,
        target_node=reviewer_node,
        edge_type="delegates_to",
        payload={
            "delegator": "Writer",
            "delegate": "Reviewer",
            "task_hash": f"hash:{hash('review_article') & 0xFFFFFFFF:08x}",
        },
        parent_event_id=exec_start_id,
    )
    tick()

    # -----------------------------------------------------------------
    # 21. agent.task_start (Reviewer)
    # -----------------------------------------------------------------
    reviewer_task_id = logger.log_event(
        "agent.task_start",
        source_node=reviewer_node,
        payload={
            "agent_role": "Reviewer",
            "task_name": "review_article",
            "tools_available": [],
        },
        parent_event_id=deleg_2_id,
    )
    tick()

    # -----------------------------------------------------------------
    # 22. data.read (Reviewer reads SharedMemory)
    # -----------------------------------------------------------------
    logger.log_event(
        "data.read",
        source_node=memory_node,
        payload={
            "channel": "SharedMemory",
            "data_shape": "dict(keys=['research_findings', 'article_draft'], len=2)",
            "reader": "Reviewer",
        },
        parent_event_id=reviewer_task_id,
    )
    tick()

    # -----------------------------------------------------------------
    # 23. llm.call_start (Reviewer thinks)
    # -----------------------------------------------------------------
    llm_start_3 = logger.log_event(
        "llm.call_start",
        source_node=llm_node,
        payload={
            "model_requested": "Qwen/Qwen2.5-72B-Instruct",
            "message_count": 7,
            "has_tools": False,
        },
        parent_event_id=reviewer_task_id,
    )
    tick()

    # -----------------------------------------------------------------
    # 24. decision.made (Reviewer decides)
    # -----------------------------------------------------------------
    logger.log_event(
        "decision.made",
        source_node=reviewer_node,
        payload={
            "decision_type": "approval",
            "options_considered": ["approve", "request_revision", "reject"],
            "selected": "approve",
            "confidence": 0.92,
        },
        parent_event_id=reviewer_task_id,
    )
    tick()

    # -----------------------------------------------------------------
    # 25. llm.call_end (Reviewer LLM response)
    # -----------------------------------------------------------------
    logger.log_event(
        "llm.call_end",
        source_node=llm_node,
        payload={
            "model_requested": "Qwen/Qwen2.5-72B-Instruct",
            "model_actual": "Qwen/Qwen2.5-72B-Instruct",
            "latency_ms": 345.6,
            "input_tokens": 2000,
            "output_tokens": 150,
            "finish_reason": "stop",
        },
        parent_event_id=llm_start_3,
    )
    tick()

    # -----------------------------------------------------------------
    # 26. guardrail.triggered (Content quality check)
    # -----------------------------------------------------------------
    logger.log_event(
        "guardrail.triggered",
        source_node=guardrail_node,
        payload={
            "guardrail_name": "QualityGuard",
            "trigger_reason": "quality_score_check",
            "action_taken": "passed",
            "quality_score": 0.87,
        },
        parent_event_id=reviewer_task_id,
    )
    tick()

    # -----------------------------------------------------------------
    # 27. agent.task_end (Reviewer finishes)
    # -----------------------------------------------------------------
    logger.log_event(
        "agent.task_end",
        source_node=reviewer_node,
        payload={
            "agent_role": "Reviewer",
            "latency_ms": 678.9,
            "status": "success",
            "result_shape": "dict(keys=['decision', 'feedback'], len=2)",
        },
        parent_event_id=reviewer_task_id,
    )
    tick()

    # -----------------------------------------------------------------
    # 28. delegation.completed (Writer -> Reviewer)
    # -----------------------------------------------------------------
    logger.log_event(
        "delegation.completed",
        source_node=writer_node,
        target_node=reviewer_node,
        edge_type="delegates_to",
        payload={
            "delegator": "Writer",
            "delegate": "Reviewer",
            "latency_ms": 1050.0,
            "status": "success",
        },
        parent_event_id=deleg_2_id,
    )
    tick()

    # -----------------------------------------------------------------
    # 29. execution.end
    # -----------------------------------------------------------------
    logger.log_event(
        "execution.end",
        source_node=crew_node,
        payload={
            "latency_ms": 4500.0,
            "status": "success",
            "result_shape": "dict(keys=['article', 'review'], len=2)",
        },
        parent_event_id=exec_start_id,
    )

    events = _read_events(events_file)
    return events, node_ids


def main() -> None:
    # -----------------------------------------------------------------
    # Run 3 simulations with different inputs
    # -----------------------------------------------------------------
    run_configs = [
        {
            "run_id": "run-001",
            "repo_id": "synthetic-repo-001",
            "input_query": "How does quantum computing work?",
        },
        {
            "run_id": "run-002",
            "repo_id": "synthetic-repo-001",
            "input_query": "Explain climate change mitigation strategies",
        },
        {
            "run_id": "run-003",
            "repo_id": "synthetic-repo-001",
            "input_query": "What are the latest advances in AI safety?",
        },
    ]

    all_run_events: list[tuple[str, list[dict], dict]] = []
    temp_files: list[str] = []

    print("=" * 80)
    print("  SYNTHETIC EXECUTION SIMULATION")
    print("=" * 80)
    print()

    for config in run_configs:
        fd, events_file = tempfile.mkstemp(
            suffix=".jsonl", prefix=f"stratum_{config['run_id']}_"
        )
        os.close(fd)
        temp_files.append(events_file)

        events, node_ids = simulate_run(
            run_id=config["run_id"],
            repo_id=config["repo_id"],
            events_file=events_file,
            input_query=config["input_query"],
        )
        all_run_events.append((config["run_id"], events, node_ids))
        print(f"  Run '{config['run_id']}': {len(events)} events emitted")

    print()

    # -----------------------------------------------------------------
    # Total events per run
    # -----------------------------------------------------------------
    print("  EVENTS PER RUN")
    print("  " + "-" * 50)
    for run_id, events, _ in all_run_events:
        print(f"  {run_id}: {len(events)} events")
    print()

    # -----------------------------------------------------------------
    # Event type distribution (aggregate across all runs)
    # -----------------------------------------------------------------
    print("  EVENT TYPE DISTRIBUTION (all runs)")
    print("  " + "-" * 60)
    all_events = []
    for _, events, _ in all_run_events:
        all_events.extend(events)

    type_counts = Counter(ev["event_type"] for ev in all_events)
    max_type_len = max(len(et) for et in type_counts)
    for et, count in type_counts.most_common():
        per_run = count / len(run_configs)
        bar = "#" * count
        print(f"  {et:<{max_type_len}}  {count:>3} total  ({per_run:.1f}/run)  {bar}")
    print()

    # -----------------------------------------------------------------
    # Node IDs generated
    # -----------------------------------------------------------------
    print("  NODE IDS GENERATED")
    print("  " + "-" * 70)
    _, _, first_run_nodes = all_run_events[0]
    for name, nid in sorted(first_run_nodes.items()):
        print(f"  {name:<16}  {nid}")
    print()

    # -----------------------------------------------------------------
    # Verify node IDs are deterministic across runs
    # -----------------------------------------------------------------
    print("  NODE ID DETERMINISM CHECK")
    print("  " + "-" * 60)
    all_node_id_sets = [node_ids for _, _, node_ids in all_run_events]
    deterministic = True
    for name in first_run_nodes:
        values = [nids[name] for nids in all_node_id_sets]
        if len(set(values)) == 1:
            print(f"  [+] {name:<16}  DETERMINISTIC (same across all {len(run_configs)} runs)")
        else:
            print(f"  [X] {name:<16}  NOT DETERMINISTIC: {values}")
            deterministic = False

    print()
    if deterministic:
        print("  Result: ALL node IDs are deterministic across runs")
    else:
        print("  Result: SOME node IDs differ across runs (UNEXPECTED)")
    print()

    # -----------------------------------------------------------------
    # Verify event parent chains
    # -----------------------------------------------------------------
    print("  PARENT EVENT CHAIN VALIDATION")
    print("  " + "-" * 60)
    for run_id, events, _ in all_run_events:
        event_ids = {ev["event_id"] for ev in events}
        parent_refs = [
            ev["parent_event_id"] for ev in events
            if "parent_event_id" in ev and ev["parent_event_id"] is not None
        ]
        orphans = [pid for pid in parent_refs if pid not in event_ids]
        if not orphans:
            print(f"  [+] {run_id}: all {len(parent_refs)} parent refs are valid")
        else:
            print(f"  [X] {run_id}: {len(orphans)} orphaned parent refs")
    print()

    # -----------------------------------------------------------------
    # Verify event ordering within each run
    # -----------------------------------------------------------------
    print("  TIMESTAMP ORDERING CHECK")
    print("  " + "-" * 60)
    for run_id, events, _ in all_run_events:
        timestamps = [ev["timestamp_ns"] for ev in events]
        is_ordered = all(timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1))
        if is_ordered:
            duration_ns = timestamps[-1] - timestamps[0]
            duration_ms = duration_ns / 1_000_000
            print(f"  [+] {run_id}: timestamps are ordered (span: {duration_ms:.1f}ms)")
        else:
            print(f"  [X] {run_id}: timestamps are NOT ordered")
    print()

    # -----------------------------------------------------------------
    # Verify run_id isolation
    # -----------------------------------------------------------------
    print("  RUN ISOLATION CHECK")
    print("  " + "-" * 60)
    for run_id, events, _ in all_run_events:
        run_ids_in_file = {ev["run_id"] for ev in events}
        if run_ids_in_file == {run_id}:
            print(f"  [+] {run_id}: all events have correct run_id")
        else:
            print(f"  [X] {run_id}: mixed run_ids found: {run_ids_in_file}")
    print()

    # -----------------------------------------------------------------
    # Event type consistency across runs
    # -----------------------------------------------------------------
    print("  EVENT TYPE CONSISTENCY ACROSS RUNS")
    print("  " + "-" * 60)
    run_type_sets = []
    for run_id, events, _ in all_run_events:
        types = set(ev["event_type"] for ev in events)
        run_type_sets.append(types)

    if all(ts == run_type_sets[0] for ts in run_type_sets):
        print(f"  [+] All runs emit the same {len(run_type_sets[0])} event types")
    else:
        union = set().union(*run_type_sets)
        intersection = run_type_sets[0].intersection(*run_type_sets[1:])
        only_some = union - intersection
        print(f"  [X] Event types differ across runs. Common: {len(intersection)}, "
              f"varying: {only_some}")
    print()

    # -----------------------------------------------------------------
    # Source node distribution
    # -----------------------------------------------------------------
    print("  SOURCE NODE DISTRIBUTION (all runs)")
    print("  " + "-" * 60)
    source_names = Counter()
    for ev in all_events:
        sn = ev.get("source_node")
        if sn:
            source_names[sn.get("node_name", "unknown")] += 1

    for name, count in source_names.most_common():
        bar = "#" * count
        print(f"  {name:<40}  {count:>3}  {bar}")
    print()

    # -----------------------------------------------------------------
    # Edge type distribution
    # -----------------------------------------------------------------
    print("  EDGE TYPE DISTRIBUTION (all runs)")
    print("  " + "-" * 40)
    edge_types = Counter(
        ev.get("edge_type") for ev in all_events if ev.get("edge_type")
    )
    for et, count in edge_types.most_common():
        print(f"  {et:<20}  {count:>3}")
    print()

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print("=" * 80)
    print("  SUMMARY")
    print("  " + "-" * 70)
    print(f"  Total runs simulated:        {len(run_configs)}")
    print(f"  Total events generated:      {len(all_events)}")
    print(f"  Events per run:              {len(all_events) // len(run_configs)}")
    print(f"  Distinct event types:        {len(type_counts)}")
    print(f"  Distinct source nodes:       {len(source_names)}")
    print(f"  Node IDs deterministic:      {'YES' if deterministic else 'NO'}")
    print("=" * 80)

    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------
    _reset_logger()
    for key in ("STRATUM_EVENTS_FILE", "STRATUM_RUN_ID",
                 "STRATUM_REPO_ID", "STRATUM_FRAMEWORK"):
        os.environ.pop(key, None)
    for f in temp_files:
        try:
            os.unlink(f)
        except OSError:
            pass


if __name__ == "__main__":
    main()
