"""Evaluation script for the graph overlay (enrichment + edge detection).

Creates a synthetic structural graph matching stratum-cli output format,
generates JSONL-style events across 3 runs, enriches the graph, and detects
emergent / dead edges.

Prints:
  (a) Per-node behavioral annotations
  (b) Per-edge behavioral annotations
  (c) Emergent edges
  (d) Dead edges
  (e) structural_prediction_match flags
"""

from __future__ import annotations

import json
import sys
import os
import time

# ---------------------------------------------------------------------------
# Path setup -- stratum_lab is NOT an installed package
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stratum_lab.overlay.enricher import enrich_graph
from stratum_lab.overlay.edges import detect_emergent_edges, detect_dead_edges

# =========================================================================
# 1. Synthetic structural graph
# =========================================================================

structural_graph: dict = {
    "repo_id": "test_repo_001",
    "framework": "crewai",
    "nodes": {
        "agent_researcher": {
            "node_type": "agent",
            "name": "Researcher",
            "source_file": "agents.py",
            "line_number": 10,
        },
        "agent_writer": {
            "node_type": "agent",
            "name": "Writer",
            "source_file": "agents.py",
            "line_number": 30,
        },
        "agent_reviewer": {
            "node_type": "agent",
            "name": "Reviewer",
            "source_file": "agents.py",
            "line_number": 50,
        },
        "cap_web_search_tool": {
            "node_type": "capability",
            "name": "WebSearch",
            "kind": "tool",
        },
        "cap_file_writer_tool": {
            "node_type": "capability",
            "name": "FileWriter",
            "kind": "tool",
        },
        "cap_summarize_tool": {
            "node_type": "capability",
            "name": "Summarize",
            "kind": "tool",
        },
        "cap_llm_call": {
            "node_type": "capability",
            "name": "LLMCall",
            "kind": "llm",
        },
        "cap_review_call": {
            "node_type": "capability",
            "name": "ReviewCall",
            "kind": "llm",
        },
        "ds_shared_memory": {
            "node_type": "data_store",
            "name": "SharedMemory",
        },
        "ext_web_api": {
            "node_type": "external",
            "name": "WebAPI",
        },
    },
    "edges": {
        "e1": {"edge_type": "delegates_to", "source": "agent_researcher", "target": "agent_writer"},
        "e2": {"edge_type": "delegates_to", "source": "agent_writer", "target": "agent_reviewer"},
        "e3": {"edge_type": "uses", "source": "agent_researcher", "target": "cap_web_search_tool"},
        "e4": {"edge_type": "uses", "source": "agent_writer", "target": "cap_file_writer_tool"},
        "e5": {"edge_type": "uses", "source": "agent_researcher", "target": "cap_llm_call"},
        "e6": {"edge_type": "uses", "source": "agent_reviewer", "target": "cap_review_call"},
        "e7": {"edge_type": "writes_to", "source": "agent_researcher", "target": "ds_shared_memory"},
        "e8": {"edge_type": "writes_to", "source": "agent_writer", "target": "ds_shared_memory"},
        "e9": {"edge_type": "reads_from", "source": "agent_reviewer", "target": "ds_shared_memory"},
        "e10": {"edge_type": "calls", "source": "cap_web_search_tool", "target": "ext_web_api"},
    },
}


# =========================================================================
# 2. Synthetic event helpers
# =========================================================================

_counter = 0


def _eid() -> str:
    global _counter
    _counter += 1
    return f"evt_{_counter:06d}"


def _ts(offset_ms: int = 0) -> int:
    """Return a nanosecond timestamp with an optional ms offset."""
    return int((time.time() + offset_ms / 1000) * 1_000_000_000)


def _source(node_name: str, node_type: str = "agent") -> dict:
    """Build a source_node dict that the enricher can match by name."""
    return {"node_type": node_type, "node_id": "", "node_name": node_name}


def _agent_task_events(
    agent_name: str,
    offset_ms: int,
    duration_ms: int = 500,
    status: str = "success",
) -> list[dict]:
    """Emit agent.task_start + agent.task_end pair."""
    start_id = _eid()
    end_id = _eid()
    return [
        {
            "event_id": start_id,
            "event_type": "agent.task_start",
            "timestamp_ns": _ts(offset_ms),
            "source_node": _source(agent_name),
            "payload": {},
        },
        {
            "event_id": end_id,
            "event_type": "agent.task_end",
            "timestamp_ns": _ts(offset_ms + duration_ms),
            "parent_event_id": start_id,
            "source_node": _source(agent_name),
            "payload": {"status": status},
        },
    ]


def _tool_events(
    tool_name: str,
    agent_name: str,
    offset_ms: int,
    duration_ms: int = 100,
    status: str = "success",
) -> list[dict]:
    """Emit tool.invoked + tool.completed pair with source+target for edge matching."""
    start_id = _eid()
    end_id = _eid()
    return [
        {
            "event_id": start_id,
            "event_type": "tool.invoked",
            "timestamp_ns": _ts(offset_ms),
            "source_node": _source(agent_name, "agent"),
            "target_node": _source(tool_name, "capability"),
            "payload": {},
        },
        {
            "event_id": end_id,
            "event_type": "tool.completed",
            "timestamp_ns": _ts(offset_ms + duration_ms),
            "parent_event_id": start_id,
            "source_node": _source(agent_name, "agent"),
            "target_node": _source(tool_name, "capability"),
            "payload": {"status": status},
        },
    ]


def _llm_events(
    llm_name: str,
    agent_name: str,
    offset_ms: int,
    duration_ms: int = 300,
    input_tokens: int = 500,
    output_tokens: int = 200,
) -> list[dict]:
    """Emit llm.call_start + llm.call_end pair."""
    start_id = _eid()
    end_id = _eid()
    return [
        {
            "event_id": start_id,
            "event_type": "llm.call_start",
            "timestamp_ns": _ts(offset_ms),
            "source_node": _source(agent_name, "agent"),
            "target_node": _source(llm_name, "capability"),
            "payload": {},
        },
        {
            "event_id": end_id,
            "event_type": "llm.call_end",
            "timestamp_ns": _ts(offset_ms + duration_ms),
            "parent_event_id": start_id,
            "source_node": _source(agent_name, "agent"),
            "target_node": _source(llm_name, "capability"),
            "payload": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        },
    ]


def _delegation_events(
    source_agent: str,
    target_agent: str,
    offset_ms: int,
    duration_ms: int = 50,
) -> list[dict]:
    """Emit delegation.initiated + delegation.completed pair."""
    start_id = _eid()
    end_id = _eid()
    return [
        {
            "event_id": start_id,
            "event_type": "delegation.initiated",
            "timestamp_ns": _ts(offset_ms),
            "source_node": _source(source_agent),
            "target_node": _source(target_agent),
            "payload": {},
        },
        {
            "event_id": end_id,
            "event_type": "delegation.completed",
            "timestamp_ns": _ts(offset_ms + duration_ms),
            "parent_event_id": start_id,
            "source_node": _source(source_agent),
            "target_node": _source(target_agent),
            "payload": {},
        },
    ]


def _error_event(
    agent_name: str,
    offset_ms: int,
    error_handling: str = "retry",
    propagated: bool = False,
) -> list[dict]:
    """Emit an error.occurred event and optionally error.propagated."""
    events = [
        {
            "event_id": _eid(),
            "event_type": "error.occurred",
            "timestamp_ns": _ts(offset_ms),
            "source_node": _source(agent_name),
            "payload": {"error_handling": error_handling},
        },
    ]
    if propagated:
        events.append(
            {
                "event_id": _eid(),
                "event_type": "error.propagated",
                "timestamp_ns": _ts(offset_ms + 5),
                "source_node": _source(agent_name),
                "payload": {"error_handling": error_handling},
            }
        )
    return events


def _write_event(agent_name: str, ds_name: str, offset_ms: int) -> list[dict]:
    """Emit a write interaction from agent to data store (edge traversal)."""
    return [
        {
            "event_id": _eid(),
            "event_type": "data.write",
            "timestamp_ns": _ts(offset_ms),
            "source_node": _source(agent_name, "agent"),
            "target_node": _source(ds_name, "data_store"),
            "payload": {"data_size_bytes": 1024},
        },
    ]


def _read_event(agent_name: str, ds_name: str, offset_ms: int) -> list[dict]:
    """Emit a read interaction from agent to data store (edge traversal)."""
    return [
        {
            "event_id": _eid(),
            "event_type": "data.read",
            "timestamp_ns": _ts(offset_ms),
            "source_node": _source(agent_name, "agent"),
            "target_node": _source(ds_name, "data_store"),
            "payload": {},
        },
    ]


# =========================================================================
# 3. Build 3 runs of synthetic events
# =========================================================================

def build_run_events(run_index: int) -> list[dict]:
    """Build a realistic set of events for one run.

    * run_index 0,1 : normal flow  (Researcher -> Writer -> Reviewer)
    * run_index 2   : includes more errors and an emergent edge
                       (Reviewer directly calls WebSearch -- no structural edge!)

    Structural edge e10 (cap_web_search_tool -> ext_web_api) is NEVER
    traversed so it will show up as a dead edge.

    Also: cap_summarize_tool is never used by any agent, making the "uses"
    edge absent at runtime.
    """
    base = run_index * 5000  # stagger timestamps across runs
    events: list[dict] = []

    # ---- Researcher works ----
    events += _agent_task_events("Researcher", base + 0, duration_ms=800)
    events += _tool_events("WebSearch", "Researcher", base + 50, duration_ms=200)
    events += _llm_events("LLMCall", "Researcher", base + 300, duration_ms=350,
                          input_tokens=600, output_tokens=250)
    events += _write_event("Researcher", "SharedMemory", base + 700)

    # ---- Researcher delegates to Writer ----
    events += _delegation_events("Researcher", "Writer", base + 800)

    # ---- Writer works ----
    events += _agent_task_events("Writer", base + 900, duration_ms=600)
    events += _tool_events("FileWriter", "Writer", base + 950, duration_ms=150)
    events += _write_event("Writer", "SharedMemory", base + 1200)

    # ---- Writer delegates to Reviewer ----
    events += _delegation_events("Writer", "Reviewer", base + 1500)

    # ---- Reviewer works ----
    events += _agent_task_events("Reviewer", base + 1600, duration_ms=400)
    events += _llm_events("ReviewCall", "Reviewer", base + 1650, duration_ms=280,
                          input_tokens=800, output_tokens=100)
    events += _read_event("Reviewer", "SharedMemory", base + 1900)

    # ---- Run-specific variations ----
    if run_index == 1:
        # Run 1: Researcher encounters an error that is retried
        events += _error_event("Researcher", base + 250, error_handling="retry")

    if run_index == 2:
        # Run 2: Reviewer directly calls WebSearch (emergent edge!)
        events += _tool_events("WebSearch", "Reviewer", base + 1700, duration_ms=120)
        # Run 2: Writer has an error propagated downstream
        events += _error_event("Writer", base + 1100, error_handling="propagate",
                               propagated=True)
        # Run 2: An interaction between Researcher and Reviewer (no structural edge)
        events += _delegation_events("Researcher", "Reviewer", base + 1580)

    return events


# Build run_records
run_records: list[dict] = []
all_events_for_edge_detection: list[dict] = []

for i in range(3):
    run_events = build_run_events(i)
    run_records.append({
        "events": run_events,
        "run_id": f"run_{i:03d}",
        "repo_id": "test_repo_001",
    })
    all_events_for_edge_detection.extend(run_events)


# =========================================================================
# 4. Enrich the graph
# =========================================================================

enriched = enrich_graph(structural_graph, run_records)


# =========================================================================
# 5. Detect emergent and dead edges
# =========================================================================

# Filter to events with both source_node and target_node (interaction events)
interaction_events = [
    e for e in all_events_for_edge_detection
    if e.get("source_node") and e.get("target_node")
]

emergent = detect_emergent_edges(
    structural_graph["edges"],
    interaction_events,
    structural_graph["nodes"],
    total_runs=3,
)

dead = detect_dead_edges(
    structural_graph["edges"],
    interaction_events,
    total_runs=3,
    structural_nodes=structural_graph["nodes"],
)


# =========================================================================
# 6. Print results
# =========================================================================

def pretty(obj: object) -> str:
    return json.dumps(obj, indent=2, default=str)


print("=" * 80)
print("(a) PER-NODE BEHAVIORAL ANNOTATIONS")
print("=" * 80)
for node_id, node_data in enriched["nodes"].items():
    behavioral = node_data["behavioral"]
    structural = node_data["structural"]
    print(f"\n--- {node_id} ({structural.get('name', '?')}) ---")
    print(f"  activation_count : {behavioral['activation_count']}")
    print(f"  activation_rate  : {behavioral['activation_rate']}")
    print(f"  throughput       : {pretty(behavioral['throughput'])}")
    print(f"  latency          : {pretty(behavioral['latency'])}")
    print(f"  error_behavior   : {pretty(behavioral['error_behavior'])}")
    print(f"  decision_behavior: {behavioral['decision_behavior']}")
    print(f"  model_sensitivity: {pretty(behavioral['model_sensitivity'])}")
    print(f"  resource_usage   : {pretty(behavioral['resource_usage'])}")

print("\n" + "=" * 80)
print("(b) PER-EDGE BEHAVIORAL ANNOTATIONS")
print("=" * 80)
for edge_id, edge_data in enriched["edges"].items():
    behavioral = edge_data["behavioral"]
    structural = edge_data["structural"]
    src, tgt = structural.get("source", "?"), structural.get("target", "?")
    print(f"\n--- {edge_id}: {src} -> {tgt} ({structural.get('edge_type', '?')}) ---")
    print(f"  traversal_count : {behavioral['traversal_count']}")
    print(f"  activation_rate : {behavioral['activation_rate']}")
    print(f"  never_activated : {behavioral['never_activated']}")
    print(f"  data_flow       : {pretty(behavioral['data_flow'])}")
    print(f"  error_crossings : {pretty(behavioral['error_crossings'])}")
    print(f"  latency_contribution_ms: {behavioral['latency_contribution_ms']}")

print("\n" + "=" * 80)
print("(c) EMERGENT EDGES")
print("=" * 80)
if emergent:
    for ee in emergent:
        print(f"\n  {ee['edge_id']}: {ee['source_node_id']} -> {ee['target_node_id']}")
        print(f"    edge_type        : {ee['edge_type']}")
        print(f"    traversal_count  : {ee['traversal_count']}")
        print(f"    activation_rate  : {ee['activation_rate']}")
        print(f"    significance     : {ee['significance']}")
        print(f"    trigger_condition: {ee['trigger_condition']}")
else:
    print("  (none detected)")

print("\n" + "=" * 80)
print("(d) DEAD EDGES")
print("=" * 80)
if dead:
    for de in dead:
        print(f"\n  {de['edge_id']}: {de['source']} -> {de['target']} ({de['edge_type']})")
        print(f"    runs_observed   : {de['runs_observed']}")
        print(f"    possible_reasons: {de['possible_reasons']}")
else:
    print("  (none detected)")

print("\n" + "=" * 80)
print("(e) STRUCTURAL PREDICTION MATCH FLAGS")
print("=" * 80)
for node_id, node_data in enriched["nodes"].items():
    eb = node_data["behavioral"]["error_behavior"]
    match_flag = eb.get("structural_prediction_match")
    errors = eb["errors_occurred"]
    print(f"  {node_id:30s}  errors={errors:3d}  prediction_match={match_flag}")

print("\n" + "=" * 80)
print("UNMAPPED EVENTS")
print("=" * 80)
print(pretty(enriched["unmapped_events"]))
print("\nDone.")
