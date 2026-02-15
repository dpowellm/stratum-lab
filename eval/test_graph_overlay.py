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

import hashlib
import json
import sys
import os
import time

# ---------------------------------------------------------------------------
# Path setup -- stratum_lab is NOT an installed package
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stratum_lab.overlay.enricher import enrich_graph

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
            "error_handling": {"strategy": "retry"},
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
        "guard_output": {
            "node_type": "guardrail",
            "name": "OutputGuard",
            "kind": "output_validation",
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
        "e11": {"edge_type": "filtered_by", "source": "agent_writer", "target": "guard_output"},
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
    output_hash: str | None = None,
    output_type: str | None = None,
) -> list[dict]:
    """Emit llm.call_start + llm.call_end pair."""
    start_id = _eid()
    end_id = _eid()
    end_payload: dict = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    if output_hash:
        end_payload["output_hash"] = output_hash
    if output_type:
        end_payload["output_type"] = output_type
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
            "payload": end_payload,
        },
    ]


def _delegation_events(
    source_agent: str,
    target_agent: str,
    offset_ms: int,
    duration_ms: int = 50,
    context_hash: str | None = None,
    context_source_node: str | None = None,
) -> list[dict]:
    """Emit delegation.initiated + delegation.completed pair."""
    start_id = _eid()
    end_id = _eid()
    init_payload: dict = {}
    if context_hash:
        init_payload["context_hash"] = context_hash
    if context_source_node:
        init_payload["context_source_node"] = context_source_node
    return [
        {
            "event_id": start_id,
            "event_type": "delegation.initiated",
            "timestamp_ns": _ts(offset_ms),
            "source_node": _source(source_agent),
            "target_node": _source(target_agent),
            "payload": init_payload,
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
                "payload": {
                    "error_handling": error_handling,
                    "error_type": "propagated_failure",
                    "propagation_path": [agent_name, "SharedMemory"],
                    "downstream_impact": True,
                },
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


def _external_call_event(
    cap_name: str, ext_name: str, offset_ms: int,
) -> list[dict]:
    """Emit an external.call event (capability -> external service)."""
    return [
        {
            "event_id": _eid(),
            "event_type": "external.call",
            "timestamp_ns": _ts(offset_ms),
            "source_node": _source(cap_name, "capability"),
            "target_node": _source(ext_name, "external"),
            "payload": {},
        },
    ]


def _decision_event(
    agent_name: str, offset_ms: int,
    option_hash: str = "opt_a", confidence: float = 0.85,
    input_hash: str = "inp_001",
) -> list[dict]:
    """Emit a decision.made event."""
    return [
        {
            "event_id": _eid(),
            "event_type": "decision.made",
            "timestamp_ns": _ts(offset_ms),
            "source_node": _source(agent_name),
            "payload": {
                "selected_option_hash": option_hash,
                "confidence": confidence,
                "input_hash": input_hash,
            },
        },
    ]


def _guardrail_event(
    guardrail_name: str, offset_ms: int,
    action_prevented: bool = False,
    bypassed: bool = False,
    retry_triggered: bool = False,
    latency_ms: float = 3.0,
) -> list[dict]:
    """Emit a guardrail.triggered event."""
    return [
        {
            "event_id": _eid(),
            "event_type": "guardrail.triggered",
            "timestamp_ns": _ts(offset_ms),
            "source_node": _source(guardrail_name, "guardrail"),
            "payload": {
                "action_prevented": action_prevented,
                "bypassed": bypassed,
                "retry_triggered": retry_triggered,
                "latency_ms": latency_ms,
            },
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

    # Semantic content hashes — Researcher's LLM output flows to Writer via delegation
    researcher_output_hash = f"res_out_{run_index:03d}"
    writer_output_hash = f"wrt_out_{run_index:03d}"

    # ---- Researcher works ----
    events += _agent_task_events("Researcher", base + 0, duration_ms=800)
    events += _tool_events("WebSearch", "Researcher", base + 50, duration_ms=200)
    events += _llm_events("LLMCall", "Researcher", base + 300, duration_ms=350,
                          input_tokens=600, output_tokens=250,
                          output_hash=researcher_output_hash,
                          output_type="structured_data")
    events += _write_event("Researcher", "SharedMemory", base + 700)

    # ---- Researcher delegates to Writer (passing its output as context) ----
    events += _delegation_events("Researcher", "Writer", base + 800,
                                  context_hash=researcher_output_hash,
                                  context_source_node="agent_researcher")

    # ---- Writer works ----
    events += _agent_task_events("Writer", base + 900, duration_ms=600)
    events += _tool_events("FileWriter", "Writer", base + 950, duration_ms=150)
    events += _write_event("Writer", "SharedMemory", base + 1200)

    # ---- Writer delegates to Reviewer (passing its output as context) ----
    events += _delegation_events("Writer", "Reviewer", base + 1500,
                                  context_hash=writer_output_hash,
                                  context_source_node="agent_writer")

    # ---- Reviewer works ----
    events += _agent_task_events("Reviewer", base + 1600, duration_ms=400)
    events += _llm_events("ReviewCall", "Reviewer", base + 1650, duration_ms=280,
                          input_tokens=800, output_tokens=100,
                          output_hash=writer_output_hash,
                          output_type="short_text")
    events += _read_event("Reviewer", "SharedMemory", base + 1900)

    # ---- External call (WebSearch -> WebAPI) ----
    events += _external_call_event("WebSearch", "WebAPI", base + 260)

    # ---- Decision event (Researcher decides) ----
    # Use per-run input_hash so that runs sharing the same input text
    # produce the same hash, enabling determinism metric computation.
    run_ih = hashlib.sha256(run_input_texts[run_index].encode("utf-8")).hexdigest()[:12]
    option = "opt_a" if run_index < 2 else "opt_b"
    events += _decision_event(
        "Researcher", base + 750,
        option_hash=option, confidence=0.85 + run_index * 0.03,
        input_hash=run_ih,
    )

    # ---- Guardrail check (Writer -> OutputGuard) ----
    events += _guardrail_event(
        "OutputGuard", base + 1400,
        action_prevented=(run_index == 0),
        bypassed=(run_index == 1),
        retry_triggered=(run_index == 2),
        latency_ms=2.5 + run_index,
    )

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
# Make run_000 and run_002 share the same input so the enricher can compute
# same_input_activation_consistency and same_input_path_consistency.
run_input_texts = [
    "Research the latest breakthroughs in renewable energy",  # run_000
    "Compare AI agent frameworks for enterprise use",          # run_001
    "Research the latest breakthroughs in renewable energy",  # run_002 — SAME as run_000
]

run_records: list[dict] = []
all_events_for_edge_detection: list[dict] = []

for i in range(3):
    run_events = build_run_events(i)
    ih = hashlib.sha256(run_input_texts[i].encode("utf-8")).hexdigest()[:12]
    run_records.append({
        "events": run_events,
        "run_id": f"run_{i:03d}",
        "repo_id": "test_repo_001",
        "metadata": {"input_hash": ih},
    })
    all_events_for_edge_detection.extend(run_events)


# =========================================================================
# 4. Enrich the graph
# =========================================================================

enriched = enrich_graph(structural_graph, run_records)


# =========================================================================
# 5. Emergent and dead edges (now computed by enrich_graph)
# =========================================================================

emergent = enriched["emergent_edges"]
dead = enriched["dead_edges"]


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
    if behavioral.get("guardrail_effectiveness"):
        print(f"  guardrail_eff    : {pretty(behavioral['guardrail_effectiveness'])}")
    if behavioral.get("determinism"):
        print(f"  determinism      : {pretty(behavioral['determinism'])}")

print("\n" + "=" * 80)
print("(f) EXECUTION PATHS & PATH ANALYSIS")
print("=" * 80)
if enriched.get("execution_paths"):
    print(f"  Runs with paths: {len(enriched['execution_paths'])}")
    for ep in enriched["execution_paths"]:
        steps = ep["execution_path"]
        print(f"  Run {ep['run_id']}: {len(steps)} edge traversals")
        for step in steps[:5]:
            print(f"    {step['edge_id']}: {step['source']} -> {step['target']} ({step['edge_type']})")
        if len(steps) > 5:
            print(f"    ... and {len(steps) - 5} more")
else:
    print("  (no execution paths)")

if enriched.get("path_analysis"):
    pa = enriched["path_analysis"]
    print(f"\n  Path analysis:")
    print(f"    distinct_paths:          {pa['distinct_paths']}")
    print(f"    dominant_path_frequency: {pa['dominant_path_frequency']}")
    print(f"    path_divergence_points:  {pa['path_divergence_points']}")
    print(f"    conditional_edges:       {len(pa.get('conditional_edge_activation_rates', {}))}")

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
print("(g) CASCADE ANALYSIS")
print("=" * 80)
cascade = enriched.get("cascade_analysis", {})
if cascade:
    print(f"  max_cascade_depth:      {cascade.get('max_cascade_depth', 0)}")
    print(f"  avg_cascade_depth:      {cascade.get('avg_cascade_depth', 0):.2f}")
    print(f"  cascading_failure_rate: {cascade.get('cascading_failure_rate', 0):.2f}")
    for c in cascade.get("cascades", []):
        print(f"\n  Origin: {c['origin_node']}")
        print(f"    origin_errors:          {c.get('origin_errors', c.get('origin_error_count', 0))}")
        print(f"    max_depth:              {c.get('max_depth', 0)}")
        print(f"    affected_downstream:    {len(c.get('affected_downstream', []))}")
        for ds in c.get("affected_downstream", []):
            print(f"      {ds['node_id']} (depth={ds['depth']}, errors={ds.get('error_count', ds.get('errors', 0))})")
else:
    print("  (no cascade analysis)")

print("\n" + "=" * 80)
print("(h) SEMANTIC LINEAGE")
print("=" * 80)
semantic = enriched.get("semantic_lineage", {})
if semantic:
    print(f"  handoff_count:                 {len(semantic.get('handoffs', []))}")
    print(f"  unvalidated_fraction:          {semantic.get('unvalidated_fraction', 0):.2f}")
    print(f"  semantic_chain_depth:          {semantic.get('semantic_chain_depth', 0)}")
    print(f"  max_blast_radius:              {semantic.get('max_blast_radius', 0)}")
    print(f"  classification_injection_count:{semantic.get('classification_injection_count', 0)}")
    det = semantic.get("semantic_determinism", {})
    if det:
        print(f"  semantic_determinism nodes:    {len(det)}")
        for nid, m in list(det.items())[:5]:
            print(f"    {nid}: deterministic={m.get('semantically_deterministic', True)}")
    for h in semantic.get("handoffs", [])[:5]:
        print(f"\n  Handoff: {h.get('source_node')} -> {h.get('target_node')}")
        print(f"    content_hash: {h.get('content_hash', '?')[:16]}...")
        print(f"    validated:    {h.get('validated', False)}")
else:
    print("  (no semantic lineage)")

# Check per-edge semantic_flow
print("\n" + "=" * 80)
print("(i) SEMANTIC FLOW ON EDGES")
print("=" * 80)
has_semantic_flow = False
for edge_id, edge_data in enriched["edges"].items():
    sf = edge_data.get("behavioral", {}).get("semantic_flow") or edge_data.get("semantic_flow")
    if sf:
        has_semantic_flow = True
        structural = edge_data.get("structural", edge_data)
        src, tgt = structural.get("source", "?"), structural.get("target", "?")
        print(f"  {edge_id}: {src} -> {tgt}")
        print(f"    content_hashes: {len(sf.get('content_hashes', []))}")
        print(f"    validated:      {sf.get('validated', False)}")
        print(f"    has_classification_dependency: {sf.get('has_classification_dependency', False)}")
if not has_semantic_flow:
    print("  (no semantic_flow on edges — expected if no output_hash in events)")

print("\n" + "=" * 80)
print("UNMAPPED EVENTS")
print("=" * 80)
print(pretty(enriched["unmapped_events"]))
print("\nDone.")
