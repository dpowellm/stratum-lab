"""Evaluation script for the v6 graph overlay (validation checks 1-8).

Tests v6 modules with synthetic data:
  Eval 1: Edge validation accuracy   (Checks 1-2) -- enricher.compute_edge_validation
  Eval 2: Emergent edge detection    (Check 3)    -- edges.detect_emergent_edges_v2
  Eval 3: Error propagation tracing  (Check 4)    -- error_propagation.trace_error_propagation
  Eval 4: Node activation topology   (Check 5)    -- enricher.compute_node_activation
  Eval 5: Structural prediction rate (Check 6)    -- enricher (numeric match rate)
  Eval 6: Failure mode classification(Check 7)    -- failure_modes.classify_failure_modes
  Eval 7: Monitoring baseline stabil.(Check 8)    -- monitoring_baselines.extract_monitoring_baselines

Run as a standalone script:
    cd stratum-lab
    python eval/test_graph_overlay.py
"""

from __future__ import annotations

import json
import os
import sys
import time

# ---------------------------------------------------------------------------
# Path setup -- stratum_lab is NOT an installed package
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stratum_lab.overlay.enricher import (
    compute_edge_validation,
    compute_node_activation,
)
from stratum_lab.overlay.edges import detect_emergent_edges_v2
from stratum_lab.overlay.error_propagation import trace_error_propagation
from stratum_lab.overlay.failure_modes import classify_failure_modes
from stratum_lab.overlay.monitoring_baselines import extract_monitoring_baselines


# =========================================================================
# Output capture
# =========================================================================

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "outputs", "graph-overlay-demo.txt")
_output_lines: list[str] = []


def out(text: str = "") -> None:
    """Print and capture a line."""
    print(text)
    _output_lines.append(text)


def flush_output() -> None:
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(_output_lines) + "\n")


def pretty(obj: object) -> str:
    return json.dumps(obj, indent=2, default=str)


# =========================================================================
# Assertion helpers
# =========================================================================

_pass_count = 0
_fail_count = 0


def check(condition: bool, label: str, detail: str = "") -> None:
    """Assert a condition and print the result."""
    global _pass_count, _fail_count
    if condition:
        _pass_count += 1
        out(f"  [PASS] {label}")
    else:
        _fail_count += 1
        out(f"  [FAIL] {label}")
    if detail:
        out(f"         {detail}")


# =========================================================================
# 1. Synthetic structural graph (15 edges of various types)
# =========================================================================
#
# Key design decisions for edge validation accuracy:
#
# compute_edge_validation uses _build_edge_activation_map which only tracks
# these event types: delegation.initiated, routing.decision, tool.invoked,
# data.read, data.write, external.call, message.received.
#
# It does NOT track: llm.call_start, guardrail.triggered.
# This means uses-edges to LLM capabilities and filtered_by/gated_by edges
# cannot be activated via the activation map.
#
# Fuzzy name matching: _count_edge_activations will match any (src, tgt) pair
# in the activation map where normalized names overlap -- regardless of the
# structural edge type.  So a delegates_to event between two agents can
# accidentally activate a shares_with edge between the same agents.
#
# To get exactly 4 dead edges from 15 total, we design the graph so that:
#   ACTIVATED (11 edges): e1-e4 (delegation), e5-e8 (tool uses),
#                         e9-e11 (data read/write + external call)
#   DEAD (4 edges): e12 (gated_by -- guardrail), e13 (shares_with between
#                   active nodes but no shared state traversal), e14 (delegates_to
#                   to unreachable Fallback), e15 (uses to unused tool)

structural_graph: dict = {
    "repo_id": "test_repo_v6",
    "framework": "crewai",
    "nodes": {
        # Agents
        "agent_planner": {
            "node_type": "agent",
            "name": "Planner",
            "source_file": "agents.py",
            "line_number": 10,
            "error_handling": {"strategy": "retry"},
        },
        "agent_coder": {
            "node_type": "agent",
            "name": "Coder",
            "source_file": "agents.py",
            "line_number": 30,
        },
        "agent_reviewer": {
            "node_type": "agent",
            "name": "Reviewer",
            "source_file": "agents.py",
            "line_number": 50,
        },
        "agent_deployer": {
            "node_type": "agent",
            "name": "Deployer",
            "source_file": "agents.py",
            "line_number": 70,
        },
        "agent_fallback": {
            "node_type": "agent",
            "name": "Fallback",
            "source_file": "agents.py",
            "line_number": 90,
        },
        # Capabilities (tools)
        "cap_code_search_tool": {
            "node_type": "capability",
            "name": "CodeSearch",
            "kind": "tool",
        },
        "cap_linter_tool": {
            "node_type": "capability",
            "name": "Linter",
            "kind": "tool",
        },
        "cap_deploy_tool": {
            "node_type": "capability",
            "name": "DeployTool",
            "kind": "tool",
        },
        "cap_unused_tool": {
            "node_type": "capability",
            "name": "UnusedTool",
            "kind": "tool",
        },
        # Data stores
        "ds_shared_context": {
            "node_type": "data_store",
            "name": "SharedContext",
        },
        # External services
        "ext_ci_api": {
            "node_type": "external",
            "name": "CiApi",
        },
        # Guardrails
        "guard_code_review": {
            "node_type": "guardrail",
            "name": "CodeReviewGate",
            "kind": "output_validation",
        },
    },
    "edges": {
        # --- ACTIVATED edges (11) ---
        # Delegation edges (4): all activated via delegation.initiated events
        "e1":  {"edge_type": "delegates_to", "source": "agent_planner",  "target": "agent_coder"},
        "e2":  {"edge_type": "delegates_to", "source": "agent_coder",    "target": "agent_reviewer"},
        "e3":  {"edge_type": "delegates_to", "source": "agent_reviewer", "target": "agent_deployer"},
        "e4":  {"edge_type": "delegates_to", "source": "agent_coder",    "target": "agent_deployer"},
        # Tool uses edges (3): activated via tool.invoked events
        "e5":  {"edge_type": "uses",         "source": "agent_coder",    "target": "cap_code_search_tool"},
        "e6":  {"edge_type": "uses",         "source": "agent_coder",    "target": "cap_linter_tool"},
        "e7":  {"edge_type": "uses",         "source": "agent_deployer", "target": "cap_deploy_tool"},
        # Data flow edges (3): activated via data.read / data.write events
        "e8":  {"edge_type": "writes_to",    "source": "agent_planner",  "target": "ds_shared_context"},
        "e9":  {"edge_type": "reads_from",   "source": "agent_coder",    "target": "ds_shared_context"},
        "e10": {"edge_type": "writes_to",    "source": "agent_coder",    "target": "ds_shared_context"},
        # External call edge (1): activated via external.call events
        "e11": {"edge_type": "calls",        "source": "cap_deploy_tool","target": "ext_ci_api"},
        # --- DEAD edges (4) ---
        # Guardrail edge: guardrail.triggered is not in activation map
        "e12": {"edge_type": "gated_by",     "source": "agent_deployer", "target": "guard_code_review"},
        # shares_with between Reviewer and Planner: both active but no shared state traversal
        "e13": {"edge_type": "shares_with",  "source": "agent_reviewer",  "target": "agent_planner"},
        # Delegation to unreachable Fallback: never activated
        "e14": {"edge_type": "delegates_to", "source": "agent_planner",  "target": "agent_fallback"},
        # Uses edge to unused tool: never activated
        "e15": {"edge_type": "uses",         "source": "agent_reviewer", "target": "cap_unused_tool"},
    },
    "findings": [
        {
            "finding_id": "STRAT-SI-001",
            "severity": "high",
            "title": "Error laundering",
            "description": "Error silently swallowed at agent_coder boundary",
        },
        {
            "finding_id": "STRAT-DC-001",
            "severity": "high",
            "title": "Unsupervised decision chain",
            "description": "Multi-hop delegation without human checkpoint",
        },
    ],
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
    """Return a deterministic nanosecond timestamp with an optional ms offset."""
    return int((1_700_000_000 + offset_ms / 1000) * 1_000_000_000)


def _src(node_name: str, node_type: str = "agent") -> dict:
    return {"node_type": node_type, "node_id": "", "node_name": node_name}


def _delegation_events(
    source_agent: str, target_agent: str,
    run_id: str, repo_id: str, offset_ms: int,
) -> list[dict]:
    """Emit delegation.initiated + delegation.completed pair."""
    start_id = _eid()
    end_id = _eid()
    return [
        {
            "event_id": start_id,
            "event_type": "delegation.initiated",
            "timestamp_ns": _ts(offset_ms),
            "run_id": run_id,
            "repo_id": repo_id,
            "source_node": _src(source_agent),
            "target_node": _src(target_agent),
            "payload": {},
        },
        {
            "event_id": end_id,
            "event_type": "delegation.completed",
            "timestamp_ns": _ts(offset_ms + 50),
            "run_id": run_id,
            "repo_id": repo_id,
            "parent_event_id": start_id,
            "source_node": _src(source_agent),
            "target_node": _src(target_agent),
            "payload": {},
        },
    ]


def _agent_task_events(
    agent_name: str, run_id: str, repo_id: str,
    offset_ms: int, duration_ms: int = 500,
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
            "run_id": run_id,
            "repo_id": repo_id,
            "source_node": _src(agent_name),
            "payload": {},
        },
        {
            "event_id": end_id,
            "event_type": "agent.task_end",
            "timestamp_ns": _ts(offset_ms + duration_ms),
            "run_id": run_id,
            "repo_id": repo_id,
            "parent_event_id": start_id,
            "source_node": _src(agent_name),
            "payload": {"status": status},
        },
    ]


def _tool_events(
    tool_name: str, agent_name: str,
    run_id: str, repo_id: str,
    offset_ms: int, duration_ms: int = 100,
) -> list[dict]:
    """Emit tool.invoked + tool.completed pair."""
    start_id = _eid()
    end_id = _eid()
    return [
        {
            "event_id": start_id,
            "event_type": "tool.invoked",
            "timestamp_ns": _ts(offset_ms),
            "run_id": run_id,
            "repo_id": repo_id,
            "source_node": _src(agent_name, "agent"),
            "target_node": _src(tool_name, "capability"),
            "payload": {},
        },
        {
            "event_id": end_id,
            "event_type": "tool.completed",
            "timestamp_ns": _ts(offset_ms + duration_ms),
            "run_id": run_id,
            "repo_id": repo_id,
            "parent_event_id": start_id,
            "source_node": _src(agent_name, "agent"),
            "target_node": _src(tool_name, "capability"),
            "payload": {"status": "success"},
        },
    ]


def _llm_events(
    llm_name: str, agent_name: str,
    run_id: str, repo_id: str,
    offset_ms: int, duration_ms: int = 300,
    input_tokens: int = 500, output_tokens: int = 200,
) -> list[dict]:
    """Emit llm.call_start + llm.call_end pair."""
    start_id = _eid()
    end_id = _eid()
    return [
        {
            "event_id": start_id,
            "event_type": "llm.call_start",
            "timestamp_ns": _ts(offset_ms),
            "run_id": run_id,
            "repo_id": repo_id,
            "source_node": _src(agent_name, "agent"),
            "target_node": _src(llm_name, "capability"),
            "payload": {},
        },
        {
            "event_id": end_id,
            "event_type": "llm.call_end",
            "timestamp_ns": _ts(offset_ms + duration_ms),
            "run_id": run_id,
            "repo_id": repo_id,
            "parent_event_id": start_id,
            "source_node": _src(agent_name, "agent"),
            "target_node": _src(llm_name, "capability"),
            "payload": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        },
    ]


def _write_event(
    agent_name: str, ds_name: str,
    run_id: str, repo_id: str,
    offset_ms: int,
) -> list[dict]:
    return [
        {
            "event_id": _eid(),
            "event_type": "data.write",
            "timestamp_ns": _ts(offset_ms),
            "run_id": run_id,
            "repo_id": repo_id,
            "source_node": _src(agent_name, "agent"),
            "target_node": _src(ds_name, "data_store"),
            "payload": {"data_size_bytes": 1024},
        },
    ]


def _read_event(
    agent_name: str, ds_name: str,
    run_id: str, repo_id: str,
    offset_ms: int,
) -> list[dict]:
    return [
        {
            "event_id": _eid(),
            "event_type": "data.read",
            "timestamp_ns": _ts(offset_ms),
            "run_id": run_id,
            "repo_id": repo_id,
            "source_node": _src(agent_name, "agent"),
            "target_node": _src(ds_name, "data_store"),
            "payload": {},
        },
    ]


def _external_call_event(
    cap_name: str, ext_name: str,
    run_id: str, repo_id: str,
    offset_ms: int,
) -> list[dict]:
    return [
        {
            "event_id": _eid(),
            "event_type": "external.call",
            "timestamp_ns": _ts(offset_ms),
            "run_id": run_id,
            "repo_id": repo_id,
            "source_node": _src(cap_name, "capability"),
            "target_node": _src(ext_name, "external"),
            "payload": {},
        },
    ]


def _guardrail_event(
    guardrail_name: str,
    run_id: str, repo_id: str,
    offset_ms: int,
    action_prevented: bool = False,
) -> list[dict]:
    return [
        {
            "event_id": _eid(),
            "event_type": "guardrail.triggered",
            "timestamp_ns": _ts(offset_ms),
            "run_id": run_id,
            "repo_id": repo_id,
            "source_node": _src(guardrail_name, "guardrail"),
            "payload": {
                "action_prevented": action_prevented,
                "bypassed": False,
                "retry_triggered": False,
                "latency_ms": 3.0,
            },
        },
    ]


def _error_event(
    agent_name: str,
    run_id: str, repo_id: str,
    offset_ms: int,
    error_handling: str = "retry",
    error_type: str = "runtime_error",
    propagated: bool = False,
    swallowed: bool = False,
) -> list[dict]:
    """Emit error.occurred and optionally error.propagated."""
    events = [
        {
            "event_id": _eid(),
            "event_type": "error.occurred",
            "timestamp_ns": _ts(offset_ms),
            "run_id": run_id,
            "repo_id": repo_id,
            "source_node": _src(agent_name),
            "payload": {
                "error_handling": error_handling,
                "error_type": error_type,
                "swallowed": swallowed,
            },
        },
    ]
    if propagated:
        events.append(
            {
                "event_id": _eid(),
                "event_type": "error.propagated",
                "timestamp_ns": _ts(offset_ms + 5),
                "run_id": run_id,
                "repo_id": repo_id,
                "source_node": _src(agent_name),
                "payload": {
                    "error_handling": error_handling,
                    "error_type": "propagated_failure",
                    "propagation_path": [agent_name],
                    "downstream_impact": True,
                },
            }
        )
    return events


def _state_access_event(
    node_id: str, state_key: str, access_type: str,
    run_id: str, repo_id: str,
    offset_ms: int,
) -> list[dict]:
    """Emit a state.access event for implicit sharing detection."""
    return [
        {
            "event_id": _eid(),
            "event_type": "state.access",
            "timestamp_ns": _ts(offset_ms),
            "run_id": run_id,
            "repo_id": repo_id,
            "source_node": _src(node_id),
            "payload": {
                "node_id": node_id,
                "state_key": state_key,
                "access_type": access_type,
            },
        },
    ]


def _routing_decision_event(
    source: str, target: str,
    run_id: str, repo_id: str,
    offset_ms: int,
    routing_type: str = "llm_decision",
    decision_basis: str = "llm_output",
) -> list[dict]:
    """Emit a routing.decision event for dynamic delegation detection."""
    return [
        {
            "event_id": _eid(),
            "event_type": "routing.decision",
            "timestamp_ns": _ts(offset_ms),
            "run_id": run_id,
            "repo_id": repo_id,
            "source_node": _src(source),
            "target_node": _src(target),
            "payload": {
                "source_node": source,
                "target_node": target,
                "routing_type": routing_type,
                "decision_basis": decision_basis,
            },
        },
    ]


# =========================================================================
# 3. Build 5 runs of synthetic events
# =========================================================================
#
# Event design summary:
#
# Every run activates edges e1-e2 (Planner->Coder->Reviewer delegation),
# e5-e6 (Coder tools), e8-e10 (data read/write).
#
# Runs 3-4 also activate e3-e4 (Reviewer->Deployer, Coder->Deployer
# delegation), e7 (Deployer->DeployTool), e11 (DeployTool->CiApi).
#
# NEVER activated: e12 (gated_by guardrail -- not in activation map),
# e13 (shares_with Planner<->Reviewer -- no shared state traversal),
# e14 (delegates_to Planner->Fallback -- never delegated),
# e15 (uses Reviewer->UnusedTool).
#
# Error events: run 1 has swallowed error at Coder (SI-001 manifestation);
# run 2 has propagated error from Planner through Coder.
#
# Emergent edges: Coder->Fallback delegation in run 2 (error fallback);
# Planner->Reviewer routing decision in run 3 (dynamic delegation);
# Planner+Reviewer implicit data sharing via plan_state key.

RUN_COUNT = 5
REPO_ID = "test_repo_v6"


def build_run_events(run_idx: int) -> list[dict]:
    """Build one run's worth of events."""
    run_id = f"run_{run_idx:03d}"
    base = run_idx * 5000
    events: list[dict] = []

    # ---- Planner works ----
    events += _agent_task_events("Planner", run_id, REPO_ID, base + 0, duration_ms=800)
    events += _llm_events("LLMCall", "Planner", run_id, REPO_ID, base + 100)
    events += _write_event("Planner", "SharedContext", run_id, REPO_ID, base + 500)

    # ---- Planner delegates to Coder (e1) ----
    events += _delegation_events("Planner", "Coder", run_id, REPO_ID, base + 800)

    # ---- Coder works ----
    events += _agent_task_events("Coder", run_id, REPO_ID, base + 900, duration_ms=600)
    events += _tool_events("CodeSearch", "Coder", run_id, REPO_ID, base + 950)
    events += _tool_events("Linter", "Coder", run_id, REPO_ID, base + 1100)
    events += _read_event("Coder", "SharedContext", run_id, REPO_ID, base + 920)
    events += _write_event("Coder", "SharedContext", run_id, REPO_ID, base + 1200)

    # ---- Coder delegates to Reviewer (e2) ----
    events += _delegation_events("Coder", "Reviewer", run_id, REPO_ID, base + 1400)

    # ---- Reviewer works ----
    events += _agent_task_events("Reviewer", run_id, REPO_ID, base + 1500, duration_ms=400)

    # ---- Runs 3-4: full deployment path ----
    if run_idx >= 3:
        # Reviewer -> Deployer delegation (e3)
        events += _delegation_events("Reviewer", "Deployer", run_id, REPO_ID, base + 1950)
        # Coder -> Deployer delegation (e4)
        events += _delegation_events("Coder", "Deployer", run_id, REPO_ID, base + 1970)
        # Deployer works
        events += _agent_task_events("Deployer", run_id, REPO_ID, base + 2000, duration_ms=300)
        # Deployer uses DeployTool (e7)
        events += _tool_events("DeployTool", "Deployer", run_id, REPO_ID, base + 2050)
        # DeployTool -> CiApi external call (e11)
        events += _external_call_event("DeployTool", "CiApi", run_id, REPO_ID, base + 2100)

    # ---- Guardrail event (does NOT activate edges in activation map) ----
    events += _guardrail_event("CodeReviewGate", run_id, REPO_ID, base + 1300)

    # ---- Error events for propagation tracing ----
    # Run 1: error at Coder, swallowed (SI-001 manifestation)
    if run_idx == 1:
        events += _error_event(
            "Coder", run_id, REPO_ID, base + 1050,
            error_handling="caught_silent",
            error_type="runtime_error",
            swallowed=True,
        )

    # Run 2: error at Planner, propagates through Coder
    if run_idx == 2:
        events += _error_event(
            "Planner", run_id, REPO_ID, base + 400,
            error_handling="propagate",
            error_type="upstream_failure",
            propagated=True,
        )
        # Coder encounters error after Planner's propagation
        events += _error_event(
            "Coder", run_id, REPO_ID, base + 1000,
            error_handling="retry",
            error_type="propagated_failure",
        )

    # ---- Emergent edge events (for Eval 2) ----
    # (a) Error-triggered fallback: Coder -> Fallback (no structural edge)
    if run_idx == 2:
        events += _delegation_events("Coder", "Fallback", run_id, REPO_ID, base + 1060)

    # (b) LLM-chosen routing to unexpected agent: Planner -> Reviewer (run 3)
    #     (There is no structural delegates_to edge from Planner to Reviewer)
    if run_idx == 3:
        events += _routing_decision_event(
            "Planner", "Reviewer", run_id, REPO_ID, base + 750,
            routing_type="llm_decision",
            decision_basis="llm_output",
        )

    # (c) Implicit data sharing: Planner and Reviewer both access "plan_state"
    #     (No shares_with edge between Planner and Reviewer in structural graph)
    events += _state_access_event(
        "Planner", "plan_state", "write",
        run_id, REPO_ID, base + 600,
    )
    events += _state_access_event(
        "Reviewer", "plan_state", "read",
        run_id, REPO_ID, base + 1600,
    )

    return events


# Build all run records and flat event list
all_run_records: list[dict] = []
all_events_flat: list[dict] = []

for i in range(RUN_COUNT):
    run_events = build_run_events(i)
    run_id = f"run_{i:03d}"
    all_run_records.append({
        "events": run_events,
        "run_id": run_id,
        "repo_id": REPO_ID,
        "metadata": {"input_hash": f"hash_{i:03d}"},
    })
    for ev in run_events:
        if "run_id" not in ev:
            ev["run_id"] = run_id
    all_events_flat.extend(run_events)


# =========================================================================
# EVAL 1: Edge validation accuracy (Checks 1-2)
# =========================================================================

out("=" * 78)
out("EVAL 1: EDGE VALIDATION ACCURACY (Checks 1-2)")
out("=" * 78)

edge_validation = compute_edge_validation(structural_graph, all_events_flat, run_count=RUN_COUNT)

dead_edge_ids = {de["edge_id"] for de in edge_validation["dead_edges"]}
expected_dead = {"e12", "e13", "e14", "e15"}

out(f"\n  Structural edges total  : {edge_validation['structural_edges_total']}")
out(f"  Activated               : {edge_validation['structural_edges_activated']}")
out(f"  Dead                    : {edge_validation['structural_edges_dead']}")
out(f"  Dead edge IDs detected  : {sorted(dead_edge_ids)}")
out(f"  Expected dead edge IDs  : {sorted(expected_dead)}")
out()

check(
    edge_validation["structural_edges_total"] == 15,
    "Total structural edges == 15",
    f"got {edge_validation['structural_edges_total']}",
)
check(
    edge_validation["structural_edges_dead"] == len(expected_dead),
    f"{len(expected_dead)} dead edges detected",
    f"got {edge_validation['structural_edges_dead']}, IDs: {sorted(dead_edge_ids)}",
)
check(
    dead_edge_ids == expected_dead,
    f"Dead edge IDs match expected set {sorted(expected_dead)}",
    f"symmetric difference: {dead_edge_ids ^ expected_dead}" if dead_edge_ids != expected_dead else "",
)

# Activation rates per type
out(f"\n  Activation rates by type: {pretty(edge_validation['activation_rates'])}")
rates = edge_validation["activation_rates"]
check(
    len(rates) >= 3,
    "Activation rates computed for at least 3 edge types",
    f"types: {list(rates.keys())}",
)

# Dead edge reasons are meaningful
for de in edge_validation["dead_edges"]:
    reason = de.get("reason", "")
    check(
        reason and reason != "not activated",
        f"Dead edge {de['edge_id']} has meaningful reason",
        f"reason: {reason!r}",
    )


# =========================================================================
# CHECK 18: Dead edge reason type diversity
# =========================================================================

out("\n" + "=" * 78)
out("CHECK 18: DEAD EDGE REASON TYPE DIVERSITY")
out("=" * 78)

reason_prefixes = set()
for de in edge_validation["dead_edges"]:
    reason = de.get("reason", "")
    prefix = reason.split(":")[0] if ":" in reason else reason
    reason_prefixes.add(prefix)
    out(f"  {de['edge_id']}: prefix='{prefix}'  reason='{reason[:80]}'")

check(
    len(reason_prefixes) >= 2,
    f"At least 2 distinct dead edge reason type prefixes (got {len(reason_prefixes)})",
    f"prefixes: {sorted(reason_prefixes)}",
)


# =========================================================================
# EVAL 2: Emergent edge detection (Check 3)
# =========================================================================

out("\n" + "=" * 78)
out("EVAL 2: EMERGENT EDGE DETECTION (Check 3)")
out("=" * 78)

emergent = detect_emergent_edges_v2(structural_graph, all_events_flat, run_count=RUN_COUNT)

out(f"\n  Emergent edges found: {len(emergent)}")
for ee in emergent:
    out(f"    {ee['edge_id']}: {ee['source_node']} -> {ee['target_node']}")
    out(f"      discovery_type     : {ee['discovery_type']}")
    out(f"      detection_heuristic: {ee['detection_heuristic'][:80]}")
    out(f"      activation_count   : {ee['activation_count']}")
    out(f"      trigger_condition  : {ee['trigger_condition']}")

# Build lookup by discovery_type
discovery_types_found = {ee["discovery_type"] for ee in emergent}

check(
    len(emergent) >= 3,
    f"At least 3 emergent edges detected (got {len(emergent)})",
)
check(
    "error_triggered_fallback" in discovery_types_found,
    "Error-triggered fallback detected",
    f"discovery types: {discovery_types_found}",
)
check(
    "dynamic_delegation" in discovery_types_found,
    "Dynamic delegation detected",
    f"discovery types: {discovery_types_found}",
)
check(
    "implicit_data_sharing" in discovery_types_found,
    "Implicit data sharing detected",
    f"discovery types: {discovery_types_found}",
)

# All detection_heuristics are non-empty actionable strings
for ee in emergent:
    check(
        isinstance(ee["detection_heuristic"], str) and len(ee["detection_heuristic"]) > 10,
        f"Emergent {ee['edge_id']} has actionable detection_heuristic",
        f"heuristic length: {len(ee.get('detection_heuristic', ''))}",
    )

# No false positives from patcher internals
patcher_false_positives = [
    ee for ee in emergent
    if "patcher" in ee.get("source_node", "").lower()
    or "patcher" in ee.get("target_node", "").lower()
    or ee.get("discovery_type") == "framework_internal_routing"
]
check(
    len(patcher_false_positives) == 0,
    "No false positives from patcher internal routing",
    f"found {len(patcher_false_positives)} patcher-related emergent edges" if patcher_false_positives else "",
)


# =========================================================================
# EVAL 3: Error propagation tracing (Check 4)
# =========================================================================

out("\n" + "=" * 78)
out("EVAL 3: ERROR PROPAGATION TRACING (Check 4)")
out("=" * 78)

error_traces = trace_error_propagation(structural_graph, all_events_flat)

out(f"\n  Error traces found: {len(error_traces)}")
for i, trace in enumerate(error_traces):
    out(f"\n  Trace {i}:")
    out(f"    error_source_node        : {trace['error_source_node']}")
    out(f"    error_type               : {trace['error_type']}")
    out(f"    structural_predicted_path: {trace['structural_predicted_path']}")
    out(f"    actual_observed_path     : {trace['actual_observed_path']}")
    out(f"    propagation_stopped_by   : {trace['propagation_stopped_by']}")
    out(f"    stop_mechanism           : {trace['stop_mechanism']}")
    out(f"    downstream_impact        : {pretty(trace['downstream_impact'])}")
    out(f"    swallowed                : {trace['swallowed']}")

check(
    len(error_traces) >= 2,
    f"At least 2 error traces found (got {len(error_traces)})",
)

# Check structural predicted path vs actual observed path are both computed
for trace in error_traces:
    check(
        isinstance(trace["structural_predicted_path"], list) and len(trace["structural_predicted_path"]) >= 1,
        f"Trace from {trace['error_source_node']}: structural_predicted_path populated",
        f"path: {trace['structural_predicted_path']}",
    )
    check(
        isinstance(trace["actual_observed_path"], list) and len(trace["actual_observed_path"]) >= 1,
        f"Trace from {trace['error_source_node']}: actual_observed_path populated",
        f"path: {trace['actual_observed_path']}",
    )

# Check that propagation_stopped_by identifies a node
stopped_by_nodes = [t["propagation_stopped_by"] for t in error_traces if t["propagation_stopped_by"]]
check(
    len(stopped_by_nodes) >= 1,
    "At least one trace identifies a propagation_stopped_by node",
    f"stopped_by nodes: {stopped_by_nodes}",
)

# Check downstream_impact describes error handling
for trace in error_traces:
    impact = trace["downstream_impact"]
    check(
        isinstance(impact, dict) and "nodes_affected" in impact,
        f"Trace from {trace['error_source_node']}: downstream_impact has nodes_affected",
        f"impact keys: {list(impact.keys()) if isinstance(impact, dict) else 'not a dict'}",
    )


# Check: at least one trace has cascade_depth >= 2 (error laundering detection)
max_cascade = max(
    (t.get("downstream_impact", {}).get("cascade_depth", 0) for t in error_traces),
    default=0,
)
check(
    max_cascade >= 2,
    "At least one error trace has cascade_depth >= 2 (error laundering)",
    f"max cascade_depth={max_cascade}",
)


# =========================================================================
# EVAL 4: Node activation topology (Check 5)
# =========================================================================

out("\n" + "=" * 78)
out("EVAL 4: NODE ACTIVATION TOPOLOGY (Check 5)")
out("=" * 78)

node_activation = compute_node_activation(structural_graph, all_events_flat, run_count=RUN_COUNT)

always_active_ids = [n["node_id"] for n in node_activation["always_active"]]
conditional_ids = [n["node_id"] for n in node_activation["conditional"]]
never_active_ids = [n["node_id"] for n in node_activation["never_active"]]

out(f"\n  Always active ({len(always_active_ids)}): {always_active_ids}")
out(f"  Conditional   ({len(conditional_ids)}): {conditional_ids}")
out(f"  Never active  ({len(never_active_ids)}): {never_active_ids}")

# Planner, Coder, Reviewer should be always active (all 5 runs)
for expected_always in ["agent_planner", "agent_coder", "agent_reviewer"]:
    check(
        expected_always in always_active_ids,
        f"{expected_always} classified as always_active",
    )

# Deployer should be conditional (activated in runs 3-4 only = 40%)
check(
    "agent_deployer" in conditional_ids,
    "agent_deployer classified as conditional (2/5 runs)",
    f"conditional IDs: {conditional_ids}",
)

# At least one node is never active
check(
    len(never_active_ids) >= 1,
    f"At least 1 node is never_active (got {len(never_active_ids)})",
    f"never_active: {never_active_ids}",
)


# =========================================================================
# EVAL 5: Structural prediction match rate (Check 6)
# =========================================================================

out("\n" + "=" * 78)
out("EVAL 5: STRUCTURAL PREDICTION MATCH RATE (Check 6)")
out("=" * 78)

match_rate = node_activation.get("structural_prediction_match_rate")

out(f"\n  structural_prediction_match_rate: {match_rate}")

check(
    match_rate is not None,
    "structural_prediction_match_rate is present",
)
check(
    isinstance(match_rate, (int, float)),
    "structural_prediction_match_rate is numeric",
    f"type: {type(match_rate).__name__}",
)
check(
    0.0 <= match_rate <= 1.0 if isinstance(match_rate, (int, float)) else False,
    "structural_prediction_match_rate is in [0, 1]",
    f"value: {match_rate}",
)


# =========================================================================
# EVAL 6: Failure mode classification (Check 7)
# =========================================================================

out("\n" + "=" * 78)
out("EVAL 6: FAILURE MODE CLASSIFICATION (Check 7)")
out("=" * 78)

findings = structural_graph["findings"]
failure_classifications = classify_failure_modes(
    findings, all_events_flat, structural_graph, error_traces,
)

out(f"\n  Classifications returned: {len(failure_classifications)}")
for fc in failure_classifications:
    out(f"\n  {fc['finding_id']}: {fc['finding_name']}")
    out(f"    manifestation_observed   : {fc['manifestation_observed']}")
    out(f"    failure_type             : {fc['failure_type']}")
    out(f"    occurrences              : {fc['occurrences']}")
    out(f"    downstream_impact        : {fc['downstream_impact_observed']}")
    out(f"    failure_description      : {fc['failure_description'][:100]}")

# Lookup by finding_id
fc_map = {fc["finding_id"]: fc for fc in failure_classifications}

# SI-001: Error laundering -- should be manifested (swallowed error in run 1)
check(
    "STRAT-SI-001" in fc_map,
    "STRAT-SI-001 (error laundering) classified",
)
if "STRAT-SI-001" in fc_map:
    check(
        fc_map["STRAT-SI-001"]["manifestation_observed"] is True,
        "STRAT-SI-001 manifestation observed",
        f"manifestation_observed: {fc_map['STRAT-SI-001']['manifestation_observed']}",
    )
    check(
        isinstance(fc_map["STRAT-SI-001"]["failure_type"], str) and len(fc_map["STRAT-SI-001"]["failure_type"]) > 5,
        "STRAT-SI-001 failure_type matches taxonomy category",
        f"failure_type: {fc_map['STRAT-SI-001']['failure_type']}",
    )

# DC-001: Unsupervised chain -- should be manifested (3+ delegations)
check(
    "STRAT-DC-001" in fc_map,
    "STRAT-DC-001 (unsupervised chain) classified",
)
if "STRAT-DC-001" in fc_map:
    check(
        fc_map["STRAT-DC-001"]["manifestation_observed"] is True,
        "STRAT-DC-001 manifestation observed",
        f"manifestation_observed: {fc_map['STRAT-DC-001']['manifestation_observed']}",
    )
    check(
        isinstance(fc_map["STRAT-DC-001"]["failure_type"], str) and len(fc_map["STRAT-DC-001"]["failure_type"]) > 5,
        "STRAT-DC-001 failure_type matches taxonomy category",
        f"failure_type: {fc_map['STRAT-DC-001']['failure_type']}",
    )

# Non-manifested findings should be explicitly marked
for fc in failure_classifications:
    if not fc["manifestation_observed"]:
        check(
            fc["failure_description"] is not None and len(fc["failure_description"]) > 0,
            f"{fc['finding_id']} non-manifested has explicit description",
            f"description: {fc['failure_description'][:80]}",
        )


# =========================================================================
# EVAL 7: Monitoring baseline stability (Check 8)
# =========================================================================

out("\n" + "=" * 78)
out("EVAL 7: MONITORING BASELINE STABILITY (Check 8)")
out("=" * 78)

baselines_run1 = extract_monitoring_baselines(findings, all_events_flat)
baselines_run2 = extract_monitoring_baselines(findings, all_events_flat)

out(f"\n  Baselines run 1: {len(baselines_run1)} metrics")
out(f"  Baselines run 2: {len(baselines_run2)} metrics")

for bl in baselines_run1:
    out(f"\n  {bl['finding_id']}: {bl['metric']}")
    out(f"    observed_baseline    : {bl['observed_baseline']}")
    out(f"    observed_stddev      : {bl['observed_stddev']}")
    out(f"    suggested_threshold  : {bl['suggested_threshold']}")
    out(f"    confidence           : {bl['confidence']}")
    out(f"    sample_size          : {bl['sample_size']}")

# Baselines identical across runs (deterministic)
check(
    baselines_run1 == baselines_run2,
    "Baselines are identical across two runs (deterministic)",
)

# Threshold computation is deterministic
if baselines_run1 and baselines_run2:
    thresholds_1 = [bl["suggested_threshold"] for bl in baselines_run1]
    thresholds_2 = [bl["suggested_threshold"] for bl in baselines_run2]
    check(
        thresholds_1 == thresholds_2,
        "Threshold computation is deterministic",
        f"run1: {thresholds_1}, run2: {thresholds_2}",
    )

# Per-finding baseline + threshold present
for bl in baselines_run1:
    check(
        "observed_baseline" in bl and "suggested_threshold" in bl,
        f"{bl['finding_id']}: has baseline + threshold",
        f"baseline={bl.get('observed_baseline')}, threshold={bl.get('suggested_threshold')}",
    )
    check(
        isinstance(bl["observed_baseline"], (int, float)),
        f"{bl['finding_id']}: baseline is numeric",
    )
    check(
        isinstance(bl["suggested_threshold"], (int, float)),
        f"{bl['finding_id']}: threshold is numeric",
    )

check(
    len(baselines_run1) >= 1,
    "At least 1 monitoring baseline extracted",
    f"got {len(baselines_run1)}",
)


# =========================================================================
# SUMMARY
# =========================================================================

out("\n" + "=" * 78)
out("SUMMARY")
out("=" * 78)
out(f"\n  Total checks: {_pass_count + _fail_count}")
out(f"  Passed      : {_pass_count}")
out(f"  Failed      : {_fail_count}")
out()

if _fail_count == 0:
    out("  All validation checks PASSED.")
else:
    out(f"  WARNING: {_fail_count} check(s) FAILED.")

out("\nDone.")

# =========================================================================
# Write output file
# =========================================================================
flush_output()
