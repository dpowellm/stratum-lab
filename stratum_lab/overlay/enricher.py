"""Graph enrichment engine.

Takes a structural graph and runtime event data, computes behavioral
overlays for every node and edge, and returns an EnrichedGraph dict.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from stratum_lab.node_ids import match_runtime_to_structural


# ---------------------------------------------------------------------------
# Node behavioral overlay
# ---------------------------------------------------------------------------

def compute_node_behavioral_overlay(
    node_id: str,
    events: list[dict[str, Any]],
    total_runs: int,
) -> dict[str, Any]:
    """Compute behavioral statistics for one structural node.

    Parameters
    ----------
    node_id:
        The structural graph node ID.
    events:
        Events that have been matched to this node.
    total_runs:
        Total number of runs observed for normalization.

    Returns
    -------
    A behavioral overlay dict with activation, throughput, latency,
    error_behavior, decision_behavior, model_sensitivity, and resource_usage.
    """
    total_runs = max(total_runs, 1)

    # ---- Activation ----
    # Event types emitted by stratum_patcher use dot notation:
    #   agent.task_start, tool.invoked, llm.call_start, etc.
    START_TYPES = {
        "agent.task_start", "tool.invoked", "llm.call_start",
        "execution.start", "message.received", "reply.generation_start",
    }
    END_TYPES = {
        "agent.task_end", "tool.completed", "llm.call_end",
        "execution.end", "reply.generated",
    }
    ERROR_TYPES = {
        "error.occurred", "error.propagated", "error.cascade",
        "tool.call_failure",
    }

    activation_events = [
        e for e in events if e.get("event_type") in START_TYPES
    ]
    activation_count = len(activation_events)
    activation_rate = activation_count / total_runs

    # ---- Throughput ----
    tasks_received = len([
        e for e in events if e.get("event_type") in START_TYPES
    ])
    tasks_completed = len([
        e for e in events if e.get("event_type") in END_TYPES
    ])
    tasks_failed = len([
        e for e in events if e.get("event_type") in ERROR_TYPES
    ])
    failure_rate = tasks_failed / max(tasks_received, 1)

    throughput = {
        "tasks_received": tasks_received,
        "completed": tasks_completed,
        "failed": tasks_failed,
        "failure_rate": round(failure_rate, 4),
    }

    # ---- Latency ----
    latencies_ns = _compute_latencies(events)
    if latencies_ns:
        arr = np.array(latencies_ns, dtype=np.float64)
        ms_arr = arr / 1_000_000
        latency = {
            "p50": round(float(np.percentile(ms_arr, 50)), 2),
            "p95": round(float(np.percentile(ms_arr, 95)), 2),
            "p99": round(float(np.percentile(ms_arr, 99)), 2),
            "variance": round(float(np.var(ms_arr)), 2),
            "sample_count": len(latencies_ns),
        }
    else:
        latency = {"p50": 0.0, "p95": 0.0, "p99": 0.0, "variance": 0.0, "sample_count": 0}

    # ---- Error behavior ----
    error_events = [e for e in events if e.get("event_type") in ERROR_TYPES]
    # Also include end events that report failure status
    error_events += [
        e for e in events
        if e.get("event_type") in END_TYPES
        and e.get("payload", {}).get("status") in ("error", "failure", "crash")
    ]
    errors_occurred = len(error_events)
    propagated_downstream = len([
        e for e in error_events if e.get("event_type") == "error.propagated"
    ])
    swallowed = errors_occurred - propagated_downstream
    propagation_rate = propagated_downstream / max(errors_occurred, 1)

    # Check for default values used (from payloads)
    default_values_used = sum(
        1 for e in error_events
        if e.get("payload", {}).get("default_value_used", False)
    )

    # Observed error handling from payload hints
    observed_handling: set[str] = set()
    for e in error_events:
        handling = e.get("payload", {}).get("error_handling")
        if handling:
            observed_handling.add(handling)

    error_behavior = {
        "errors_occurred": errors_occurred,
        "propagated_downstream": propagated_downstream,
        "swallowed": swallowed,
        "propagation_rate": round(propagation_rate, 4),
        "default_values_used": default_values_used,
        "observed_error_handling": sorted(observed_handling),
        "structural_prediction_match": None,  # Set during enrichment
    }

    # ---- Decision behavior ----
    decision_events = [
        e for e in events if e.get("event_type") == "decision"
    ]
    decisions_made = len(decision_events)

    # Shannon entropy of decision outcomes
    outcome_counts: Counter[str] = Counter()
    for e in decision_events:
        outcome = e.get("payload", {}).get("outcome", "unknown")
        outcome_counts[outcome] += 1

    decision_entropy = _shannon_entropy(outcome_counts)

    # Consistency: group decisions by input hash
    input_groups: dict[str, list[str]] = defaultdict(list)
    for e in decision_events:
        input_hash = e.get("payload", {}).get("input_hash", "")
        outcome = e.get("payload", {}).get("outcome", "unknown")
        if input_hash:
            input_groups[input_hash].append(outcome)

    # Consistency across same input: fraction of input groups with identical outcomes
    same_input_consistent = 0
    same_input_total = 0
    for outcomes in input_groups.values():
        if len(outcomes) >= 2:
            same_input_total += 1
            if len(set(outcomes)) == 1:
                same_input_consistent += 1

    consistency_same_input = (
        same_input_consistent / same_input_total if same_input_total > 0 else None
    )

    # Consistency across different inputs: fraction of unique outcomes
    all_outcomes = [
        e.get("payload", {}).get("outcome", "unknown") for e in decision_events
    ]
    unique_outcomes = len(set(all_outcomes)) if all_outcomes else 0
    consistency_diff_input = unique_outcomes / max(len(all_outcomes), 1) if all_outcomes else None

    decision_behavior = {
        "decisions_made": decisions_made,
        "decision_entropy": round(decision_entropy, 4),
        "consistency_across_same_input": consistency_same_input,
        "consistency_across_diff_input": consistency_diff_input,
    } if decisions_made > 0 else None

    # ---- Model sensitivity ----
    TOOL_CALL_TYPES = {"tool.invoked", "tool.completed", "tool.call_failure"}
    tool_call_events = [e for e in events if e.get("event_type") in TOOL_CALL_TYPES]
    tool_call_failures = len([
        e for e in tool_call_events
        if e.get("event_type") == "tool.call_failure"
        or (e.get("event_type") == "tool.completed"
            and e.get("payload", {}).get("status") in ("error", "failure"))
    ])
    tool_call_total = len([
        e for e in tool_call_events if e.get("event_type") == "tool.invoked"
    ])
    tool_call_failure_rate = tool_call_failures / max(tool_call_total, 1)

    retry_activations = len([
        e for e in events if e.get("payload", {}).get("is_retry", False)
    ])

    # Quality-dependent if failure rate is high or there are retries
    quality_dependent = tool_call_failure_rate > 0.1 or retry_activations > 0

    model_sensitivity = {
        "tool_call_failures": tool_call_failures,
        "tool_call_failure_rate": round(tool_call_failure_rate, 4),
        "retry_activations": retry_activations,
        "quality_dependent": quality_dependent,
    }

    # ---- Resource usage ----
    tool_start_events = [e for e in events if e.get("event_type") == "tool.invoked"]
    llm_start_events = [e for e in events if e.get("event_type") == "llm.call_start"]

    token_counts = []
    for e in events:
        if e.get("event_type") == "llm.call_end":
            payload = e.get("payload", {})
            input_t = payload.get("input_tokens", 0) or 0
            output_t = payload.get("output_tokens", 0) or 0
            token_counts.append(input_t + output_t)
    avg_tokens = sum(token_counts) / max(len(token_counts), 1) if token_counts else 0.0

    # Count iterations (agent.task_start events as proxy)
    iteration_events = [e for e in events if e.get("event_type") == "agent.task_start"]

    resource_usage = {
        "avg_tokens": round(avg_tokens, 2),
        "avg_tool_calls": round(len(tool_start_events) / total_runs, 2),
        "avg_llm_calls": round(len(llm_start_events) / total_runs, 2),
        "avg_iterations": round(len(iteration_events) / total_runs, 2),
    }

    return {
        "activation_count": activation_count,
        "activation_rate": round(activation_rate, 4),
        "throughput": throughput,
        "latency": latency,
        "error_behavior": error_behavior,
        "decision_behavior": decision_behavior,
        "model_sensitivity": model_sensitivity,
        "resource_usage": resource_usage,
    }


def _compute_latencies(events: list[dict[str, Any]]) -> list[int]:
    """Compute start-to-end latencies in nanoseconds from paired events.

    Pairs start/end events using event_id -> parent_event_id linkage.
    Also extracts explicit latency_ms from payloads.
    """
    starts: dict[str, int] = {}
    latencies: list[int] = []

    _START = {
        "agent.task_start", "tool.invoked", "llm.call_start",
        "execution.start", "reply.generation_start",
    }
    _END = {
        "agent.task_end", "tool.completed", "llm.call_end",
        "execution.end", "reply.generated",
    }

    for ev in events:
        etype = ev.get("event_type", "")
        eid = ev.get("event_id", "")
        ts = ev.get("timestamp_ns", 0)

        if etype in _START:
            starts[eid] = ts
        elif etype in _END:
            # Try parent_event_id linkage first
            parent = ev.get("parent_event_id")
            if parent and parent in starts:
                latency = ts - starts[parent]
                if latency >= 0:
                    latencies.append(latency)
            else:
                # Fall back to explicit latency_ms in payload
                lat_ms = ev.get("payload", {}).get("latency_ms")
                if lat_ms is not None and lat_ms > 0:
                    latencies.append(int(lat_ms * 1_000_000))  # ms -> ns

    return latencies


def _shannon_entropy(counts: Counter[str]) -> float:
    """Compute Shannon entropy of a distribution given outcome counts."""
    total = sum(counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy


# ---------------------------------------------------------------------------
# Edge behavioral overlay
# ---------------------------------------------------------------------------

def compute_edge_behavioral_overlay(
    edge_id: str,
    events: list[dict[str, Any]],
    total_runs: int,
) -> dict[str, Any]:
    """Compute behavioral statistics for one structural edge.

    Parameters
    ----------
    edge_id:
        The structural graph edge ID.
    events:
        Events that have been matched to this edge (traversal events).
    total_runs:
        Total number of runs for normalization.

    Returns
    -------
    A behavioral overlay dict with traversal, data_flow, error_crossings,
    latency_contribution_ms, and conditional_behavior.
    """
    total_runs = max(total_runs, 1)

    traversal_count = len(events)
    activation_rate = traversal_count / total_runs
    never_activated = traversal_count == 0

    # ---- Data flow ----
    data_sizes = [
        e.get("payload", {}).get("data_size_bytes", 0) for e in events
        if "data_size_bytes" in e.get("payload", {})
    ]
    avg_data_size = sum(data_sizes) / max(len(data_sizes), 1) if data_sizes else 0.0

    schema_mismatch = sum(
        1 for e in events
        if e.get("payload", {}).get("schema_mismatch", False)
    )
    null_data = sum(
        1 for e in events
        if e.get("payload", {}).get("data") is None
        or e.get("payload", {}).get("null_data", False)
    )

    data_flow = {
        "avg_data_size_bytes": round(avg_data_size, 2),
        "schema_mismatch_count": schema_mismatch,
        "null_data_count": null_data,
    }

    # ---- Error crossings ----
    error_traversals = [
        e for e in events
        if e.get("payload", {}).get("carries_error", False)
        or e.get("event_type") in ("error_propagation",)
    ]
    error_types_seen: set[str] = set()
    for e in error_traversals:
        etype = e.get("payload", {}).get("error_type")
        if etype:
            error_types_seen.add(etype)

    downstream_impact = sum(
        1 for e in error_traversals
        if e.get("payload", {}).get("downstream_impact", False)
    )

    error_crossings = {
        "errors_traversed": len(error_traversals),
        "error_types": sorted(error_types_seen),
        "downstream_impact": downstream_impact,
    }

    # ---- Latency contribution ----
    latency_values = [
        e.get("payload", {}).get("latency_ms", 0.0) for e in events
        if "latency_ms" in e.get("payload", {})
    ]
    latency_contribution_ms = (
        sum(latency_values) / len(latency_values) if latency_values else 0.0
    )

    # ---- Conditional behavior ----
    condition_evals = [
        e for e in events if "condition_result" in e.get("payload", {})
    ]
    if condition_evals:
        true_count = sum(
            1 for e in condition_evals
            if e.get("payload", {}).get("condition_result", False)
        )
        false_count = len(condition_evals) - true_count
        total_evals = len(condition_evals)
        conditional_behavior = {
            "condition_true_rate": round(true_count / total_evals, 4),
            "condition_false_rate": round(false_count / total_evals, 4),
        }
    else:
        conditional_behavior = None

    return {
        "traversal_count": traversal_count,
        "activation_rate": round(activation_rate, 4),
        "never_activated": never_activated,
        "data_flow": data_flow,
        "error_crossings": error_crossings,
        "latency_contribution_ms": round(latency_contribution_ms, 2),
        "conditional_behavior": conditional_behavior,
    }


# ---------------------------------------------------------------------------
# Main enrichment function
# ---------------------------------------------------------------------------

def enrich_graph(
    structural_graph: dict[str, Any],
    run_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Enrich a structural graph with behavioral data from runtime events.

    Parameters
    ----------
    structural_graph:
        A structural graph dict with ``nodes`` and ``edges`` dicts.
        Each node: {node_id: {node_type, name, source_file, ...}}.
        Each edge: {edge_id: {edge_type, source, target, ...}}.
    run_records:
        List of RunRecord dicts for the same repo, each containing
        the original events or summary data.

    Returns
    -------
    An EnrichedGraph dict with behavioral overlays on each node and edge.
    """
    structural_nodes = structural_graph.get("nodes", {})
    structural_edges = structural_graph.get("edges", {})
    repo_id = structural_graph.get("repo_id", "unknown")
    framework = structural_graph.get("framework", "unknown")
    total_runs = len(run_records) if run_records else 1

    # Collect all events from run records
    all_events = _collect_all_events(run_records)

    # ---- Map runtime events to structural nodes ----
    node_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unmapped_node_events: list[dict[str, Any]] = []

    for ev in all_events:
        matched = _match_event_to_node(ev, structural_nodes)
        if matched:
            node_events[matched].append(ev)
        else:
            unmapped_node_events.append(ev)

    # ---- Map runtime events to structural edges ----
    edge_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unmapped_edge_events: list[dict[str, Any]] = []

    for ev in all_events:
        matched = _match_event_to_edge(ev, structural_edges, structural_nodes)
        if matched:
            edge_events[matched].append(ev)
        else:
            # Only count events that look like edge traversals
            if ev.get("source_node") and ev.get("target_node"):
                unmapped_edge_events.append(ev)

    # ---- Compute node behavioral overlays ----
    enriched_nodes: dict[str, dict[str, Any]] = {}
    for node_id, node_data in structural_nodes.items():
        behavioral = compute_node_behavioral_overlay(
            node_id, node_events.get(node_id, []), total_runs
        )
        # Structural prediction match: compare structural error handling
        # inference vs runtime observation
        structural_error_handling = node_data.get("error_handling", {})
        if structural_error_handling and behavioral["error_behavior"]["errors_occurred"] > 0:
            predicted = structural_error_handling.get("strategy")
            observed = behavioral["error_behavior"]["observed_error_handling"]
            behavioral["error_behavior"]["structural_prediction_match"] = (
                predicted in observed if predicted and observed else None
            )

        enriched_nodes[node_id] = {
            "structural": node_data,
            "behavioral": behavioral,
        }

    # ---- Compute edge behavioral overlays ----
    enriched_edges: dict[str, dict[str, Any]] = {}
    for edge_id, edge_data in structural_edges.items():
        behavioral = compute_edge_behavioral_overlay(
            edge_id, edge_events.get(edge_id, []), total_runs
        )
        enriched_edges[edge_id] = {
            "structural": edge_data,
            "behavioral": behavioral,
        }

    return {
        "repo_id": repo_id,
        "framework": framework,
        "total_runs": total_runs,
        "nodes": enriched_nodes,
        "edges": enriched_edges,
        "emergent_edges": [],  # Populated by edges.py
        "dead_edges": [],      # Populated by edges.py
        "unmapped_events": {
            "node_events": len(unmapped_node_events),
            "edge_events": len(unmapped_edge_events),
        },
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_all_events(run_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect all raw events from run records.

    Run records may carry their full event lists under an ``events`` key,
    or we reconstruct from the summary data.  The harness stores the raw
    events alongside metadata.
    """
    all_events: list[dict[str, Any]] = []
    for record in run_records:
        events = record.get("events", [])
        if events:
            all_events.extend(events)
    return all_events


def _match_event_to_node(
    event: dict[str, Any],
    structural_nodes: dict[str, dict[str, Any]],
) -> str | None:
    """Try to match an event to a structural node.

    Uses the source_node field from the event and the node_ids matching
    module.
    """
    source = event.get("source_node")
    if not source:
        return None

    # The patcher emits nodes as: {"node_type": ..., "node_id": ..., "node_name": ...}
    # Direct ID match against structural graph
    source_id = source.get("node_id", "") or source.get("id", "")
    if source_id in structural_nodes:
        return source_id

    # Try runtime ID format matching (framework:ClassName:file:line)
    if source_id and ":" in source_id:
        matched = match_runtime_to_structural(source_id, structural_nodes)
        if matched:
            return matched

    # Try name-based matching
    name = source.get("node_name", "") or source.get("name", "")
    if name:
        from stratum_lab.node_ids import structural_agent_id
        agent_id = structural_agent_id(name)
        if agent_id in structural_nodes:
            return agent_id

    # Fuzzy: normalized name contained in a structural node ID
    if name:
        from stratum_lab.node_ids import normalize_name
        normalized = normalize_name(name)
        for node_id in structural_nodes:
            if normalized and normalized in node_id:
                return node_id

    return None


def _match_event_to_edge(
    event: dict[str, Any],
    structural_edges: dict[str, dict[str, Any]],
    structural_nodes: dict[str, dict[str, Any]],
) -> str | None:
    """Try to match an event to a structural edge.

    Matches by comparing the event's source/target pair against structural
    edge source/target pairs.
    """
    source = event.get("source_node")
    target = event.get("target_node")
    if not source or not target:
        return None

    source_matched = _match_event_to_node(
        {"source_node": source}, structural_nodes
    )
    # Reuse _match_event_to_node for target by passing it as source_node
    target_matched = _match_event_to_node(
        {"source_node": target}, structural_nodes
    )

    if not source_matched or not target_matched:
        return None

    # Find structural edge matching the source->target pair
    for edge_id, edge_data in structural_edges.items():
        edge_source = edge_data.get("source", "")
        edge_target = edge_data.get("target", "")
        if edge_source == source_matched and edge_target == target_matched:
            return edge_id

    return None
