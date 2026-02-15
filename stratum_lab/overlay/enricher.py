"""Graph enrichment engine.

Takes a structural graph and runtime event data, computes behavioral
overlays for every node and edge, and returns an EnrichedGraph dict.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from stratum_lab.overlay.edges import detect_dead_edges, detect_emergent_edges
from stratum_lab.node_ids import (
    match_runtime_to_structural,
    normalize_name,
    structural_agent_id,
    structural_capability_id,
    structural_data_store_id,
    structural_external_id,
)


# ---------------------------------------------------------------------------
# Module-level event type constants
# ---------------------------------------------------------------------------

_ACTIVATION_START_TYPES = frozenset({
    "agent.task_start", "tool.invoked", "llm.call_start",
    "execution.start", "message.received", "reply.generation_start",
    "data.read", "data.write", "external.call", "guardrail.triggered",
})

_ACTIVATION_END_TYPES = frozenset({
    "agent.task_end", "tool.completed", "llm.call_end",
    "execution.end", "reply.generated",
})

_ERROR_TYPES = frozenset({
    "error.occurred", "error.propagated", "error.cascade",
    "tool.call_failure",
})


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

    START_TYPES = _ACTIVATION_START_TYPES
    END_TYPES = _ACTIVATION_END_TYPES
    ERROR_TYPES = _ERROR_TYPES

    # ---- Activation ----
    # activation_count: total start-like events at this node
    activation_events = [
        e for e in events if e.get("event_type") in START_TYPES
    ]
    activation_count = len(activation_events)

    # activation_rate: fraction of runs where this node was activated (0-1)
    # Bug fix 1C: use distinct run_ids, not raw count / total_runs
    runs_with_activation: set[str] = set()
    for e in events:
        if e.get("event_type") in START_TYPES:
            rid = e.get("run_id", "")
            if rid:
                runs_with_activation.add(rid)
    activation_rate = len(runs_with_activation) / total_runs

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

    default_values_used = sum(
        1 for e in error_events
        if e.get("payload", {}).get("default_value_used", False)
    )

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
        "structural_prediction_match": None,
    }

    # ---- Decision behavior (Bug fix 1D: match "decision.made") ----
    decision_events = [
        e for e in events
        if e.get("event_type") in ("decision.made", "decision")
    ]
    decisions_made = len(decision_events)

    # Shannon entropy of decision outcomes using selected_option_hash or outcome
    outcome_counts: Counter[str] = Counter()
    confidence_values: list[float] = []
    for e in decision_events:
        payload = e.get("payload", {})
        outcome = (
            payload.get("selected_option_hash")
            or payload.get("outcome", "unknown")
        )
        outcome_counts[outcome] += 1
        conf = payload.get("confidence")
        if conf is not None:
            try:
                confidence_values.append(float(conf))
            except (ValueError, TypeError):
                pass

    decision_entropy = _shannon_entropy(outcome_counts)

    # Consistency: group decisions by input hash
    input_groups: dict[str, list[str]] = defaultdict(list)
    for e in decision_events:
        payload = e.get("payload", {})
        input_hash = payload.get("input_hash", "")
        outcome = (
            payload.get("selected_option_hash")
            or payload.get("outcome", "unknown")
        )
        if input_hash:
            input_groups[input_hash].append(outcome)

    # Consistency across same input: fraction of same-input run pairs with matching decisions
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

    # Consistency across different inputs
    all_outcomes = [
        (e.get("payload", {}).get("selected_option_hash")
         or e.get("payload", {}).get("outcome", "unknown"))
        for e in decision_events
    ]
    unique_outcomes = len(set(all_outcomes)) if all_outcomes else 0
    consistency_diff_input = unique_outcomes / max(len(all_outcomes), 1) if all_outcomes else None

    # Confidence stats
    confidence_mean = float(np.mean(confidence_values)) if confidence_values else None
    confidence_variance = float(np.var(confidence_values)) if confidence_values else None

    decision_behavior = {
        "decisions_made": decisions_made,
        "decision_entropy": round(decision_entropy, 4),
        "consistency_across_same_input": consistency_same_input,
        "consistency_across_diff_input": consistency_diff_input,
        "confidence_mean": round(confidence_mean, 4) if confidence_mean is not None else None,
        "confidence_variance": round(confidence_variance, 4) if confidence_variance is not None else None,
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

    iteration_events = [e for e in events if e.get("event_type") == "agent.task_start"]

    resource_usage = {
        "avg_tokens": round(avg_tokens, 2),
        "avg_tool_calls": round(len(tool_start_events) / total_runs, 2),
        "avg_llm_calls": round(len(llm_start_events) / total_runs, 2),
        "avg_iterations": round(len(iteration_events) / total_runs, 2),
    }

    # ---- Guardrail effectiveness (for guardrail nodes) ----
    guardrail_events = [e for e in events if e.get("event_type") == "guardrail.triggered"]
    guardrail_effectiveness = None
    if guardrail_events:
        trigger_count = len(guardrail_events)
        prevented = sum(1 for e in guardrail_events
                        if e.get("payload", {}).get("action_prevented", False))
        bypass = sum(1 for e in guardrail_events
                     if e.get("payload", {}).get("bypassed", False))
        retry = sum(1 for e in guardrail_events
                    if e.get("payload", {}).get("retry_triggered", False))
        latencies_gr = [e.get("payload", {}).get("latency_ms", 0.0)
                        for e in guardrail_events
                        if "latency_ms" in e.get("payload", {})]
        avg_lat = float(np.mean(latencies_gr)) if latencies_gr else 0.0
        false_pos = sum(1 for e in guardrail_events
                        if e.get("payload", {}).get("false_positive", False))
        guardrail_effectiveness = {
            "trigger_count": trigger_count,
            "prevented_action_count": prevented,
            "bypass_count": bypass,
            "retry_count": retry,
            "avg_latency_impact_ms": round(avg_lat, 2),
            "false_positive_indicators": false_pos,
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
        "guardrail_effectiveness": guardrail_effectiveness,
    }


def _compute_latencies(events: list[dict[str, Any]]) -> list[int]:
    """Compute start-to-end latencies in nanoseconds from paired events."""
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
            parent = ev.get("parent_event_id")
            if parent and parent in starts:
                latency = ts - starts[parent]
                if latency >= 0:
                    latencies.append(latency)
            else:
                lat_ms = ev.get("payload", {}).get("latency_ms")
                if lat_ms is not None and lat_ms > 0:
                    latencies.append(int(lat_ms * 1_000_000))

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
    """Compute behavioral statistics for one structural edge."""
    total_runs = max(total_runs, 1)

    traversal_count = len(events)

    # activation_rate for edges: fraction of runs where this edge was traversed
    runs_traversed: set[str] = set()
    for e in events:
        rid = e.get("run_id", "")
        if rid:
            runs_traversed.add(rid)
    activation_rate = len(runs_traversed) / total_runs
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

    # ---- Error crossings (Bug fix 1F) ----
    error_traversals = [
        e for e in events
        if e.get("payload", {}).get("carries_error", False)
        or e.get("event_type") in ("error.propagated", "error.cascade")
        or e.get("payload", {}).get("error_type")
    ]
    error_types_seen: set[str] = set()
    for e in error_traversals:
        etype = e.get("payload", {}).get("error_type")
        if etype:
            error_types_seen.add(etype)

    downstream_impact = sum(
        1 for e in error_traversals
        if e.get("payload", {}).get("downstream_impact", False)
        or e.get("payload", {}).get("reached_irreversible", False)
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
    """Enrich a structural graph with behavioral data from runtime events."""
    structural_nodes = structural_graph.get("nodes", {})
    structural_edges = structural_graph.get("edges", {})
    repo_id = structural_graph.get("repo_id", "unknown")
    framework = structural_graph.get("framework", "unknown")
    total_runs = len(run_records) if run_records else 1

    all_events = _collect_all_events(run_records)

    # ---- Map runtime events to structural nodes ----
    # Bug fix 1A: map EVERY event to at least one node, handling all node types
    node_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unmapped_node_events: list[dict[str, Any]] = []

    for ev in all_events:
        matched_nodes = _match_event_to_nodes(ev, structural_nodes)
        if matched_nodes:
            for nid in matched_nodes:
                node_events[nid].append(ev)
        else:
            unmapped_node_events.append(ev)

    # ---- Map runtime events to structural edges ----
    # Bug fix 1B: infer edges from event type semantics, not just source/target
    edge_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unmapped_edge_events: list[dict[str, Any]] = []

    for ev in all_events:
        matched_edges = _match_event_to_edges(ev, structural_edges, structural_nodes)
        if matched_edges:
            for eid in matched_edges:
                edge_events[eid].append(ev)
        elif _is_edge_event(ev):
            unmapped_edge_events.append(ev)

    # ---- Bug fix 1F: Process error propagation paths for edges ----
    _apply_error_propagation(all_events, edge_events, structural_edges, structural_nodes)

    # ---- Compute node behavioral overlays ----
    enriched_nodes: dict[str, dict[str, Any]] = {}
    for node_id, node_data in structural_nodes.items():
        behavioral = compute_node_behavioral_overlay(
            node_id, node_events.get(node_id, []), total_runs
        )

        # Bug fix 1E: structural_prediction_match
        _compute_structural_prediction_match(node_data, behavioral)

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

    # ---- Part 2A: Cross-run determinism metrics ----
    input_groups = _group_runs_by_input(run_records)
    for node_id in enriched_nodes:
        determinism = _compute_node_determinism(
            node_id,
            node_events.get(node_id, []),
            input_groups,
            total_runs,
        )
        enriched_nodes[node_id]["behavioral"]["determinism"] = determinism

    # ---- Part 2B: Execution path signatures ----
    execution_paths = _build_execution_paths(
        run_records, structural_edges, structural_nodes,
    )
    path_analysis = _compute_path_analysis(execution_paths, total_runs)

    # ---- Part 2C: Tool failure impact chains ----
    _compute_failure_impacts(enriched_nodes, run_records, structural_nodes)

    # ---- Emergent & dead edge detection ----
    # Use all events (including single-endpoint) for dead edge detection.
    emergent_edges = detect_emergent_edges(
        structural_edges,
        [e for e in all_events if e.get("source_node") and e.get("target_node")],
        structural_nodes,
        total_runs=total_runs,
    )
    dead_edges = detect_dead_edges(
        structural_edges,
        all_events,
        total_runs=total_runs,
        structural_nodes=structural_nodes,
    )

    # Subtract unmapped edge events that correspond to emergent edges —
    # those events are accounted for by the emergent edge detection.
    emergent_pairs: set[tuple[str, str]] = set()
    for em in emergent_edges:
        emergent_pairs.add((em.get("source_node_id", ""), em.get("target_node_id", "")))

    truly_unmapped_edge_events = []
    for ev in unmapped_edge_events:
        src = ev.get("source_node")
        tgt = ev.get("target_node")
        if src and tgt:
            src_id = (
                _match_node_ref(src, structural_nodes)
                if isinstance(src, dict)
                else _resolve_node_ref(src, structural_nodes)
            )
            tgt_id = (
                _match_node_ref(tgt, structural_nodes)
                if isinstance(tgt, dict)
                else _resolve_node_ref(tgt, structural_nodes)
            )
            if src_id and tgt_id and (src_id, tgt_id) in emergent_pairs:
                continue  # accounted for as emergent
        truly_unmapped_edge_events.append(ev)

    return {
        "repo_id": repo_id,
        "framework": framework,
        "total_runs": total_runs,
        "nodes": enriched_nodes,
        "edges": enriched_edges,
        "execution_paths": execution_paths,
        "path_analysis": path_analysis,
        "emergent_edges": emergent_edges,
        "dead_edges": dead_edges,
        "unmapped_events": {
            "node_events": len(unmapped_node_events),
            "edge_events": len(truly_unmapped_edge_events),
        },
    }


# ---------------------------------------------------------------------------
# Bug fix 1E: Structural prediction match
# ---------------------------------------------------------------------------

def _compute_structural_prediction_match(
    node_data: dict[str, Any],
    behavioral: dict[str, Any],
) -> None:
    """Compare structural inferences against observed runtime behavior."""
    error_beh = behavioral["error_behavior"]

    # Check error handling strategy prediction
    structural_error_handling = node_data.get("error_handling", {})
    predicted_strategy = None
    if isinstance(structural_error_handling, dict):
        predicted_strategy = structural_error_handling.get("strategy")
    elif isinstance(structural_error_handling, str):
        predicted_strategy = structural_error_handling

    if predicted_strategy and error_beh["errors_occurred"] > 0:
        observed = error_beh["observed_error_handling"]
        if observed:
            error_beh["structural_prediction_match"] = predicted_strategy in observed
        else:
            # Errors occurred but no handling pattern observed — prediction can't be verified
            error_beh["structural_prediction_match"] = None
    elif predicted_strategy and error_beh["errors_occurred"] == 0:
        # Structural scan predicted an error handling pattern, but no errors occurred.
        # This is neither confirmed nor denied — insufficient data.
        error_beh["structural_prediction_match"] = None

    # Check timeout prediction
    timeout_config = node_data.get("timeout_config")
    if timeout_config == "none":
        # Predicted no timeout — check if timeouts occurred at this node
        timeout_events = [
            e for e in behavioral.get("_raw_events", [])
            if e.get("event_type") == "error.occurred"
            and "timeout" in str(e.get("payload", {}).get("error_type", "")).lower()
        ]
        if timeout_events:
            error_beh["structural_prediction_match"] = False


# ---------------------------------------------------------------------------
# Bug fix 1F: Error propagation through edges
# ---------------------------------------------------------------------------

def _apply_error_propagation(
    all_events: list[dict[str, Any]],
    edge_events: dict[str, list[dict[str, Any]]],
    structural_edges: dict[str, dict[str, Any]],
    structural_nodes: dict[str, dict[str, Any]],
) -> None:
    """Process error.propagated events and update edge error_crossings."""
    for ev in all_events:
        etype = ev.get("event_type", "")
        payload = ev.get("payload", {})

        if etype == "error.propagated":
            prop_path = payload.get("propagation_path", [])
            error_type = payload.get("error_type", "unknown_error")
            # For each consecutive pair in the path, find and update the edge
            for i in range(len(prop_path) - 1):
                node_a = prop_path[i]
                node_b = prop_path[i + 1]
                # Resolve to structural IDs
                a_id = _resolve_node_ref(node_a, structural_nodes)
                b_id = _resolve_node_ref(node_b, structural_nodes)
                if a_id and b_id:
                    matched_edge = _find_edge(a_id, b_id, structural_edges)
                    if matched_edge:
                        # Create a synthetic error-propagation event for this edge
                        error_ev = {
                            "event_type": "error.propagated",
                            "run_id": ev.get("run_id", ""),
                            "payload": {
                                "carries_error": True,
                                "error_type": error_type,
                                "downstream_impact": payload.get("downstream_impact", False),
                                "reached_irreversible": payload.get("reached_irreversible", False),
                            },
                        }
                        edge_events[matched_edge].append(error_ev)

        elif etype == "error.cascade":
            origin = payload.get("origin_node")
            terminal = payload.get("terminal_node")
            error_type = payload.get("error_type", "cascade")
            if origin and terminal:
                o_id = _resolve_node_ref(origin, structural_nodes)
                t_id = _resolve_node_ref(terminal, structural_nodes)
                if o_id and t_id:
                    matched_edge = _find_edge(o_id, t_id, structural_edges)
                    if matched_edge:
                        error_ev = {
                            "event_type": "error.cascade",
                            "run_id": ev.get("run_id", ""),
                            "payload": {
                                "carries_error": True,
                                "error_type": error_type,
                                "downstream_impact": True,
                            },
                        }
                        edge_events[matched_edge].append(error_ev)


def _resolve_node_ref(
    ref: Any,
    structural_nodes: dict[str, dict[str, Any]],
) -> str | None:
    """Resolve a node reference (string or dict) to a structural node ID."""
    if isinstance(ref, str):
        if ref in structural_nodes:
            return ref
        # Try as a name
        agent_id = structural_agent_id(ref)
        if agent_id in structural_nodes:
            return agent_id
        normalized = normalize_name(ref)
        for nid in structural_nodes:
            if normalized and normalized in nid:
                return nid
        return None
    elif isinstance(ref, dict):
        return _match_node_ref(ref, structural_nodes)
    return None


def _find_edge(
    source_id: str,
    target_id: str,
    structural_edges: dict[str, dict[str, Any]],
) -> str | None:
    """Find structural edge connecting source to target."""
    for edge_id, edge_data in structural_edges.items():
        s = edge_data.get("source", "")
        t = edge_data.get("target", "")
        if s == source_id and t == target_id:
            return edge_id
    return None


# ---------------------------------------------------------------------------
# Bug fix 1A: Multi-node event mapping
# ---------------------------------------------------------------------------

# Maps event_type -> which node roles the event touches
# Maps event_type -> which node roles the event touches.
# "source" and "target" refer to source_node / target_node on the event.
# "payload_agent" extracts agent info from payload.agent_id / agent_name.
# For tool/llm events, the patcher may emit source=agent,target=capability
# OR source=capability with payload.agent_id. We list all possible roles
# and _match_event_to_nodes deduplicates.
_EVENT_NODE_ROLES: dict[str, list[str]] = {
    "agent.task_start": ["source"],
    "agent.task_end": ["source"],
    "tool.invoked": ["source", "target", "payload_agent"],
    "tool.completed": ["source", "target", "payload_agent"],
    "tool.call_failure": ["source", "target", "payload_agent"],
    "data.read": ["source", "target"],
    "data.write": ["source", "target"],
    "llm.call_start": ["source", "target", "payload_agent"],
    "llm.call_end": ["source", "target", "payload_agent"],
    "external.call": ["source", "target"],
    "guardrail.triggered": ["source"],
    "decision.made": ["source"],
    "error.occurred": ["source"],
    "error.propagated": ["source"],
    "error.cascade": ["source"],
    "delegation.initiated": ["source", "target"],
    "delegation.completed": ["source", "target"],
    "execution.start": ["source"],
    "execution.end": ["source"],
    "message.received": ["source"],
    "file.read": ["source"],
    "file.write": ["source"],
}


def _match_event_to_nodes(
    event: dict[str, Any],
    structural_nodes: dict[str, dict[str, Any]],
) -> list[str]:
    """Match an event to one or more structural nodes.

    Bug fix 1A: Events can map to multiple nodes. For example, tool.invoked
    maps to both the tool (capability) and the invoking agent.
    """
    etype = event.get("event_type", "")
    roles = _EVENT_NODE_ROLES.get(etype, ["source"])
    matched: list[str] = []
    seen: set[str] = set()

    for role in roles:
        node_ref = None
        if role == "source":
            node_ref = event.get("source_node")
        elif role == "target":
            node_ref = event.get("target_node")
        elif role == "payload_agent":
            # Agent ID embedded in payload
            payload = event.get("payload", {})
            agent_id = payload.get("agent_id") or payload.get("agent_name")
            if agent_id:
                node_ref = {"node_name": agent_id, "node_type": "agent"}

        if node_ref:
            nid = _match_node_ref(node_ref, structural_nodes)
            if nid and nid not in seen:
                matched.append(nid)
                seen.add(nid)

    # Fallback: if nothing matched but we have a source_node, try harder
    if not matched:
        source = event.get("source_node")
        if source:
            nid = _match_node_ref(source, structural_nodes)
            if nid:
                matched.append(nid)

    return matched


def _match_node_ref(
    node_ref: dict[str, Any],
    structural_nodes: dict[str, dict[str, Any]],
) -> str | None:
    """Match a node reference dict to a structural node ID.

    Tries: direct ID, runtime ID, agent name, capability name, data store
    name, external service name, guardrail name, then fuzzy match.
    """
    node_id = node_ref.get("node_id", "") or node_ref.get("id", "")
    node_name = node_ref.get("node_name", "") or node_ref.get("name", "")
    node_type = node_ref.get("node_type", "")

    # Direct ID match
    if node_id and node_id in structural_nodes:
        return node_id

    # Runtime ID format (framework:ClassName:file:line)
    if node_id and ":" in node_id:
        matched = match_runtime_to_structural(node_id, structural_nodes)
        if matched:
            return matched

    # Type-aware name matching
    if node_name:
        # Try agent
        aid = structural_agent_id(node_name)
        if aid in structural_nodes:
            return aid

        # Try capability patterns
        normalized = normalize_name(node_name)
        for nid, ndata in structural_nodes.items():
            structural = ndata.get("structural", ndata)
            stype = structural.get("node_type", "")

            # Match by name field on structural node
            struct_name = structural.get("name", "")
            if struct_name and normalize_name(struct_name) == normalized:
                return nid

            # Match by class_name for capabilities
            struct_class = structural.get("class_name", "")
            if struct_class and normalize_name(struct_class) == normalized:
                return nid

        # Try data store ID
        ds_id = structural_data_store_id(node_name)
        if ds_id in structural_nodes:
            return ds_id

        # Try external service ID
        ext_id = structural_external_id(node_name)
        if ext_id in structural_nodes:
            return ext_id

        # Fuzzy: normalized name contained in structural node ID
        for nid in structural_nodes:
            if normalized and normalized in nid:
                return nid

    return None


# ---------------------------------------------------------------------------
# Bug fix 1B: Edge event mapping
# ---------------------------------------------------------------------------

# Maps event_type -> list of (source_role, target_role, expected_edge_types)
# Multiple entries allow matching different event formats (e.g., source=agent,target=tool
# or source=tool with payload_agent).
_EVENT_EDGE_MAP: dict[str, list[tuple[str, str, list[str]]]] = {
    "tool.invoked": [
        ("source", "target", ["uses", "tool_of", "calls"]),
        ("payload_agent", "source", ["uses", "tool_of", "calls"]),
    ],
    "tool.completed": [
        ("source", "target", ["uses", "tool_of", "calls"]),
        ("payload_agent", "source", ["uses", "tool_of", "calls"]),
    ],
    "data.read": [("source", "target", ["reads_from"])],
    "data.write": [("source", "target", ["writes_to"])],
    "delegation.initiated": [("source", "target", ["delegates_to"])],
    "delegation.completed": [("source", "target", ["delegates_to"])],
    "external.call": [("source", "target", ["calls"])],
    "guardrail.triggered": [("source", "target", ["guarded_by", "filtered_by"])],
}


def _match_event_to_edges(
    event: dict[str, Any],
    structural_edges: dict[str, dict[str, Any]],
    structural_nodes: dict[str, dict[str, Any]],
) -> list[str]:
    """Match an event to structural edges based on event type semantics."""
    etype = event.get("event_type", "")
    mappings = _EVENT_EDGE_MAP.get(etype)

    matched: list[str] = []
    seen: set[str] = set()

    if mappings:
        for src_role, tgt_role, expected_types in mappings:
            src_ref = _get_event_role_ref(event, src_role)
            tgt_ref = _get_event_role_ref(event, tgt_role)

            if src_ref and tgt_ref:
                src_id = _match_node_ref(src_ref, structural_nodes) if isinstance(src_ref, dict) else _resolve_node_ref(src_ref, structural_nodes)
                tgt_id = _match_node_ref(tgt_ref, structural_nodes) if isinstance(tgt_ref, dict) else _resolve_node_ref(tgt_ref, structural_nodes)

                if src_id and tgt_id:
                    for edge_id, edge_data in structural_edges.items():
                        if edge_id in seen:
                            continue
                        es = edge_data.get("source", "")
                        et = edge_data.get("target", "")
                        edge_type = edge_data.get("edge_type", "")
                        if es == src_id and et == tgt_id:
                            matched.append(edge_id)
                            seen.add(edge_id)
                        elif es == tgt_id and et == src_id:
                            if edge_type in expected_types:
                                matched.append(edge_id)
                                seen.add(edge_id)

            # Partial match: only one endpoint available (e.g. guardrail events
            # with source_node but no target_node).  Find structural edges where
            # the known node participates with an expected edge type.
            elif src_ref or tgt_ref:
                ref = src_ref or tgt_ref
                node_id = _match_node_ref(ref, structural_nodes) if isinstance(ref, dict) else _resolve_node_ref(ref, structural_nodes)
                if node_id:
                    for edge_id, edge_data in structural_edges.items():
                        if edge_id in seen:
                            continue
                        edge_type = edge_data.get("edge_type", "")
                        if edge_type not in expected_types:
                            continue
                        es = edge_data.get("source", "")
                        et = edge_data.get("target", "")
                        if es == node_id or et == node_id:
                            matched.append(edge_id)
                            seen.add(edge_id)

    # Fallback: explicit source/target pair
    if not matched:
        source = event.get("source_node")
        target = event.get("target_node")
        if source and target:
            src_id = _match_node_ref(source, structural_nodes) if isinstance(source, dict) else None
            tgt_id = _match_node_ref(target, structural_nodes) if isinstance(target, dict) else None
            if src_id and tgt_id:
                for edge_id, edge_data in structural_edges.items():
                    if edge_id in seen:
                        continue
                    if edge_data.get("source") == src_id and edge_data.get("target") == tgt_id:
                        matched.append(edge_id)
                        seen.add(edge_id)

    return matched


def _get_event_role_ref(event: dict[str, Any], role: str) -> dict[str, Any] | None:
    """Get a node reference dict for a given role from an event."""
    if role == "source":
        return event.get("source_node")
    elif role == "target":
        return event.get("target_node")
    elif role == "payload_agent":
        payload = event.get("payload", {})
        agent_id = payload.get("agent_id") or payload.get("agent_name")
        if agent_id:
            return {"node_name": agent_id, "node_type": "agent"}
        # Fall back to source_node if it's an agent
        source = event.get("source_node")
        if source and source.get("node_type") == "agent":
            return source
        return None
    return None


def _is_edge_event(event: dict[str, Any]) -> bool:
    """Check if an event implies an edge traversal."""
    etype = event.get("event_type", "")
    if etype in _EVENT_EDGE_MAP:
        return True
    if event.get("source_node") and event.get("target_node"):
        return True
    return False


# ---------------------------------------------------------------------------
# Part 2A: Cross-run determinism metrics
# ---------------------------------------------------------------------------

def _group_runs_by_input(
    run_records: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Group run IDs by their input_hash from metadata."""
    groups: dict[str, list[str]] = defaultdict(list)
    for rec in run_records:
        input_hash = (
            rec.get("metadata", {}).get("input_hash")
            or rec.get("run_id", "unknown")
        )
        groups[input_hash].append(rec.get("run_id", ""))
    return dict(groups)


def _compute_node_determinism(
    node_id: str,
    node_events_list: list[dict[str, Any]],
    input_groups: dict[str, list[str]],
    total_runs: int,
) -> dict[str, Any] | None:
    """Compute cross-run determinism metrics for a single node.

    Returns None if fewer than 2 runs.
    """
    if total_runs < 2:
        return None

    # Build per-run activation counts
    run_activation_counts: dict[str, int] = defaultdict(int)
    for ev in node_events_list:
        if ev.get("event_type") in _ACTIVATION_START_TYPES:
            rid = ev.get("run_id", "")
            if rid:
                run_activation_counts[rid] += 1

    # Same-input activation consistency: across same-input run pairs,
    # what fraction have matching activation state (both active or both inactive)?
    same_pairs = 0
    same_matches = 0
    for _ih, run_ids in input_groups.items():
        if len(run_ids) < 2:
            continue
        for i in range(len(run_ids)):
            for j in range(i + 1, len(run_ids)):
                same_pairs += 1
                a_active = run_activation_counts.get(run_ids[i], 0) > 0
                b_active = run_activation_counts.get(run_ids[j], 0) > 0
                if a_active == b_active:
                    same_matches += 1

    same_input_activation_consistency = (
        round(same_matches / same_pairs, 4) if same_pairs > 0 else None
    )

    # Same-input path consistency: across same-input runs, do activation counts match?
    # (Stronger than just active/inactive — checks if the node activates the same number of times)
    path_pairs = 0
    path_matches = 0
    for _ih, run_ids in input_groups.items():
        if len(run_ids) < 2:
            continue
        for i in range(len(run_ids)):
            for j in range(i + 1, len(run_ids)):
                path_pairs += 1
                a_count = run_activation_counts.get(run_ids[i], 0)
                b_count = run_activation_counts.get(run_ids[j], 0)
                if a_count == b_count:
                    path_matches += 1

    same_input_path_consistency = (
        round(path_matches / path_pairs, 4) if path_pairs > 0 else None
    )

    # Cross-input variance: coefficient of variation of activation counts
    # across different input groups (using per-group mean activation count)
    group_means: list[float] = []
    for _ih, run_ids in input_groups.items():
        counts = [run_activation_counts.get(rid, 0) for rid in run_ids]
        if counts:
            group_means.append(sum(counts) / len(counts))

    if len(group_means) >= 2:
        arr = np.array(group_means, dtype=np.float64)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        cross_input_variance = round(std_val / max(mean_val, 1e-9), 4)
    else:
        cross_input_variance = 0.0

    return {
        "same_input_activation_consistency": same_input_activation_consistency,
        "same_input_path_consistency": same_input_path_consistency,
        "cross_input_variance": cross_input_variance,
    }


# ---------------------------------------------------------------------------
# Part 2B: Execution path signatures
# ---------------------------------------------------------------------------

def _build_execution_paths(
    run_records: list[dict[str, Any]],
    structural_edges: dict[str, dict[str, Any]],
    structural_nodes: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build per-run execution paths: ordered sequences of edge traversals."""
    paths: list[dict[str, Any]] = []
    for rec in run_records:
        run_id = rec.get("run_id", "")
        events = sorted(
            rec.get("events", []),
            key=lambda e: e.get("timestamp_ns", 0),
        )
        path: list[dict[str, Any]] = []
        seen_in_step: set[str] = set()
        for ev in events:
            matched_edges = _match_event_to_edges(ev, structural_edges, structural_nodes)
            for eid in matched_edges:
                ts = ev.get("timestamp_ns", 0)
                dedup_key = f"{eid}:{ts}"
                if dedup_key in seen_in_step:
                    continue
                seen_in_step.add(dedup_key)
                edge = structural_edges[eid]
                path.append({
                    "edge_id": eid,
                    "timestamp_ns": ts,
                    "source": edge.get("source", ""),
                    "target": edge.get("target", ""),
                    "edge_type": edge.get("edge_type", ""),
                })
        paths.append({"run_id": run_id, "execution_path": path})
    return paths


def _compute_path_analysis(
    execution_paths: list[dict[str, Any]],
    total_runs: int,
) -> dict[str, Any]:
    """Compute aggregate path analysis from per-run execution paths."""
    if not execution_paths:
        return {
            "distinct_paths": 0,
            "dominant_path_frequency": 0.0,
            "path_divergence_points": [],
            "conditional_edge_activation_rates": {},
        }

    # Convert each path to a hashable signature (tuple of edge_ids in order)
    path_sigs: list[tuple[str, ...]] = []
    for p in execution_paths:
        sig = tuple(step["edge_id"] for step in p["execution_path"])
        path_sigs.append(sig)

    distinct_paths = len(set(path_sigs))

    # Dominant path frequency
    sig_counts: Counter[tuple[str, ...]] = Counter(path_sigs)
    dominant_count = sig_counts.most_common(1)[0][1] if sig_counts else 0
    dominant_path_frequency = dominant_count / max(len(path_sigs), 1)

    # Path divergence points: edges appearing in some but not all runs
    all_edge_ids: set[str] = set()
    per_run_edges: list[set[str]] = []
    for p in execution_paths:
        run_edges = {step["edge_id"] for step in p["execution_path"]}
        per_run_edges.append(run_edges)
        all_edge_ids |= run_edges

    divergence_nodes: set[str] = set()
    edge_source_map: dict[str, str] = {}
    for p in execution_paths:
        for step in p["execution_path"]:
            if step["edge_id"] not in edge_source_map:
                edge_source_map[step["edge_id"]] = step["source"]

    for eid in all_edge_ids:
        runs_with = sum(1 for re in per_run_edges if eid in re)
        if 0 < runs_with < len(per_run_edges):
            if eid in edge_source_map:
                divergence_nodes.add(edge_source_map[eid])

    # Conditional edge activation rates (edges that don't fire in every run)
    conditional_rates: dict[str, dict[str, Any]] = {}
    for eid in all_edge_ids:
        runs_with = sum(1 for re in per_run_edges if eid in re)
        rate = runs_with / max(len(per_run_edges), 1)
        if rate < 1.0:
            conditional_rates[eid] = {"activation_rate": round(rate, 4)}

    return {
        "distinct_paths": distinct_paths,
        "dominant_path_frequency": round(dominant_path_frequency, 4),
        "path_divergence_points": sorted(divergence_nodes),
        "conditional_edge_activation_rates": conditional_rates,
    }


# ---------------------------------------------------------------------------
# Part 2C: Tool failure impact chains
# ---------------------------------------------------------------------------

def _compute_failure_impacts(
    enriched_nodes: dict[str, dict[str, Any]],
    run_records: list[dict[str, Any]],
    structural_nodes: dict[str, dict[str, Any]],
) -> None:
    """Compute failure_impact for capability nodes that experienced tool failures.

    Modifies enriched_nodes in place, adding failure_impact to model_sensitivity.
    """
    # Build per-run sorted event lists
    run_events: dict[str, list[dict[str, Any]]] = {}
    for rec in run_records:
        run_id = rec.get("run_id", "")
        events = sorted(
            rec.get("events", []),
            key=lambda e: e.get("timestamp_ns", 0),
        )
        run_events[run_id] = events

    for node_id, node_data in enriched_nodes.items():
        structural = node_data["structural"]
        node_type = structural.get("node_type", "")
        if node_type != "capability":
            continue

        behavioral = node_data["behavioral"]
        ms = behavioral.get("model_sensitivity", {})
        if ms.get("tool_call_failures", 0) == 0:
            continue

        node_name = structural.get("name", "") or structural.get("class_name", "")
        norm_name = normalize_name(node_name)

        total_downstream_degradation = 0
        cascade_to_others = False
        recovery_observed = False
        recovery_times: list[float] = []
        silent_degradation = 0

        for _run_id, events in run_events.items():
            for idx, ev in enumerate(events):
                # Find failure events for this capability node
                etype = ev.get("event_type", "")
                is_failure = False
                if etype == "tool.call_failure":
                    is_failure = True
                elif etype == "tool.completed":
                    if ev.get("payload", {}).get("status") in ("error", "failure"):
                        is_failure = True
                if not is_failure:
                    continue

                # Check if this failure is for our capability node
                source = ev.get("source_node", {})
                src_name = source.get("node_name", "") or source.get("node_id", "")
                payload_tool = ev.get("payload", {}).get("tool_name", "")
                if not (
                    normalize_name(src_name) == norm_name
                    or normalize_name(payload_tool) == norm_name
                    or source.get("node_id", "") == node_id
                ):
                    continue

                failure_ts = ev.get("timestamp_ns", 0)
                invoking_agent = ev.get("payload", {}).get("agent_id", "")
                norm_invoking = normalize_name(invoking_agent)
                subsequent = events[idx + 1:]

                # Downstream degradation: errors after this failure
                for sub_ev in subsequent:
                    sub_type = sub_ev.get("event_type", "")
                    if sub_type in ("error.occurred", "error.propagated", "error.cascade"):
                        total_downstream_degradation += 1
                        sub_source = sub_ev.get("source_node", {})
                        sub_agent = sub_source.get("node_name", "") or sub_source.get("node_id", "")
                        if sub_agent and invoking_agent and normalize_name(sub_agent) != norm_invoking:
                            cascade_to_others = True

                # Recovery: retry or successful completion after failure
                for sub_ev in subsequent:
                    sub_type = sub_ev.get("event_type", "")
                    sub_payload = sub_ev.get("payload", {})
                    sub_tool = sub_payload.get("tool_name", "")
                    if sub_payload.get("is_retry") and normalize_name(sub_tool) == norm_name:
                        recovery_observed = True
                        dt = (sub_ev.get("timestamp_ns", 0) - failure_ts) / 1_000_000
                        if dt > 0:
                            recovery_times.append(dt)
                        break
                    if (
                        sub_type == "tool.completed"
                        and sub_payload.get("status") == "success"
                        and normalize_name(sub_tool) == norm_name
                    ):
                        recovery_observed = True
                        dt = (sub_ev.get("timestamp_ns", 0) - failure_ts) / 1_000_000
                        if dt > 0:
                            recovery_times.append(dt)
                        break

                # Silent degradation: downstream agents proceeded without noticing
                downstream_agents = [
                    e for e in subsequent
                    if e.get("event_type") in ("agent.task_start", "agent.task_end")
                    and normalize_name(
                        e.get("source_node", {}).get("node_name", "")
                    ) != norm_invoking
                ]
                if downstream_agents:
                    downstream_errors = [
                        e for e in subsequent
                        if e.get("event_type") in ("error.occurred", "error.propagated")
                        and normalize_name(
                            e.get("source_node", {}).get("node_name", "")
                        ) != norm_invoking
                    ]
                    if not downstream_errors:
                        silent_degradation += 1

        avg_recovery_time = (
            round(sum(recovery_times) / len(recovery_times), 2)
            if recovery_times
            else 0.0
        )

        ms["failure_impact"] = {
            "downstream_degradation_count": total_downstream_degradation,
            "cascade_to_other_agents": cascade_to_others,
            "recovery_observed": recovery_observed,
            "avg_recovery_time_ms": avg_recovery_time,
            "silent_degradation_count": silent_degradation,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_all_events(run_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect all raw events from run records.

    Ensures every event has a ``run_id`` — if the event doesn't carry one,
    it is injected from the parent run record.
    """
    all_events: list[dict[str, Any]] = []
    for record in run_records:
        run_id = record.get("run_id", "")
        events = record.get("events", [])
        for ev in events:
            if not ev.get("run_id") and run_id:
                ev = {**ev, "run_id": run_id}
            all_events.append(ev)
    return all_events
