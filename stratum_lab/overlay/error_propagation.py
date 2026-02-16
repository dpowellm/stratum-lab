"""Error propagation path reconstruction.

For each error observed at runtime, trace:
1. Where did the error originate?
2. What was the structural prediction for propagation?
3. What actually happened? (observed path)
4. Was the error swallowed, propagated, or rerouted?
5. What was the downstream impact?
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional


def trace_error_propagation(
    structural_graph: Dict,
    events: List[Dict],
) -> List[Dict]:
    """Trace error propagation paths from runtime events."""
    traces = []

    error_events = [
        e for e in events
        if e.get("event_type", "").startswith("error.")
        or (e.get("event_type") == "agent.task_end"
            and e.get("payload", {}).get("status") == "error")
    ]

    by_run: Dict[str, List[Dict]] = defaultdict(list)
    for e in events:
        rid = e.get("run_id", "")
        by_run.setdefault(rid, []).append(e)

    for error_event in error_events:
        run_id = error_event.get("run_id", "")
        run_events = sorted(by_run.get(run_id, []),
                           key=lambda e: e.get("timestamp_ns", 0))

        error_data = error_event.get("payload", {})
        source_node = (error_data.get("error_node_id")
                      or error_data.get("node_id", ""))
        if not source_node:
            src = error_event.get("source_node", {})
            source_node = src.get("node_id", "") or src.get("node_name", "")
        error_type = error_data.get("error_type", "unknown")

        if not source_node:
            continue

        name_to_id = _build_name_to_structural_id(structural_graph)
        structural_source = _resolve_to_structural_id(source_node, name_to_id)
        predicted_path = _get_structural_downstream(
            structural_graph, structural_source or source_node
        )
        observed_path = _observe_propagation(
            run_events, error_event, source_node, structural_graph
        )
        stopped_by, stop_mechanism = _find_propagation_stop(
            run_events, error_event, observed_path
        )
        impact = _assess_downstream_impact(run_events, error_event, observed_path)

        traces.append({
            "error_source_node": source_node,
            "error_type": error_type,
            "error_message": error_data.get("error_message", str(error_data.get("error", "")))[:200],
            "structural_predicted_path": predicted_path,
            "actual_observed_path": observed_path,
            "propagation_stopped_by": stopped_by,
            "stop_mechanism": stop_mechanism,
            "downstream_impact": impact,
            "structural_prediction_match": predicted_path == observed_path,
            "swallowed": error_data.get("swallowed", len(observed_path) <= 1),
        })

    return traces


def _build_name_to_structural_id(graph: Dict) -> Dict[str, str]:
    """Map node names and runtime IDs to structural graph node IDs.

    Structural graphs use IDs like ``agent_researcher`` in edges, while
    runtime events use names like ``Researcher`` or patcher IDs like
    ``crewai:Researcher:agents.py:10``.  This mapping bridges the gap.
    """
    mapping: Dict[str, str] = {}
    for node_id, node_data in graph.get("nodes", {}).items():
        mapping[node_id] = node_id
        name = node_data.get("node_name") or node_data.get("name", "")
        if name:
            mapping[name] = node_id
    return mapping


def _resolve_to_structural_id(node_ref: str, name_to_id: Dict[str, str]) -> str:
    """Resolve a runtime node reference to its structural ID."""
    if node_ref in name_to_id:
        return name_to_id[node_ref]
    # Try runtime ID format: "framework:ClassName:file:line"
    parts = node_ref.split(":")
    if len(parts) >= 2 and parts[1] in name_to_id:
        return name_to_id[parts[1]]
    return ""


def _get_structural_downstream(graph: Dict, source_node: str) -> List[str]:
    """BFS from source_node following structural edges."""
    edges = graph.get("edges", {})
    adjacency: Dict[str, List[str]] = defaultdict(list)
    for edge_data in edges.values():
        s = edge_data.get("source", "")
        t = edge_data.get("target", "")
        if s and t:
            adjacency[s].append(t)

    path = [source_node]
    visited = {source_node}
    queue = [source_node]
    while queue:
        current = queue.pop(0)
        for neighbor in adjacency.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                queue.append(neighbor)

    return path


def _observe_propagation(
    run_events: List[Dict],
    error_event: Dict,
    source_node: str,
    structural_graph: Dict | None = None,
) -> List[str]:
    """Observe what actually happened after the error.

    Combines two signals:
    1. Explicit error events at downstream nodes (error.occurred, error.propagated,
       or agent.task_end with status=error).
    2. Error laundering: downstream nodes in the structural graph that executed
       AFTER the error — even if they didn't throw errors themselves — because
       they processed potentially degraded input from the errored upstream.
    """
    error_ts = error_event.get("timestamp_ns", 0)
    path = [source_node]
    seen = {source_node}

    subsequent = [
        e for e in run_events
        if e.get("timestamp_ns", 0) > error_ts
        and e.get("event_type") in ("agent.task_start", "agent.task_end",
                                     "error.occurred", "error.propagated")
    ]

    # 1. Explicit error propagation: nodes that had error events after the source
    for e in subsequent:
        payload = e.get("payload", {})
        node = payload.get("node_id", "")
        if not node:
            src = e.get("source_node", {})
            node = src.get("node_id", "") or src.get("node_name", "")
        if node and node not in seen:
            if e.get("event_type", "").startswith("error.") or \
               payload.get("status") == "error":
                path.append(node)
                seen.add(node)

    # 2. Error laundering: structurally downstream nodes that executed after
    #    the error without raising their own error. This is the "crown jewel"
    #    finding (STRAT-SI-001) — errors silently consumed by default values
    #    or swallowed exceptions, with downstream nodes continuing on degraded data.
    #
    #    Handles the runtime-vs-structural ID mismatch: structural edges use
    #    IDs like "agent_researcher" while events use "Researcher" or
    #    "crewai:Researcher:agents.py:10".
    if structural_graph:
        name_to_id = _build_name_to_structural_id(structural_graph)
        structural_source = _resolve_to_structural_id(source_node, name_to_id)

        if structural_source:
            downstream = _get_structural_downstream(
                structural_graph, structural_source
            )

            # Build set of structural IDs active after error
            active_structural: set[str] = set()
            for e in subsequent:
                payload = e.get("payload", {})
                node = payload.get("node_id", "")
                src = e.get("source_node", {})
                if not node:
                    node = src.get("node_id", "") or src.get("node_name", "")
                node_name = src.get("node_name", "")
                for ref in (node, node_name):
                    if ref:
                        resolved = _resolve_to_structural_id(ref, name_to_id)
                        if resolved:
                            active_structural.add(resolved)

            # Translate seen set to structural IDs to avoid duplicates
            seen_structural = set()
            for s in seen:
                resolved = _resolve_to_structural_id(s, name_to_id)
                if resolved:
                    seen_structural.add(resolved)

            # Walk structural downstream; extend for nodes active after error
            for node in downstream:
                if node not in seen_structural and node in active_structural:
                    path.append(node)
                    seen_structural.add(node)

    return path


def _find_propagation_stop(
    run_events: List[Dict],
    error_event: Dict,
    observed_path: List[str],
) -> tuple[str, str]:
    """Determine where and how error propagation stopped."""
    if len(observed_path) <= 1:
        return observed_path[0] if observed_path else "", "swallowed_at_source"

    last_node = observed_path[-1]

    error_ts = error_event.get("timestamp_ns", 0)
    subsequent = [
        e for e in run_events
        if e.get("timestamp_ns", 0) > error_ts
    ]

    for e in subsequent:
        payload = e.get("payload", {})
        node = payload.get("node_id", "")
        if not node:
            src = e.get("source_node", {})
            node = src.get("node_id", "") or src.get("node_name", "")

        if node == last_node:
            handling = payload.get("error_handling", "")
            if handling in ("retry", "caught_retry"):
                return last_node, "retry"
            if handling in ("caught_silent", "caught_default"):
                return last_node, "error_handler"
            if e.get("event_type") == "guardrail.triggered":
                return last_node, "guardrail"

    return last_node, "end_of_chain"


def _assess_downstream_impact(
    run_events: List[Dict],
    error_event: Dict,
    observed_path: List[str],
) -> Dict:
    """Assess the downstream impact of an error."""
    error_ts = error_event.get("timestamp_ns", 0)

    subsequent_errors = [
        e for e in run_events
        if e.get("timestamp_ns", 0) > error_ts
        and e.get("event_type", "").startswith("error.")
    ]

    subsequent_tasks = [
        e for e in run_events
        if e.get("timestamp_ns", 0) > error_ts
        and e.get("event_type") == "agent.task_end"
    ]

    failed_tasks = [
        e for e in subsequent_tasks
        if e.get("payload", {}).get("status") == "error"
    ]

    return {
        "nodes_affected": len(observed_path) - 1,
        "downstream_errors": len(subsequent_errors),
        "downstream_tasks_failed": len(failed_tasks),
        "cascade_depth": len(observed_path) - 1,
    }
