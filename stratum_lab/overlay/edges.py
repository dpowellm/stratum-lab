"""Emergent and dead edge detection.

Compares structural graph edges against runtime interactions to identify:
  - Emergent edges: runtime interactions with no structural edge
  - Dead edges: structural edges never traversed at runtime
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from stratum_lab.node_ids import match_runtime_to_structural


# ---------------------------------------------------------------------------
# Emergent edge detection
# ---------------------------------------------------------------------------

def detect_emergent_edges(
    structural_edges: dict[str, dict[str, Any]],
    runtime_interactions: list[dict[str, Any]],
    structural_nodes: dict[str, dict[str, Any]] | None = None,
    total_runs: int = 1,
) -> list[dict[str, Any]]:
    """Find runtime interactions with no corresponding structural edge.

    Parameters
    ----------
    structural_edges:
        Dict of structural edge_id -> edge data (with ``source``, ``target``).
    runtime_interactions:
        List of runtime event dicts that represent interactions (have both
        ``source_node`` and ``target_node``).
    structural_nodes:
        Optional dict of structural nodes for runtime ID matching.
    total_runs:
        Total runs observed, used for activation rate computation.

    Returns
    -------
    List of EmergentEdge dicts.
    """
    structural_nodes = structural_nodes or {}
    total_runs = max(total_runs, 1)

    # Build set of known structural source->target pairs
    structural_pairs: set[tuple[str, str]] = set()
    for edge_data in structural_edges.values():
        source = edge_data.get("source", "")
        target = edge_data.get("target", "")
        if source and target:
            structural_pairs.add((source, target))

    # Gather all runtime source->target pairs with event details
    runtime_pair_events: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for event in runtime_interactions:
        source_node = event.get("source_node")
        target_node = event.get("target_node")
        if not source_node or not target_node:
            continue

        source_id = _resolve_node_id(source_node, structural_nodes)
        target_id = _resolve_node_id(target_node, structural_nodes)

        if source_id and target_id:
            runtime_pair_events[(source_id, target_id)].append(event)

    # Find pairs in runtime but not in structural
    emergent_edges: list[dict[str, Any]] = []
    edge_counter = 0

    for (source_id, target_id), events in runtime_pair_events.items():
        if (source_id, target_id) not in structural_pairs:
            edge_counter += 1
            traversal_count = len(events)
            activation_rate = traversal_count / total_runs

            # Infer edge type from event types
            edge_type = _infer_edge_type(events)

            # Infer trigger condition from payloads
            trigger_condition = _infer_trigger_condition(events)

            emergent_edge = {
                "edge_id": f"emergent_{edge_counter}",
                "edge_type": edge_type,
                "source_node_id": source_id,
                "target_node_id": target_id,
                "runtime_only": True,
                "traversal_count": traversal_count,
                "activation_rate": round(activation_rate, 4),
                "trigger_condition": trigger_condition,
                "significance": classify_edge_significance(
                    source_id, target_id, structural_nodes, events
                ),
            }
            emergent_edges.append(emergent_edge)

    # Sort by traversal count descending (most significant first)
    emergent_edges.sort(key=lambda e: e["traversal_count"], reverse=True)
    return emergent_edges


def detect_emergent_edges_v2(
    structural_graph: dict[str, Any],
    events: list[dict[str, Any]],
    run_count: int,
) -> list[dict[str, Any]]:
    """Detect runtime interactions with no structural counterpart.

    Discovery types:
    - error_triggered_fallback: delegation that only occurs when upstream errors
    - dynamic_delegation: LLM-chosen routing not in static graph
    - implicit_data_sharing: agents access same state without structural shares_with
    - framework_internal_routing: framework orchestrator creates undeclared paths
    """
    structural_edges = structural_graph.get("edges", {})
    structural_edge_set = set()
    for edge_data in structural_edges.values():
        s = edge_data.get("source", "")
        t = edge_data.get("target", "")
        if s and t:
            structural_edge_set.add((s, t))

    run_count = max(run_count, 1)
    emergent = []

    # 1. Detect from delegation events
    delegation_events = [e for e in events if e.get("event_type") == "delegation.initiated"]
    seen_pairs: set[tuple[str, str]] = set()
    for event in delegation_events:
        data = event.get("payload", {})
        source = data.get("source_node_id", "")
        target = data.get("target_node_id", "")
        if not source:
            src_node = event.get("source_node", {})
            source = src_node.get("node_id", "") or src_node.get("node_name", "")
        if not target:
            tgt_node = event.get("target_node", {})
            target = tgt_node.get("node_id", "") or tgt_node.get("node_name", "")

        if not source or not target:
            continue
        if (source, target) in structural_edge_set or (source, target) in seen_pairs:
            continue

        # Check if it also doesn't match via fuzzy matching
        if _fuzzy_match_structural(source, target, structural_edge_set):
            continue

        seen_pairs.add((source, target))
        discovery_type = _classify_emergent_type(event, events)
        activation_count = sum(
            1 for e in delegation_events
            if (e.get("payload", {}).get("source_node_id", "") == source or
                (e.get("source_node", {}).get("node_name", "") == source))
            and (e.get("payload", {}).get("target_node_id", "") == target or
                 (e.get("target_node", {}).get("node_name", "") == target))
        )

        emergent.append({
            "edge_id": f"emergent_{len(emergent):03d}",
            "source_node": source,
            "target_node": target,
            "edge_type": "delegates_to",
            "activation_count": activation_count,
            "activation_rate": round(activation_count / run_count, 4),
            "trigger_condition": _describe_emergent_trigger(event, events),
            "discovery_type": discovery_type,
            "detection_heuristic": _generate_detection_heuristic(discovery_type, source, target),
        })

    # 2. Detect from shared state access patterns
    state_events = [e for e in events if e.get("event_type") == "state.access"]
    implicit_shares = _detect_implicit_sharing_v2(state_events, structural_edge_set)
    for share in implicit_shares:
        emergent.append({
            "edge_id": f"emergent_{len(emergent):03d}",
            "source_node": share["writer"],
            "target_node": share["reader"],
            "edge_type": "shares_with",
            "activation_count": share["interaction_count"],
            "activation_rate": round(share["interaction_count"] / run_count, 4),
            "trigger_condition": f"both agents access state key '{share['state_key']}'",
            "discovery_type": "implicit_data_sharing",
            "detection_heuristic": (
                f"Monitor for agents accessing the same state key "
                f"'{share['state_key']}' — structural graph should have "
                f"a shares_with edge between {share['writer']} and {share['reader']}"
            ),
            "shared_state_key": share["state_key"],
        })

    # 3. Detect from routing decisions
    routing_events = [e for e in events if e.get("event_type") == "routing.decision"]
    for event in routing_events:
        data = event.get("payload", {})
        source = data.get("source_node", "")
        target = data.get("target_node", "")

        if not source or not target:
            continue
        if (source, target) in structural_edge_set or (source, target) in seen_pairs:
            continue
        if _fuzzy_match_structural(source, target, structural_edge_set):
            continue

        seen_pairs.add((source, target))
        routing_type = data.get("routing_type", "unknown")
        basis = data.get("decision_basis", "unknown")

        emergent.append({
            "edge_id": f"emergent_{len(emergent):03d}",
            "source_node": source,
            "target_node": target,
            "edge_type": "delegates_to",
            "activation_count": 1,
            "activation_rate": round(1 / run_count, 4),
            "trigger_condition": f"routing decision ({routing_type}) based on {basis}",
            "discovery_type": ("dynamic_delegation" if basis == "llm_output"
                              else "framework_internal_routing"),
            "detection_heuristic": _generate_routing_heuristic(routing_type, basis, source, target),
        })

    return emergent


def _fuzzy_match_structural(source: str, target: str, structural_edge_set: set) -> bool:
    """Check if source->target fuzzy-matches any structural edge."""
    from stratum_lab.node_ids import normalize_name
    src_norm = normalize_name(source)
    tgt_norm = normalize_name(target)
    for (s, t) in structural_edge_set:
        s_norm = normalize_name(s.replace("agent_", "").replace("cap_", ""))
        t_norm = normalize_name(t.replace("agent_", "").replace("cap_", ""))
        if (src_norm == s_norm or src_norm in s_norm or s_norm in src_norm) and \
           (tgt_norm == t_norm or tgt_norm in t_norm or t_norm in tgt_norm):
            return True
    return False


def _classify_emergent_type(event: dict, all_events: list[dict]) -> str:
    """Classify an emergent delegation type."""
    data = event.get("payload", {})
    timestamp = event.get("timestamp_ns", event.get("timestamp", 0))
    source = data.get("source_node_id", "")
    if not source:
        source = event.get("source_node", {}).get("node_name", "")

    recent_errors = [
        e for e in all_events
        if e.get("event_type", "").startswith("error.")
        and abs(e.get("timestamp_ns", e.get("timestamp", 0)) - timestamp) < 5_000_000_000
    ]
    if recent_errors:
        return "error_triggered_fallback"

    delegation_type = data.get("delegation_type", "")
    if delegation_type == "framework_internal":
        return "framework_internal_routing"

    return "dynamic_delegation"


def _describe_emergent_trigger(event: dict, all_events: list[dict]) -> str:
    """Describe what triggers this emergent edge."""
    dtype = _classify_emergent_type(event, all_events)
    if dtype == "error_triggered_fallback":
        return "error in upstream node triggered fallback delegation"
    elif dtype == "framework_internal_routing":
        return "framework orchestrator created routing path"
    return "LLM-chosen or dynamic routing decision"


def _generate_detection_heuristic(discovery_type: str, source: str, target: str) -> str:
    """Generate a detection heuristic importable by the reliability scanner."""
    if discovery_type == "error_triggered_fallback":
        return (
            f"Monitor for delegation to '{target}' when '{source}' errors. "
            f"Structural graph should add a conditional delegates_to edge "
            f"with error_triggered=True."
        )
    elif discovery_type == "dynamic_delegation":
        return (
            f"Monitor for LLM-chosen routing from '{source}' to '{target}'. "
            f"Consider adding delegates_to edge with dynamic=True in structural graph."
        )
    elif discovery_type == "framework_internal_routing":
        return (
            f"Framework orchestrator routes from '{source}' to '{target}'. "
            f"This is a framework-internal path not visible in user code."
        )
    return f"Emergent interaction between '{source}' and '{target}'."


def _generate_routing_heuristic(routing_type: str, basis: str, source: str, target: str) -> str:
    """Generate heuristic for routing-discovered edges."""
    return (
        f"Routing decision ({routing_type}) based on {basis}: "
        f"'{source}' -> '{target}'. Consider adding structural edge."
    )


def _detect_implicit_sharing_v2(state_events: list[dict], structural_edge_set: set) -> list[dict]:
    """Detect implicit data sharing from state.access events."""
    from collections import defaultdict

    writes: dict[str, list[str]] = defaultdict(list)
    reads: dict[str, list[str]] = defaultdict(list)

    for event in state_events:
        data = event.get("payload", {})
        node_id = data.get("node_id", "")
        state_key = data.get("state_key", "")
        access_type = data.get("access_type", "")

        if not node_id or not state_key:
            continue

        if access_type in ("write", "read_write"):
            writes[state_key].append(node_id)
        if access_type in ("read", "read_write"):
            reads[state_key].append(node_id)

    shares = []
    for state_key in set(writes.keys()) & set(reads.keys()):
        writers = set(writes[state_key])
        readers = set(reads[state_key])
        for w in writers:
            for r in readers:
                if w != r and (w, r) not in structural_edge_set:
                    interaction_count = min(len(writes[state_key]), len(reads[state_key]))
                    shares.append({
                        "writer": w,
                        "reader": r,
                        "state_key": state_key,
                        "interaction_count": interaction_count,
                    })

    return shares


# ---------------------------------------------------------------------------
# Dead edge detection
# ---------------------------------------------------------------------------

def detect_dead_edges(
    structural_edges: dict[str, dict[str, Any]],
    runtime_interactions: list[dict[str, Any]],
    total_runs: int,
    structural_nodes: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Find structural edges that were never traversed at runtime.

    Parameters
    ----------
    structural_edges:
        Dict of structural edge_id -> edge data.
    runtime_interactions:
        List of runtime event dicts representing interactions.
    total_runs:
        Total number of runs observed.
    structural_nodes:
        Optional structural node dict for runtime ID matching.

    Returns
    -------
    List of DeadEdge dicts.
    """
    structural_nodes = structural_nodes or {}

    # Build set of runtime source->target pairs
    runtime_pairs: set[tuple[str, str]] = set()
    # Also track single-endpoint events with their resolved node IDs
    # so edges like filtered_by/guarded_by can be matched when events
    # only carry a source_node (e.g. guardrail.triggered).
    runtime_single_nodes: set[str] = set()
    _SINGLE_ENDPOINT_EDGE_TYPES = frozenset({"guarded_by", "filtered_by"})

    for event in runtime_interactions:
        source_node = event.get("source_node")
        target_node = event.get("target_node")

        if source_node and target_node:
            source_id = _resolve_node_id(source_node, structural_nodes)
            target_id = _resolve_node_id(target_node, structural_nodes)
            if source_id and target_id:
                runtime_pairs.add((source_id, target_id))
        elif source_node:
            source_id = _resolve_node_id(source_node, structural_nodes)
            if source_id:
                runtime_single_nodes.add(source_id)
        elif target_node:
            target_id = _resolve_node_id(target_node, structural_nodes)
            if target_id:
                runtime_single_nodes.add(target_id)

    # Find structural edges whose source->target was never observed
    dead_edges: list[dict[str, Any]] = []

    for edge_id, edge_data in structural_edges.items():
        source = edge_data.get("source", "")
        target = edge_data.get("target", "")

        if (source, target) in runtime_pairs:
            continue

        # For edge types like guarded_by/filtered_by, events may only carry
        # one endpoint.  Consider the edge alive if either endpoint appeared
        # in a single-endpoint event.
        edge_type = edge_data.get("edge_type", "unknown")
        if edge_type in _SINGLE_ENDPOINT_EDGE_TYPES:
            if source in runtime_single_nodes or target in runtime_single_nodes:
                continue

        possible_reasons = _infer_dead_reasons(edge_data, total_runs)
        dead_edges.append({
            "edge_id": edge_id,
            "dead": True,
            "runs_observed": total_runs,
            "possible_reasons": possible_reasons,
            "source": source,
            "target": target,
            "edge_type": edge_type,
        })

    return dead_edges


# ---------------------------------------------------------------------------
# Edge significance classification
# ---------------------------------------------------------------------------

def classify_edge_significance(
    source_id: str,
    target_id: str,
    structural_nodes: dict[str, dict[str, Any]],
    events: list[dict[str, Any]],
) -> str:
    """Classify the significance of an emergent edge.

    Returns
    -------
    ``"high"``, ``"medium"``, or ``"low"``.

    Classification rules:
      - high: connects to a node not in the structural graph, or involves
              an irreversible capability (e.g., writes, sends, external calls).
      - medium: connects known nodes but via an unexpected path.
      - low: rare edge that might be input-dependent.
    """
    source_known = source_id in structural_nodes
    target_known = target_id in structural_nodes

    # High: connects to an unknown node
    if not source_known or not target_known:
        return "high"

    # High: involves irreversible capability types
    irreversible_types = {"writes_to", "sends_to", "calls"}
    for ev in events:
        edge_type = ev.get("edge_type", "")
        if edge_type in irreversible_types:
            return "high"
        payload = ev.get("payload", {})
        if payload.get("irreversible", False):
            return "high"

    # Low: very rare (<=2 traversals)
    if len(events) <= 2:
        return "low"

    # Medium: everything else (known nodes, unexpected path)
    return "medium"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_node_id(
    node_ref: dict[str, str],
    structural_nodes: dict[str, dict[str, Any]],
) -> str | None:
    """Resolve a runtime node reference to a structural or runtime node ID.

    Tries direct ID match, then name-based matching, then runtime ID matching.
    Falls back to the raw ID or name if nothing matches structurally.
    """
    # The patcher emits: {"node_type": ..., "node_id": ..., "node_name": ...}
    node_id = node_ref.get("node_id", "") or node_ref.get("id", "")

    # Direct structural match
    if node_id and node_id in structural_nodes:
        return node_id

    # Runtime ID format matching (framework:ClassName:file:line)
    if node_id and ":" in node_id:
        matched = match_runtime_to_structural(node_id, structural_nodes)
        if matched:
            return matched

    # Name-based matching
    name = node_ref.get("node_name", "") or node_ref.get("name", "")
    if name:
        from stratum_lab.node_ids import structural_agent_id, normalize_name
        agent_id = structural_agent_id(name)
        if agent_id in structural_nodes:
            return agent_id
        # Fuzzy match
        normalized = normalize_name(name)
        for sid in structural_nodes:
            if normalized and normalized in sid:
                return sid

    # Fall back to raw ID or name for emergent detection
    return node_id or name or None


def _infer_edge_type(events: list[dict[str, Any]]) -> str:
    """Infer the most likely edge type from the runtime events."""
    type_counts: dict[str, int] = defaultdict(int)
    for ev in events:
        etype = ev.get("edge_type")
        if etype:
            type_counts[etype] += 1
        else:
            # Infer from event type
            event_type = ev.get("event_type", "")
            if "delegat" in event_type:
                type_counts["delegates_to"] += 1
            elif "tool" in event_type:
                type_counts["calls"] += 1
            elif "read" in event_type:
                type_counts["reads_from"] += 1
            elif "write" in event_type:
                type_counts["writes_to"] += 1
            else:
                type_counts["unknown"] += 1

    if not type_counts:
        return "unknown"

    return max(type_counts, key=type_counts.get)  # type: ignore[arg-type]


def _infer_trigger_condition(events: list[dict[str, Any]]) -> str | None:
    """Infer what triggers this emergent edge from event payloads."""
    conditions: set[str] = set()
    for ev in events:
        trigger = ev.get("payload", {}).get("trigger")
        if trigger:
            conditions.add(str(trigger))

    if conditions:
        return "; ".join(sorted(conditions))
    return None


def _infer_dead_reasons(
    edge_data: dict[str, Any],
    total_runs: int,
) -> list[str]:
    """Infer possible reasons why a structural edge was never traversed.

    Returns a list of possible reasons.
    """
    reasons: list[str] = []

    edge_type = edge_data.get("edge_type", "")
    condition = edge_data.get("condition")

    # If the edge has a condition, it might never have been true
    if condition:
        reasons.append("conditional_never_true")

    # If it's gated, the gate might never have opened
    if edge_type in ("gated_by", "filtered_by"):
        reasons.append("conditional_never_true")

    # If total_runs is low, it might be input-dependent
    if total_runs < 5:
        reasons.append("input_dependent")

    # Default reason: could be dead code
    if not reasons:
        reasons.append("dead_code")
        reasons.append("input_dependent")

    return reasons
