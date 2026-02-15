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
    for event in runtime_interactions:
        source_node = event.get("source_node")
        target_node = event.get("target_node")
        if not source_node or not target_node:
            continue

        source_id = _resolve_node_id(source_node, structural_nodes)
        target_id = _resolve_node_id(target_node, structural_nodes)

        if source_id and target_id:
            runtime_pairs.add((source_id, target_id))

    # Find structural edges whose source->target was never observed
    dead_edges: list[dict[str, Any]] = []

    for edge_id, edge_data in structural_edges.items():
        source = edge_data.get("source", "")
        target = edge_data.get("target", "")

        if (source, target) not in runtime_pairs:
            possible_reasons = _infer_dead_reasons(edge_data, total_runs)
            dead_edges.append({
                "edge_id": edge_id,
                "dead": True,
                "runs_observed": total_runs,
                "possible_reasons": possible_reasons,
                "source": source,
                "target": target,
                "edge_type": edge_data.get("edge_type", "unknown"),
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
