"""Fragility map builder.

Identifies model-sensitive structural positions by cross-referencing
tool_call_failure_rate and quality_dependent flags with structural
graph position data.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Structural role classification
# ---------------------------------------------------------------------------

def _classify_structural_role(
    node_id: str,
    nodes: dict[str, dict[str, Any]],
    edges: dict[str, dict[str, Any]],
) -> str:
    """Classify a node's structural role in the graph.

    Returns one of: ``"hub_node"``, ``"leaf_node"``, ``"chain_node"``,
    ``"isolated_node"``, or ``"connector_node"``.
    """
    # Count incoming and outgoing edges
    in_degree = 0
    out_degree = 0

    for edge_data in edges.values():
        structural = edge_data.get("structural", edge_data)
        if structural.get("source") == node_id:
            out_degree += 1
        if structural.get("target") == node_id:
            in_degree += 1

    total_degree = in_degree + out_degree

    if total_degree == 0:
        return "isolated_node"

    # Hub: high out-degree (3+)
    if out_degree >= 3:
        return "hub_node"

    # Leaf: only incoming, no outgoing delegation
    delegation_out = 0
    for edge_data in edges.values():
        structural = edge_data.get("structural", edge_data)
        if (
            structural.get("source") == node_id
            and structural.get("edge_type") == "delegates_to"
        ):
            delegation_out += 1

    if delegation_out == 0 and in_degree > 0:
        return "leaf_node"

    # Chain: exactly 1 incoming delegation and 1 outgoing delegation
    delegation_in = 0
    for edge_data in edges.values():
        structural = edge_data.get("structural", edge_data)
        if (
            structural.get("target") == node_id
            and structural.get("edge_type") == "delegates_to"
        ):
            delegation_in += 1

    if delegation_in == 1 and delegation_out == 1:
        return "chain_node"

    return "connector_node"


# ---------------------------------------------------------------------------
# Sensitivity score
# ---------------------------------------------------------------------------

def compute_sensitivity_score(node_behavioral: dict[str, Any]) -> float:
    """Compute a 0-1 sensitivity score for how model-sensitive a node is.

    The score is a weighted combination of:
      - tool_call_failure_rate (weight: 0.4)
      - quality_dependent flag (weight: 0.3)
      - retry_activations normalized (weight: 0.2)
      - decision entropy normalized (weight: 0.1)

    Parameters
    ----------
    node_behavioral:
        The behavioral overlay dict for a single node.

    Returns
    -------
    Float between 0.0 and 1.0.
    """
    ms = node_behavioral.get("model_sensitivity", {})
    decision_beh = node_behavioral.get("decision_behavior")

    # Tool call failure rate (already 0-1)
    tool_failure_rate = min(ms.get("tool_call_failure_rate", 0.0), 1.0)

    # Quality dependent (binary -> 0 or 1)
    quality_dep = 1.0 if ms.get("quality_dependent", False) else 0.0

    # Retry activations: normalize using sigmoid-like mapping
    retry_count = ms.get("retry_activations", 0)
    retry_normalized = min(retry_count / 10.0, 1.0)  # 10+ retries -> 1.0

    # Decision entropy: higher entropy -> more model-sensitive
    # Normalize: assume max practical entropy is ~4 bits
    entropy = 0.0
    if decision_beh and decision_beh.get("decisions_made", 0) > 0:
        entropy = min(decision_beh.get("decision_entropy", 0.0) / 4.0, 1.0)

    score = (
        tool_failure_rate * 0.4
        + quality_dep * 0.3
        + retry_normalized * 0.2
        + entropy * 0.1
    )

    return round(min(max(score, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# Fragility map
# ---------------------------------------------------------------------------

def build_fragility_map(
    enriched_graphs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build a fragility map identifying model-sensitive structural positions.

    Cross-references tool_call_failure_rate and quality_dependent flags
    with structural position, groups by structural role, and returns
    fragility entries.

    Parameters
    ----------
    enriched_graphs:
        List of enriched graph dicts.

    Returns
    -------
    List of fragility entry dicts, each with:
      - structural_position (role)
      - avg_tool_call_failure_rate
      - sensitivity_score (0-1)
      - affected_repos_count
      - node_details (representative examples)
    """
    # Collect per-role data across all graphs
    role_data: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for graph in enriched_graphs:
        repo_id = graph.get("repo_id", "unknown")
        nodes = graph.get("nodes", {})
        edges = graph.get("edges", {})

        for node_id, node_data in nodes.items():
            behavioral = node_data.get("behavioral", {})
            ms = behavioral.get("model_sensitivity", {})

            # Only include nodes that have some model interaction
            if ms.get("tool_call_failures", 0) == 0 and not ms.get("quality_dependent", False):
                # Check if the node had any LLM or tool activity
                if behavioral.get("activation_count", 0) == 0:
                    continue

            role = _classify_structural_role(node_id, nodes, edges)
            sensitivity = compute_sensitivity_score(behavioral)

            role_data[role].append({
                "repo_id": repo_id,
                "node_id": node_id,
                "tool_call_failure_rate": ms.get("tool_call_failure_rate", 0.0),
                "quality_dependent": ms.get("quality_dependent", False),
                "retry_activations": ms.get("retry_activations", 0),
                "sensitivity_score": sensitivity,
                "activation_count": behavioral.get("activation_count", 0),
                "node_type": node_data.get("structural", {}).get("node_type", "unknown"),
            })

    # Build fragility entries per structural role
    fragility_entries: list[dict[str, Any]] = []

    for role, entries in role_data.items():
        if not entries:
            continue

        failure_rates = [e["tool_call_failure_rate"] for e in entries]
        sensitivity_scores = [e["sensitivity_score"] for e in entries]
        affected_repos = len(set(e["repo_id"] for e in entries))
        quality_dep_count = sum(1 for e in entries if e["quality_dependent"])

        # Top examples sorted by sensitivity score
        sorted_examples = sorted(entries, key=lambda e: e["sensitivity_score"], reverse=True)
        top_examples = sorted_examples[:5]

        fragility_entries.append({
            "structural_position": role,
            "avg_tool_call_failure_rate": round(float(np.mean(failure_rates)), 4),
            "max_tool_call_failure_rate": round(float(np.max(failure_rates)), 4),
            "sensitivity_score": round(float(np.mean(sensitivity_scores)), 4),
            "max_sensitivity_score": round(float(np.max(sensitivity_scores)), 4),
            "affected_repos_count": affected_repos,
            "total_nodes_analyzed": len(entries),
            "quality_dependent_count": quality_dep_count,
            "quality_dependent_rate": round(quality_dep_count / len(entries), 4),
            "avg_retry_activations": round(
                float(np.mean([e["retry_activations"] for e in entries])), 2
            ),
            "top_fragile_nodes": [
                {
                    "repo_id": ex["repo_id"],
                    "node_id": ex["node_id"],
                    "node_type": ex["node_type"],
                    "sensitivity_score": ex["sensitivity_score"],
                    "tool_call_failure_rate": ex["tool_call_failure_rate"],
                }
                for ex in top_examples
            ],
        })

    # Sort by average sensitivity score descending
    fragility_entries.sort(key=lambda e: e["sensitivity_score"], reverse=True)
    return fragility_entries
