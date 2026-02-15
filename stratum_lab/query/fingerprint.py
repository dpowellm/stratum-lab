"""Structural fingerprint computation for graph similarity matching.

Computes a 20-element numeric feature vector and associated structural
metadata from an enriched graph, enabling cosine-similarity-based matching
against the behavioral dataset.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from typing import Any

import numpy as np

from stratum_lab.knowledge.patterns import extract_structural_motifs


# ---------------------------------------------------------------------------
# Node type constants used for counting
# ---------------------------------------------------------------------------

_NODE_TYPE_COUNTS = [
    ("agent", "agent_count"),
    ("capability", "capability_count"),
    ("data_store", "data_store_count"),
    ("external", "external_service_count"),
    ("guardrail", "guardrail_count"),
]

# Edge types that represent capability connections (for control coverage)
_CAPABILITY_EDGE_TYPES = frozenset({
    "calls", "tool_of", "delegates_to", "feeds_into",
    "reads_from", "writes_to", "sends_to", "shares_with",
})

# Edge types that represent guardrail/control connections
_GUARDRAIL_EDGE_TYPES = frozenset({"filtered_by", "gated_by"})

# Edge types that represent observability hooks
_OBSERVABILITY_EDGE_TYPES = frozenset({"filtered_by", "gated_by", "feeds_into"})

# Edge types that indicate irreversible actions
_IRREVERSIBLE_EDGE_TYPES = frozenset({"writes_to", "sends_to", "calls"})


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_graph_fingerprint(graph: dict[str, Any]) -> dict[str, Any]:
    """Compute a structural fingerprint from an enriched graph.

    The fingerprint enables cosine-similarity-based matching against
    the behavioral dataset.

    Parameters
    ----------
    graph:
        A graph dict with ``nodes`` and ``edges``.  Each node/edge may
        have its structural data nested under a ``"structural"`` key or
        stored at the top level.

    Returns
    -------
    Dict with keys:
      - ``feature_vector``: 20-element ``list[float]`` for cosine similarity.
      - ``motifs``: ``list[str]`` of structural motif names present.
      - ``topology_hash``: ``str`` SHA-256 hash of adjacency structure.
      - ``node_type_distribution``: ``dict`` proportions of each node type.
      - ``edge_type_distribution``: ``dict`` proportions of each edge type.
      - ``structural_metrics``: ``dict`` of all individual computed metrics.
    """
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", {})

    # -- Resolve structural data for every node and edge -----------------
    resolved_nodes = _resolve_structural_map(nodes)
    resolved_edges = _resolve_structural_map(edges)

    # -- Basic counts ----------------------------------------------------
    type_counts = _count_node_types(resolved_nodes)
    edge_type_counts = _count_edge_types(resolved_edges)

    agent_count = type_counts.get("agent", 0)
    capability_count = type_counts.get("capability", 0)
    data_store_count = type_counts.get("data_store", 0)
    external_service_count = type_counts.get("external", 0)
    guardrail_count = type_counts.get("guardrail", 0)
    total_edges = len(resolved_edges)

    # -- Delegation depth (BFS from roots through delegates_to) ----------
    delegation_adj = _build_adjacency(resolved_edges, edge_type="delegates_to")
    delegation_depths = _compute_delegation_depths(delegation_adj, resolved_edges)
    delegation_depth_max = max(delegation_depths) if delegation_depths else 0.0
    delegation_depth_mean = (
        float(np.mean(delegation_depths)) if delegation_depths else 0.0
    )

    # -- Control coverage ratio ------------------------------------------
    capability_edge_count = sum(
        1 for s in resolved_edges.values()
        if s.get("edge_type", "") in _CAPABILITY_EDGE_TYPES
    )
    guardrail_edge_count = sum(
        1 for s in resolved_edges.values()
        if s.get("edge_type", "") in _GUARDRAIL_EDGE_TYPES
    )
    control_coverage_ratio = (
        guardrail_edge_count / capability_edge_count
        if capability_edge_count > 0
        else 0.0
    )

    # -- Observability coverage ratio ------------------------------------
    nodes_with_observability = set()
    for s in resolved_edges.values():
        if s.get("edge_type", "") in _OBSERVABILITY_EDGE_TYPES:
            nodes_with_observability.add(s.get("source", ""))
            nodes_with_observability.add(s.get("target", ""))

    agent_and_capability_nodes = {
        nid for nid, s in resolved_nodes.items()
        if s.get("node_type", "") in ("agent", "capability")
    }
    observability_coverage_ratio = (
        len(nodes_with_observability & agent_and_capability_nodes)
        / len(agent_and_capability_nodes)
        if agent_and_capability_nodes
        else 0.0
    )

    # -- Shared state writers --------------------------------------------
    shared_state_writer_count = _count_shared_state_writers(
        resolved_nodes, resolved_edges
    )

    # -- Trust boundary crossings ----------------------------------------
    trust_boundary_crossing_count = _count_trust_boundary_crossings(
        resolved_nodes, resolved_edges
    )

    # -- Feedback loops (cycles) -----------------------------------------
    full_adj = _build_adjacency(resolved_edges)
    feedback_loop_count = _count_cycles(full_adj)

    # -- Hub-spoke ratio -------------------------------------------------
    degrees = _compute_degrees(resolved_nodes, resolved_edges)
    degree_values = list(degrees.values()) if degrees else [0]
    max_degree = max(degree_values) if degree_values else 0
    mean_degree = float(np.mean(degree_values)) if degree_values else 0.0
    hub_spoke_ratio = max_degree / mean_degree if mean_degree > 0 else 0.0

    # -- Betweenness centrality ------------------------------------------
    bc_values = _approximate_betweenness_centrality(resolved_nodes, resolved_edges)
    betweenness_centrality_max = max(bc_values.values()) if bc_values else 0.0
    betweenness_centrality_mean = (
        float(np.mean(list(bc_values.values()))) if bc_values else 0.0
    )

    # -- Error propagation paths -----------------------------------------
    error_propagation_path_count = _count_error_propagation_paths(
        resolved_nodes, resolved_edges
    )

    # -- Irreversible capabilities ---------------------------------------
    irreversible_capability_count = _count_irreversible_capabilities(
        resolved_nodes, resolved_edges
    )

    # -- Agent to capability ratio ---------------------------------------
    agent_to_capability_ratio = (
        agent_count / capability_count if capability_count > 0 else 0.0
    )

    # -- Human checkpoint detection --------------------------------------
    has_human_checkpoint = _detect_human_checkpoint(resolved_nodes, resolved_edges)

    # -- Motifs ----------------------------------------------------------
    motif_results = extract_structural_motifs(graph)
    motif_names = sorted({m["motif_name"] for m in motif_results})

    # -- Topology hash ---------------------------------------------------
    topology_hash = _compute_topology_hash(resolved_edges)

    # -- Distributions ---------------------------------------------------
    total_nodes = len(resolved_nodes)
    node_type_distribution = {
        ntype: round(count / total_nodes, 4) if total_nodes > 0 else 0.0
        for ntype, count in type_counts.items()
    }
    total_edge_count = len(resolved_edges)
    edge_type_distribution = {
        etype: round(count / total_edge_count, 4) if total_edge_count > 0 else 0.0
        for etype, count in edge_type_counts.items()
    }

    # -- Feature vector (20 elements, raw / unnormalized) ----------------
    feature_vector: list[float] = [
        float(agent_count),                     # 1
        float(capability_count),                # 2
        float(data_store_count),                # 3
        float(external_service_count),          # 4
        float(guardrail_count),                 # 5
        float(total_edges),                     # 6
        float(delegation_depth_max),            # 7
        float(delegation_depth_mean),           # 8
        float(control_coverage_ratio),          # 9
        float(observability_coverage_ratio),    # 10
        float(shared_state_writer_count),       # 11
        float(trust_boundary_crossing_count),   # 12
        float(feedback_loop_count),             # 13
        float(hub_spoke_ratio),                 # 14
        float(betweenness_centrality_max),      # 15
        float(betweenness_centrality_mean),     # 16
        float(error_propagation_path_count),    # 17
        float(irreversible_capability_count),   # 18
        float(agent_to_capability_ratio),       # 19
        float(has_human_checkpoint),            # 20
    ]

    structural_metrics: dict[str, Any] = {
        "agent_count": agent_count,
        "capability_count": capability_count,
        "data_store_count": data_store_count,
        "external_service_count": external_service_count,
        "guardrail_count": guardrail_count,
        "total_edges": total_edges,
        "delegation_depth_max": delegation_depth_max,
        "delegation_depth_mean": round(delegation_depth_mean, 4),
        "control_coverage_ratio": round(control_coverage_ratio, 4),
        "observability_coverage_ratio": round(observability_coverage_ratio, 4),
        "shared_state_writer_count": shared_state_writer_count,
        "trust_boundary_crossing_count": trust_boundary_crossing_count,
        "feedback_loop_count": feedback_loop_count,
        "hub_spoke_ratio": round(hub_spoke_ratio, 4),
        "betweenness_centrality_max": round(betweenness_centrality_max, 4),
        "betweenness_centrality_mean": round(betweenness_centrality_mean, 4),
        "error_propagation_path_count": error_propagation_path_count,
        "irreversible_capability_count": irreversible_capability_count,
        "agent_to_capability_ratio": round(agent_to_capability_ratio, 4),
        "has_human_checkpoint": has_human_checkpoint,
    }

    return {
        "feature_vector": feature_vector,
        "motifs": motif_names,
        "topology_hash": topology_hash,
        "node_type_distribution": node_type_distribution,
        "edge_type_distribution": edge_type_distribution,
        "structural_metrics": structural_metrics,
    }


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize_feature_vector(
    feature_vector: list[float],
    normalization_constants: dict[str, dict[str, float]],
) -> list[float]:
    """Normalize a feature vector to [0, 1] using stored min/max constants.

    Parameters
    ----------
    feature_vector:
        A 20-element raw feature vector.
    normalization_constants:
        Dict mapping feature index (as string) to ``{"min": ..., "max": ...}``.

    Returns
    -------
    A 20-element list of floats in [0, 1].
    """
    normalized: list[float] = []

    for i, value in enumerate(feature_vector):
        key = str(i)
        constants = normalization_constants.get(key, {})
        min_val = constants.get("min", 0.0)
        max_val = constants.get("max", 1.0)

        range_val = max_val - min_val
        if range_val > 0:
            norm = (value - min_val) / range_val
        else:
            # All values identical; map to 0.0 (or 0.5 if value == min_val)
            norm = 0.0

        # Clamp to [0, 1]
        normalized.append(max(0.0, min(1.0, norm)))

    return normalized


def compute_normalization_constants(
    fingerprints: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Compute per-feature min/max from a collection of fingerprints.

    Parameters
    ----------
    fingerprints:
        List of fingerprint dicts (each with a ``"feature_vector"`` key).

    Returns
    -------
    Dict mapping feature index (as string) to ``{"min": ..., "max": ...}``.
    """
    if not fingerprints:
        return {}

    vectors = [fp["feature_vector"] for fp in fingerprints]
    vector_length = len(vectors[0])

    constants: dict[str, dict[str, float]] = {}

    for i in range(vector_length):
        values = [v[i] for v in vectors]
        constants[str(i)] = {
            "min": float(min(values)),
            "max": float(max(values)),
        }

    return constants


# ---------------------------------------------------------------------------
# Internal helpers — structural data resolution
# ---------------------------------------------------------------------------

def _resolve_structural_map(
    items: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Extract structural data from a node/edge map.

    For each item, returns ``data.get("structural", data)`` so that
    callers always work with the flat structural fields.
    """
    resolved: dict[str, dict[str, Any]] = {}
    for item_id, data in items.items():
        resolved[item_id] = data.get("structural", data)
    return resolved


# ---------------------------------------------------------------------------
# Internal helpers — counting
# ---------------------------------------------------------------------------

def _count_node_types(
    resolved_nodes: dict[str, dict[str, Any]],
) -> dict[str, int]:
    """Count nodes by ``node_type``."""
    counts: dict[str, int] = defaultdict(int)
    for structural in resolved_nodes.values():
        node_type = structural.get("node_type", "unknown")
        counts[node_type] += 1
    return dict(counts)


def _count_edge_types(
    resolved_edges: dict[str, dict[str, Any]],
) -> dict[str, int]:
    """Count edges by ``edge_type``."""
    counts: dict[str, int] = defaultdict(int)
    for structural in resolved_edges.values():
        edge_type = structural.get("edge_type", "unknown")
        counts[edge_type] += 1
    return dict(counts)


def _count_shared_state_writers(
    resolved_nodes: dict[str, dict[str, Any]],
    resolved_edges: dict[str, dict[str, Any]],
) -> int:
    """Count agents writing to the same data store (shared-state writers).

    Returns the total number of *excess* writers across all data stores —
    i.e. for each data store with N writers, contributes N (only stores
    with 2+ writers are counted).
    """
    data_stores: set[str] = {
        nid for nid, s in resolved_nodes.items()
        if s.get("node_type") == "data_store"
    }

    writers_per_store: dict[str, set[str]] = defaultdict(set)

    for structural in resolved_edges.values():
        if structural.get("edge_type") == "writes_to":
            target = structural.get("target", "")
            source = structural.get("source", "")
            if target in data_stores and source:
                writers_per_store[target].add(source)

    # Count: total writers across stores with 2+ writers
    total = 0
    for writers in writers_per_store.values():
        if len(writers) >= 2:
            total += len(writers)

    return total


def _count_trust_boundary_crossings(
    resolved_nodes: dict[str, dict[str, Any]],
    resolved_edges: dict[str, dict[str, Any]],
) -> int:
    """Count edges that cross trust boundaries.

    A trust boundary crossing occurs when an edge connects an internal
    node (agent, capability, data_store, guardrail) to an external node
    (external, mcp_server).
    """
    external_node_ids: set[str] = {
        nid for nid, s in resolved_nodes.items()
        if s.get("node_type", "") in ("external", "mcp_server")
    }

    count = 0
    for structural in resolved_edges.values():
        source = structural.get("source", "")
        target = structural.get("target", "")
        source_external = source in external_node_ids
        target_external = target in external_node_ids

        if source_external != target_external:
            count += 1

    return count


def _count_irreversible_capabilities(
    resolved_nodes: dict[str, dict[str, Any]],
    resolved_edges: dict[str, dict[str, Any]],
) -> int:
    """Count capability nodes that are connected via irreversible edge types.

    A capability is considered irreversible if it is the source of a
    ``writes_to``, ``sends_to``, or ``calls`` edge targeting an external
    service or data store.
    """
    capability_nodes: set[str] = {
        nid for nid, s in resolved_nodes.items()
        if s.get("node_type") == "capability"
    }

    irreversible_capabilities: set[str] = set()

    for structural in resolved_edges.values():
        edge_type = structural.get("edge_type", "")
        source = structural.get("source", "")

        if edge_type in _IRREVERSIBLE_EDGE_TYPES and source in capability_nodes:
            irreversible_capabilities.add(source)

    # Also count agent nodes that directly perform irreversible actions
    # (in graphs without explicit capability nodes)
    if not capability_nodes:
        agent_nodes = {
            nid for nid, s in resolved_nodes.items()
            if s.get("node_type") == "agent"
        }
        for structural in resolved_edges.values():
            edge_type = structural.get("edge_type", "")
            source = structural.get("source", "")
            if edge_type in _IRREVERSIBLE_EDGE_TYPES and source in agent_nodes:
                irreversible_capabilities.add(source)

    return len(irreversible_capabilities)


def _detect_human_checkpoint(
    resolved_nodes: dict[str, dict[str, Any]],
    resolved_edges: dict[str, dict[str, Any]],
) -> int:
    """Detect whether a human-in-the-loop checkpoint exists.

    Looks for:
      - A guardrail node whose name contains "human" (case-insensitive)
      - A ``gated_by`` edge to a guardrail with human/approval/review hints
      - A node with ``node_type`` containing "human"

    Returns 1 if found, 0 otherwise.
    """
    for structural in resolved_nodes.values():
        node_type = structural.get("node_type", "").lower()
        name = structural.get("name", "").lower()

        if "human" in node_type or "human" in name:
            return 1
        if node_type == "guardrail" and any(
            kw in name
            for kw in ("approval", "review", "confirm", "checkpoint")
        ):
            return 1

    # Check edge labels or properties
    for structural in resolved_edges.values():
        edge_type = structural.get("edge_type", "")
        label = structural.get("label", "").lower()

        if edge_type == "gated_by":
            target = structural.get("target", "")
            target_node = resolved_nodes.get(target, {})
            target_name = target_node.get("name", "").lower()
            if any(
                kw in target_name
                for kw in ("human", "approval", "review", "confirm")
            ):
                return 1

        if "human" in label:
            return 1

    return 0


# ---------------------------------------------------------------------------
# Internal helpers — graph algorithms
# ---------------------------------------------------------------------------

def _build_adjacency(
    resolved_edges: dict[str, dict[str, Any]],
    edge_type: str | None = None,
) -> dict[str, list[str]]:
    """Build a forward adjacency list from resolved edges.

    Parameters
    ----------
    resolved_edges:
        Resolved edge structural data.
    edge_type:
        If given, only include edges of this type.
    """
    adj: dict[str, list[str]] = defaultdict(list)

    for structural in resolved_edges.values():
        etype = structural.get("edge_type", "")
        if edge_type is not None and etype != edge_type:
            continue

        source = structural.get("source", "")
        target = structural.get("target", "")
        if source and target:
            adj[source].append(target)

    return dict(adj)


def _compute_delegation_depths(
    delegation_adj: dict[str, list[str]],
    resolved_edges: dict[str, dict[str, Any]],
) -> list[float]:
    """Compute delegation depths via BFS from root nodes.

    Roots are nodes that appear as sources in delegation edges but never
    as targets.

    Returns
    -------
    List of depths (one per root).  Each depth is the longest path from
    that root through ``delegates_to`` edges.
    """
    if not delegation_adj:
        return []

    # Find roots
    all_sources = set(delegation_adj.keys())
    all_targets: set[str] = set()
    for targets in delegation_adj.values():
        all_targets.update(targets)

    roots = all_sources - all_targets
    if not roots:
        # Cycle: pick all sources as potential roots
        roots = all_sources

    depths: list[float] = []

    for root in roots:
        # BFS to find max depth from this root
        visited: set[str] = {root}
        queue: deque[tuple[str, int]] = deque([(root, 1)])
        max_depth = 1

        while queue:
            node, depth = queue.popleft()
            max_depth = max(max_depth, depth)

            for target in delegation_adj.get(node, []):
                if target not in visited:
                    visited.add(target)
                    queue.append((target, depth + 1))

        depths.append(float(max_depth))

    return depths


def _compute_degrees(
    resolved_nodes: dict[str, dict[str, Any]],
    resolved_edges: dict[str, dict[str, Any]],
) -> dict[str, int]:
    """Compute total degree (in + out) for every node."""
    degrees: dict[str, int] = {nid: 0 for nid in resolved_nodes}

    for structural in resolved_edges.values():
        source = structural.get("source", "")
        target = structural.get("target", "")

        if source in degrees:
            degrees[source] += 1
        elif source:
            degrees[source] = 1

        if target in degrees:
            degrees[target] += 1
        elif target:
            degrees[target] = 1

    return degrees


def _count_cycles(adj: dict[str, list[str]]) -> int:
    """Count distinct cycles in the directed graph using DFS.

    Uses the same cycle-detection approach as
    :func:`~stratum_lab.knowledge.patterns._detect_feedback_loops`,
    but only returns a count.
    """
    all_nodes: set[str] = set(adj.keys())
    for targets in adj.values():
        all_nodes.update(targets)

    visited: set[str] = set()
    rec_stack: set[str] = set()
    cycles: list[tuple[str, ...]] = []

    def _dfs(node: str, path: list[str]) -> None:
        visited.add(node)
        rec_stack.add(node)

        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                _dfs(neighbor, path + [neighbor])
            elif neighbor in rec_stack:
                cycle_start = (
                    path.index(neighbor) if neighbor in path else -1
                )
                if cycle_start >= 0:
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(tuple(cycle))

        rec_stack.discard(node)

    for node in sorted(all_nodes):
        if node not in visited:
            _dfs(node, [node])

    # Deduplicate by normalizing cycle representation
    seen: set[str] = set()
    unique_count = 0

    for cycle in cycles:
        if len(cycle) < 2:
            continue
        core = list(cycle[:-1]) if cycle[0] == cycle[-1] else list(cycle)
        if not core:
            continue
        min_idx = core.index(min(core))
        normalized = core[min_idx:] + core[:min_idx]
        key = "->".join(normalized)
        if key not in seen:
            seen.add(key)
            unique_count += 1

    return unique_count


def _approximate_betweenness_centrality(
    resolved_nodes: dict[str, dict[str, Any]],
    resolved_edges: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """Approximate betweenness centrality using edge-based shortest paths.

    For each pair of nodes (s, t) with s != t, find a shortest path via
    BFS and increment the centrality counter for every intermediate node
    on that path.  The result is normalized by the number of node pairs.

    Parameters
    ----------
    resolved_nodes:
        Resolved structural node data.
    resolved_edges:
        Resolved structural edge data.

    Returns
    -------
    Dict mapping node_id to betweenness centrality (float).
    """
    node_ids = list(resolved_nodes.keys())
    n = len(node_ids)

    if n < 3:
        return {nid: 0.0 for nid in node_ids}

    # Build adjacency
    adj: dict[str, list[str]] = defaultdict(list)
    for structural in resolved_edges.values():
        source = structural.get("source", "")
        target = structural.get("target", "")
        if source and target:
            adj[source].append(target)

    centrality: dict[str, float] = {nid: 0.0 for nid in node_ids}

    # BFS from each node
    for source in node_ids:
        # BFS to find shortest paths from source to all other nodes
        predecessors: dict[str, list[str]] = defaultdict(list)
        distances: dict[str, int] = {source: 0}
        queue: deque[str] = deque([source])
        order: list[str] = []

        while queue:
            current = queue.popleft()
            order.append(current)

            for neighbor in adj.get(current, []):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
                    predecessors[neighbor].append(current)
                elif distances[neighbor] == distances[current] + 1:
                    predecessors[neighbor].append(current)

        # Back-propagation of dependency
        dependency: dict[str, float] = {nid: 0.0 for nid in node_ids}
        num_shortest: dict[str, float] = {nid: 0.0 for nid in node_ids}
        num_shortest[source] = 1.0

        for node in order:
            for pred in predecessors.get(node, []):
                num_shortest[node] += num_shortest[pred]

        for node in reversed(order):
            for pred in predecessors.get(node, []):
                if num_shortest[node] > 0:
                    fraction = num_shortest[pred] / num_shortest[node]
                    dependency[pred] += (1.0 + dependency[node]) * fraction

            if node != source:
                centrality[node] += dependency[node]

    # Normalize: divide by (n-1)*(n-2) for directed graphs
    normalization_factor = (n - 1) * (n - 2) if n > 2 else 1.0
    for nid in centrality:
        centrality[nid] = centrality[nid] / normalization_factor

    return centrality


def _count_error_propagation_paths(
    resolved_nodes: dict[str, dict[str, Any]],
    resolved_edges: dict[str, dict[str, Any]],
) -> int:
    """Count structural error propagation paths.

    An error propagation path is a directed path from a node that
    interacts with an external service (or MCP server) to a node that
    performs an irreversible action — meaning an error at the source
    could propagate and cause irreversible damage.

    Returns the count of distinct (source, sink) pairs connected by a
    directed path.
    """
    # Identify error-source nodes: nodes connected to external services
    external_node_ids: set[str] = {
        nid for nid, s in resolved_nodes.items()
        if s.get("node_type", "") in ("external", "mcp_server")
    }

    # Nodes directly connected to external nodes (could receive errors)
    error_source_nodes: set[str] = set()
    for structural in resolved_edges.values():
        source = structural.get("source", "")
        target = structural.get("target", "")

        if target in external_node_ids and source not in external_node_ids:
            error_source_nodes.add(source)
        if source in external_node_ids and target not in external_node_ids:
            error_source_nodes.add(target)

    # Identify irreversible sink nodes
    irreversible_sinks: set[str] = set()
    for structural in resolved_edges.values():
        edge_type = structural.get("edge_type", "")
        source = structural.get("source", "")
        if edge_type in _IRREVERSIBLE_EDGE_TYPES:
            irreversible_sinks.add(source)

    if not error_source_nodes or not irreversible_sinks:
        return 0

    # Build full adjacency for reachability
    adj: dict[str, list[str]] = defaultdict(list)
    for structural in resolved_edges.values():
        source = structural.get("source", "")
        target = structural.get("target", "")
        if source and target:
            adj[source].append(target)

    # BFS from each error source to find reachable irreversible sinks
    path_count = 0
    for err_source in error_source_nodes:
        visited: set[str] = {err_source}
        queue: deque[str] = deque([err_source])

        while queue:
            current = queue.popleft()

            if current in irreversible_sinks and current != err_source:
                path_count += 1

            for neighbor in adj.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    return path_count


# ---------------------------------------------------------------------------
# Internal helpers — topology hash
# ---------------------------------------------------------------------------

def _compute_topology_hash(
    resolved_edges: dict[str, dict[str, Any]],
) -> str:
    """Hash the sorted list of (source, target, edge_type) tuples.

    Returns a SHA-256 hex digest of the canonical adjacency representation.
    """
    edge_tuples: list[tuple[str, str, str]] = []

    for structural in resolved_edges.values():
        source = structural.get("source", "")
        target = structural.get("target", "")
        edge_type = structural.get("edge_type", "")
        edge_tuples.append((source, target, edge_type))

    edge_tuples.sort()

    canonical = json.dumps(edge_tuples, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
