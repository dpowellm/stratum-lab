"""Pattern knowledge base builder.

Extracts structural subgraph motifs from enriched graphs, pairs them with
behavioral statistics, and produces a cross-repo pattern knowledge base.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Motif definitions
# ---------------------------------------------------------------------------
# Each motif has a name, a detection function, and a structural signature
# template.

MOTIF_NAMES = [
    "shared_state_without_arbitration",
    "linear_delegation_chain",
    "hub_and_spoke",
    "feedback_loop",
    "trust_boundary_crossing",
]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_pattern_knowledge_base(
    enriched_graphs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the pattern knowledge base from all enriched graphs.

    Parameters
    ----------
    enriched_graphs:
        List of enriched graph dicts (one per repo).

    Returns
    -------
    List of Pattern dicts matching the spec format, each containing:
      - pattern_id, pattern_name, structural_signature
      - prevalence (repos_count, total_repos, prevalence_rate)
      - behavioral_distribution (failure stats with confidence intervals)
      - fragility_data
      - risk_assessment
    """
    total_repos = len(enriched_graphs)
    if total_repos == 0:
        return []

    # Extract motifs from all graphs
    all_motifs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    # Maps motif_name -> list of {repo_id, graph, motif_instances}

    for graph in enriched_graphs:
        repo_id = graph.get("repo_id", "unknown")
        motifs = extract_structural_motifs(graph)
        for motif in motifs:
            motif_name = motif["motif_name"]
            all_motifs[motif_name].append({
                "repo_id": repo_id,
                "graph": graph,
                "motif": motif,
            })

    # Build pattern records
    patterns: list[dict[str, Any]] = []
    pattern_counter = 0

    for motif_name, instances in all_motifs.items():
        pattern_counter += 1

        # Unique repos with this motif
        repos_with_motif = list({inst["repo_id"] for inst in instances})
        graphs_with_motif = [
            inst["graph"] for inst in instances
        ]

        # Behavioral distribution
        behavioral_dist = compute_behavioral_distribution(motif_name, graphs_with_motif)

        # Structural signature from first instance
        structural_sig = instances[0]["motif"].get("structural_signature", {})

        # Fragility: extract model sensitivity data for nodes in this motif
        fragility = _compute_motif_fragility(instances)

        # Risk assessment
        risk = _compute_risk_assessment(behavioral_dist, fragility, len(repos_with_motif))

        sig_hash = hashlib.sha256(
            json.dumps(structural_sig, sort_keys=True).encode()
        ).hexdigest()[:12]

        patterns.append({
            "pattern_id": f"pat_{motif_name}_{sig_hash}",
            "pattern_name": motif_name,
            "structural_signature": structural_sig,
            "prevalence": {
                "repos_count": len(repos_with_motif),
                "total_repos": total_repos,
                "prevalence_rate": round(len(repos_with_motif) / total_repos, 4),
                "repo_ids": repos_with_motif,
            },
            "behavioral_distribution": behavioral_dist,
            "fragility_data": fragility,
            "risk_assessment": risk,
        })

    # Sort by prevalence descending
    patterns.sort(key=lambda p: p["prevalence"]["repos_count"], reverse=True)
    return patterns


# ---------------------------------------------------------------------------
# Structural motif extraction
# ---------------------------------------------------------------------------

def extract_structural_motifs(graph: dict[str, Any]) -> list[dict[str, Any]]:
    """Identify structural patterns (motifs) in an enriched graph.

    Parameters
    ----------
    graph:
        An enriched graph dict with ``nodes`` and ``edges``.

    Returns
    -------
    List of motif dicts, each with ``motif_name``, ``structural_signature``,
    and ``involved_nodes`` / ``involved_edges``.
    """
    motifs: list[dict[str, Any]] = []

    nodes = graph.get("nodes", {})
    edges = graph.get("edges", {})

    motifs.extend(_detect_shared_state_without_arbitration(nodes, edges))
    motifs.extend(_detect_linear_delegation_chains(nodes, edges))
    motifs.extend(_detect_hub_and_spoke(nodes, edges))
    motifs.extend(_detect_feedback_loops(nodes, edges))
    motifs.extend(_detect_trust_boundary_crossings(nodes, edges))

    return motifs


def _detect_shared_state_without_arbitration(
    nodes: dict[str, dict[str, Any]],
    edges: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect 2+ agents writing to the same data_store with no arbitrated_by edge.

    This is a known precondition for race conditions and data corruption.
    """
    motifs: list[dict[str, Any]] = []

    # Find data store nodes
    data_stores: set[str] = set()
    for node_id, node_data in nodes.items():
        structural = node_data.get("structural", node_data)
        if structural.get("node_type") == "data_store":
            data_stores.add(node_id)

    # For each data store, find agents that write to it
    for ds_id in data_stores:
        writers: list[str] = []
        has_arbitration = False

        for edge_id, edge_data in edges.items():
            structural = edge_data.get("structural", edge_data)
            target = structural.get("target", "")
            source = structural.get("source", "")
            edge_type = structural.get("edge_type", "")

            if target == ds_id and edge_type == "writes_to":
                writers.append(source)

            # Check for arbitration edges
            if (source == ds_id or target == ds_id) and edge_type == "arbitrated_by":
                has_arbitration = True

        if len(writers) >= 2 and not has_arbitration:
            motifs.append({
                "motif_name": "shared_state_without_arbitration",
                "structural_signature": {
                    "data_store": ds_id,
                    "writers": writers,
                    "writer_count": len(writers),
                    "has_arbitration": False,
                },
                "involved_nodes": writers + [ds_id],
                "involved_edges": [
                    eid for eid, ed in edges.items()
                    if ed.get("structural", ed).get("target") == ds_id
                    and ed.get("structural", ed).get("edge_type") == "writes_to"
                ],
            })

    return motifs


def _detect_linear_delegation_chains(
    nodes: dict[str, dict[str, Any]],
    edges: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect linear delegation chains A->B->C->D (length >= 3).

    These are potential cascade failure points.
    """
    motifs: list[dict[str, Any]] = []

    # Build delegation adjacency: source -> [targets]
    delegation_adj: dict[str, list[str]] = defaultdict(list)
    delegation_edge_ids: dict[tuple[str, str], str] = {}

    for edge_id, edge_data in edges.items():
        structural = edge_data.get("structural", edge_data)
        if structural.get("edge_type") == "delegates_to":
            source = structural.get("source", "")
            target = structural.get("target", "")
            if source and target:
                delegation_adj[source].append(target)
                delegation_edge_ids[(source, target)] = edge_id

    # Find chain roots: nodes that delegate but are not delegated to
    all_sources = set(delegation_adj.keys())
    all_targets = set()
    for targets in delegation_adj.values():
        all_targets.update(targets)

    roots = all_sources - all_targets

    # Trace chains from roots
    for root in roots:
        chain = _trace_chain(root, delegation_adj)
        if len(chain) >= 3:
            involved_edges = []
            for i in range(len(chain) - 1):
                pair = (chain[i], chain[i + 1])
                if pair in delegation_edge_ids:
                    involved_edges.append(delegation_edge_ids[pair])

            motifs.append({
                "motif_name": "linear_delegation_chain",
                "structural_signature": {
                    "chain": chain,
                    "chain_length": len(chain),
                    "root": root,
                    "leaf": chain[-1],
                },
                "involved_nodes": chain,
                "involved_edges": involved_edges,
            })

    return motifs


def _trace_chain(
    start: str,
    adj: dict[str, list[str]],
    visited: set[str] | None = None,
) -> list[str]:
    """Trace a linear chain from start, following single-outgoing delegation edges."""
    visited = visited or set()
    chain = [start]
    visited.add(start)

    current = start
    while True:
        targets = adj.get(current, [])
        # Only follow linear chains (single delegation target)
        unvisited = [t for t in targets if t not in visited]
        if len(unvisited) != 1:
            break
        next_node = unvisited[0]
        chain.append(next_node)
        visited.add(next_node)
        current = next_node

    return chain


def _detect_hub_and_spoke(
    nodes: dict[str, dict[str, Any]],
    edges: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect hub-and-spoke patterns: one agent delegating to 3+ others."""
    motifs: list[dict[str, Any]] = []

    delegation_out: dict[str, list[str]] = defaultdict(list)
    delegation_edge_ids: dict[tuple[str, str], str] = {}

    for edge_id, edge_data in edges.items():
        structural = edge_data.get("structural", edge_data)
        if structural.get("edge_type") == "delegates_to":
            source = structural.get("source", "")
            target = structural.get("target", "")
            if source and target:
                delegation_out[source].append(target)
                delegation_edge_ids[(source, target)] = edge_id

    for hub, spokes in delegation_out.items():
        if len(spokes) >= 3:
            involved_edges = [
                delegation_edge_ids[(hub, s)]
                for s in spokes
                if (hub, s) in delegation_edge_ids
            ]
            motifs.append({
                "motif_name": "hub_and_spoke",
                "structural_signature": {
                    "hub": hub,
                    "spokes": spokes,
                    "spoke_count": len(spokes),
                },
                "involved_nodes": [hub] + spokes,
                "involved_edges": involved_edges,
            })

    return motifs


def _detect_feedback_loops(
    nodes: dict[str, dict[str, Any]],
    edges: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect cycles in the delegation graph (feedback loops)."""
    motifs: list[dict[str, Any]] = []

    # Build adjacency for all delegation edges
    adj: dict[str, list[str]] = defaultdict(list)
    for edge_id, edge_data in edges.items():
        structural = edge_data.get("structural", edge_data)
        if structural.get("edge_type") in ("delegates_to", "feeds_into"):
            source = structural.get("source", "")
            target = structural.get("target", "")
            if source and target:
                adj[source].append(target)

    # DFS-based cycle detection
    all_nodes = set(adj.keys())
    for targets in adj.values():
        all_nodes.update(targets)

    visited: set[str] = set()
    rec_stack: set[str] = set()
    cycles: list[list[str]] = []

    def _dfs(node: str, path: list[str]) -> None:
        visited.add(node)
        rec_stack.add(node)

        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                _dfs(neighbor, path + [neighbor])
            elif neighbor in rec_stack:
                # Found a cycle: extract from neighbor to current position
                cycle_start = path.index(neighbor) if neighbor in path else -1
                if cycle_start >= 0:
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)

        rec_stack.discard(node)

    for node in all_nodes:
        if node not in visited:
            _dfs(node, [node])

    # Deduplicate cycles (normalize by rotating to smallest element)
    seen_cycles: set[str] = set()
    for cycle in cycles:
        if len(cycle) < 2:
            continue
        # Remove trailing duplicate (the cycle-closing node)
        core = cycle[:-1] if cycle[0] == cycle[-1] else cycle
        if not core:
            continue
        # Normalize by rotating to smallest element
        min_idx = core.index(min(core))
        normalized = core[min_idx:] + core[:min_idx]
        key = "->".join(normalized)
        if key not in seen_cycles:
            seen_cycles.add(key)
            motifs.append({
                "motif_name": "feedback_loop",
                "structural_signature": {
                    "cycle": normalized,
                    "cycle_length": len(normalized),
                },
                "involved_nodes": normalized,
                "involved_edges": [],
            })

    return motifs


def _detect_trust_boundary_crossings(
    nodes: dict[str, dict[str, Any]],
    edges: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect edges that cross trust boundaries.

    A trust boundary crossing occurs when an agent interacts with an
    external service or an MCP server.
    """
    motifs: list[dict[str, Any]] = []

    # Identify external and MCP nodes
    external_nodes: set[str] = set()
    for node_id, node_data in nodes.items():
        structural = node_data.get("structural", node_data)
        node_type = structural.get("node_type", "")
        if node_type in ("external", "mcp_server"):
            external_nodes.add(node_id)

    # Find edges crossing to external nodes
    crossing_edges: list[dict[str, Any]] = []
    for edge_id, edge_data in edges.items():
        structural = edge_data.get("structural", edge_data)
        source = structural.get("source", "")
        target = structural.get("target", "")

        crosses = False
        if source in external_nodes and target not in external_nodes:
            crosses = True
        elif target in external_nodes and source not in external_nodes:
            crosses = True

        if crosses:
            crossing_edges.append({
                "edge_id": edge_id,
                "source": source,
                "target": target,
                "edge_type": structural.get("edge_type", ""),
            })

    if crossing_edges:
        all_involved_nodes = set()
        all_involved_edges = []
        for ce in crossing_edges:
            all_involved_nodes.add(ce["source"])
            all_involved_nodes.add(ce["target"])
            all_involved_edges.append(ce["edge_id"])

        motifs.append({
            "motif_name": "trust_boundary_crossing",
            "structural_signature": {
                "crossing_count": len(crossing_edges),
                "external_nodes": sorted(external_nodes),
                "crossings": crossing_edges,
            },
            "involved_nodes": sorted(all_involved_nodes),
            "involved_edges": all_involved_edges,
        })

    return motifs


# ---------------------------------------------------------------------------
# Behavioral distribution computation
# ---------------------------------------------------------------------------

def compute_behavioral_distribution(
    motif_name: str,
    graphs_with_motif: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute behavioral distribution for a given motif across repos.

    Parameters
    ----------
    motif_name:
        The motif name (e.g., ``"shared_state_without_arbitration"``).
    graphs_with_motif:
        List of enriched graph dicts containing this motif.

    Returns
    -------
    Dict with failure rate, confidence interval, and sample size.
    """
    if not graphs_with_motif:
        return {
            "failure_rate": 0.0,
            "confidence_interval_95": [0.0, 0.0],
            "sample_size": 0,
            "failure_modes": {},
        }

    n = len(graphs_with_motif)
    failure_count = 0
    failure_modes: Counter[str] = Counter()

    # Latencies across all repos for this motif
    latencies: list[float] = []
    error_rates: list[float] = []

    for graph in graphs_with_motif:
        nodes = graph.get("nodes", {})
        has_failure = False

        for node_id, node_data in nodes.items():
            behavioral = node_data.get("behavioral", {})
            error_beh = behavioral.get("error_behavior", {})
            throughput = behavioral.get("throughput", {})

            if error_beh.get("errors_occurred", 0) > 0:
                has_failure = True
                for handling in error_beh.get("observed_error_handling", []):
                    failure_modes[handling] += 1

            fr = throughput.get("failure_rate", 0.0)
            if fr > 0:
                error_rates.append(fr)

            lat = behavioral.get("latency", {})
            p50 = lat.get("p50", 0.0)
            if p50 > 0:
                latencies.append(p50)

        if has_failure:
            failure_count += 1

    # Failure rate across repos
    failure_rate = failure_count / n

    # Wilson score confidence interval for binomial proportion
    ci_low, ci_high = _wilson_confidence_interval(failure_count, n)

    return {
        "failure_rate": round(failure_rate, 4),
        "confidence_interval_95": [round(ci_low, 4), round(ci_high, 4)],
        "sample_size": n,
        "repos_with_failure": failure_count,
        "failure_modes": dict(failure_modes.most_common()),
        "avg_error_rate": round(float(np.mean(error_rates)), 4) if error_rates else 0.0,
        "avg_latency_p50_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
    }


def _wilson_confidence_interval(
    successes: int,
    n: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute Wilson score confidence interval for a binomial proportion.

    Parameters
    ----------
    successes:
        Number of successes (e.g., repos with failure).
    n:
        Total number of trials (e.g., total repos with motif).
    confidence:
        Confidence level (default 0.95).

    Returns
    -------
    (lower, upper) bounds of the confidence interval.
    """
    if n == 0:
        return (0.0, 0.0)

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / n
    denominator = 1 + z**2 / n

    center = (p_hat + z**2 / (2 * n)) / denominator
    spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denominator

    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)

    return (float(lower), float(upper))


# ---------------------------------------------------------------------------
# Novel pattern detection
# ---------------------------------------------------------------------------

def detect_novel_patterns(
    enriched_graphs: list[dict[str, Any]],
    known_taxonomy: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Cluster behavioral data to find failure modes not in the known taxonomy.

    Parameters
    ----------
    enriched_graphs:
        All enriched graphs.
    known_taxonomy:
        List of known taxonomy precondition IDs.

    Returns
    -------
    List of novel pattern dicts with behavioral fingerprints.
    """
    known_taxonomy = known_taxonomy or []

    # Extract behavioral fingerprints: per-repo vectors of
    # [failure_rate, error_propagation_rate, avg_latency, decision_entropy]
    fingerprints: list[dict[str, Any]] = []

    for graph in enriched_graphs:
        repo_id = graph.get("repo_id", "unknown")
        nodes = graph.get("nodes", {})

        failure_rates: list[float] = []
        propagation_rates: list[float] = []
        latencies: list[float] = []
        entropies: list[float] = []

        for node_id, node_data in nodes.items():
            beh = node_data.get("behavioral", {})
            tp = beh.get("throughput", {})
            failure_rates.append(tp.get("failure_rate", 0.0))

            err = beh.get("error_behavior", {})
            propagation_rates.append(err.get("propagation_rate", 0.0))

            lat = beh.get("latency", {})
            latencies.append(lat.get("p50", 0.0))

            dec = beh.get("decision_behavior")
            if dec:
                entropies.append(dec.get("decision_entropy", 0.0))

        if failure_rates:
            fingerprints.append({
                "repo_id": repo_id,
                "avg_failure_rate": float(np.mean(failure_rates)),
                "avg_propagation_rate": float(np.mean(propagation_rates)),
                "avg_latency_p50": float(np.mean(latencies)) if latencies else 0.0,
                "avg_decision_entropy": float(np.mean(entropies)) if entropies else 0.0,
            })

    if len(fingerprints) < 3:
        return []

    # Simple clustering: identify outliers using z-scores on the feature dimensions
    feature_keys = [
        "avg_failure_rate",
        "avg_propagation_rate",
        "avg_latency_p50",
        "avg_decision_entropy",
    ]
    feature_matrix = np.array([
        [fp[k] for k in feature_keys] for fp in fingerprints
    ])

    # Standardize
    means = np.mean(feature_matrix, axis=0)
    stds = np.std(feature_matrix, axis=0)
    stds[stds == 0] = 1.0  # Avoid division by zero
    z_scores = (feature_matrix - means) / stds

    # Flag repos where any feature has |z| > 2 as potentially novel
    novel_patterns: list[dict[str, Any]] = []
    for i, fp in enumerate(fingerprints):
        max_z = float(np.max(np.abs(z_scores[i])))
        if max_z > 2.0:
            # Determine which dimension is most anomalous
            anomalous_dim_idx = int(np.argmax(np.abs(z_scores[i])))
            anomalous_dim = feature_keys[anomalous_dim_idx]

            novel_patterns.append({
                "repo_id": fp["repo_id"],
                "anomaly_score": round(max_z, 4),
                "most_anomalous_dimension": anomalous_dim,
                "behavioral_fingerprint": {
                    k: round(fp[k], 4) for k in feature_keys
                },
                "z_scores": {
                    k: round(float(z_scores[i][j]), 4)
                    for j, k in enumerate(feature_keys)
                },
            })

    # Sort by anomaly score descending
    novel_patterns.sort(key=lambda p: p["anomaly_score"], reverse=True)
    return novel_patterns


# ---------------------------------------------------------------------------
# Framework comparison
# ---------------------------------------------------------------------------

def compare_frameworks(
    enriched_graphs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare behavioral outcomes for the same structural pattern across frameworks.

    Groups repos by motif, then by framework, and computes per-framework
    behavioral summaries.

    Returns
    -------
    List of comparison dicts, one per motif.
    """
    if not enriched_graphs:
        return []

    # Group graphs by framework
    by_framework: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for graph in enriched_graphs:
        fw = graph.get("framework", "unknown")
        by_framework[fw].append(graph)

    # For each motif, compare across frameworks
    comparisons: list[dict[str, Any]] = []

    for motif_name in MOTIF_NAMES:
        fw_results: dict[str, dict[str, Any]] = {}

        for fw, graphs in by_framework.items():
            # Filter graphs that contain this motif
            graphs_with_motif: list[dict[str, Any]] = []
            for g in graphs:
                motifs = extract_structural_motifs(g)
                if any(m["motif_name"] == motif_name for m in motifs):
                    graphs_with_motif.append(g)

            if not graphs_with_motif:
                continue

            dist = compute_behavioral_distribution(motif_name, graphs_with_motif)
            fw_results[fw] = {
                "framework": fw,
                "repos_count": len(graphs_with_motif),
                "behavioral_distribution": dist,
            }

        # Only include motifs observed in 2+ frameworks
        if len(fw_results) >= 2:
            comparisons.append({
                "motif_name": motif_name,
                "frameworks_compared": sorted(fw_results.keys()),
                "per_framework": fw_results,
            })

    return comparisons


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_motif_fragility(
    instances: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute fragility data for nodes involved in a motif."""
    sensitivity_scores: list[float] = []
    tool_failure_rates: list[float] = []

    for inst in instances:
        graph = inst["graph"]
        motif = inst["motif"]
        involved = motif.get("involved_nodes", [])
        nodes = graph.get("nodes", {})

        for node_id in involved:
            if node_id not in nodes:
                continue
            beh = nodes[node_id].get("behavioral", {})
            ms = beh.get("model_sensitivity", {})
            if ms:
                tool_failure_rates.append(ms.get("tool_call_failure_rate", 0.0))
                quality_dep = 1.0 if ms.get("quality_dependent", False) else 0.0
                sensitivity_scores.append(quality_dep)

    return {
        "avg_tool_call_failure_rate": round(
            float(np.mean(tool_failure_rates)), 4
        ) if tool_failure_rates else 0.0,
        "quality_dependent_rate": round(
            float(np.mean(sensitivity_scores)), 4
        ) if sensitivity_scores else 0.0,
        "sample_count": len(tool_failure_rates),
    }


def _compute_risk_assessment(
    behavioral_dist: dict[str, Any],
    fragility: dict[str, Any],
    repos_count: int,
) -> dict[str, Any]:
    """Compute a risk assessment for a pattern based on behavioral and fragility data."""
    failure_rate = behavioral_dist.get("failure_rate", 0.0)
    fragility_rate = fragility.get("avg_tool_call_failure_rate", 0.0)

    # Composite risk score: weighted combination
    risk_score = (failure_rate * 0.5) + (fragility_rate * 0.3) + (
        min(repos_count, 100) / 100 * 0.2
    )

    if risk_score > 0.6:
        risk_level = "high"
    elif risk_score > 0.3:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "risk_score": round(risk_score, 4),
        "risk_level": risk_level,
        "contributing_factors": {
            "failure_rate_contribution": round(failure_rate * 0.5, 4),
            "fragility_contribution": round(fragility_rate * 0.3, 4),
            "prevalence_contribution": round(min(repos_count, 100) / 100 * 0.2, 4),
        },
    }
