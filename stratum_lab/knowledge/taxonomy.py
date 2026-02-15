"""Taxonomy probability computation.

Computes manifestation probabilities for taxonomy preconditions by
cross-referencing structural presence with runtime failure observations.
Also correlates structural metrics with runtime failure rates.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# The 32 taxonomy preconditions
# ---------------------------------------------------------------------------
# These correspond to the stratum taxonomy of failure preconditions for
# multi-agent systems.  Each has an ID used in structural scan output.

TAXONOMY_PRECONDITIONS = [
    "shared_state_no_arbitration",
    "unbounded_delegation_depth",
    "no_timeout_on_delegation",
    "no_output_validation",
    "single_point_of_failure",
    "implicit_ordering_dependency",
    "unhandled_tool_failure",
    "shared_tool_no_concurrency_control",
    "no_fallback_for_external",
    "circular_delegation",
    "no_rate_limiting",
    "data_store_no_schema_enforcement",
    "trust_boundary_no_sanitization",
    "no_error_propagation_strategy",
    "capability_overlap_no_priority",
    "no_guardrail_on_output",
    "unbounded_iteration_loop",
    "no_idempotency_guarantee",
    "missing_input_validation",
    "no_observability_hooks",
    "hardcoded_model_dependency",
    "no_graceful_degradation",
    "unprotected_state_mutation",
    "no_consensus_mechanism",
    "missing_capability_for_task",
    "over_permissioned_agent",
    "no_audit_trail",
    "unencrypted_data_in_transit",
    "no_resource_limits",
    "missing_retry_strategy",
    "no_dead_letter_queue",
    "no_circuit_breaker",
]


# ---------------------------------------------------------------------------
# Manifestation probabilities
# ---------------------------------------------------------------------------

def compute_manifestation_probabilities(
    enriched_graphs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute the probability that each taxonomy precondition manifests as failure.

    For each of the 32 preconditions:
    1. Find all repos where the precondition was structurally present
    2. Compute what percentage showed actual failure at runtime
    3. Compute confidence intervals and severity metrics

    Parameters
    ----------
    enriched_graphs:
        List of enriched graph dicts.

    Returns
    -------
    Dict mapping precondition_id -> {probability, confidence_interval,
    sample_size, severity_when_manifested}.
    """
    results: dict[str, dict[str, Any]] = {}

    # Build a lookup: repo_id -> (preconditions_present, had_failures_by_type)
    repo_data: list[dict[str, Any]] = []
    for graph in enriched_graphs:
        repo_id = graph.get("repo_id", "unknown")

        # Gather taxonomy preconditions from the structural data
        # They can be at graph level or inferred from structural signatures
        structural_preconditions = set()

        # Check graph-level preconditions
        for pc in graph.get("taxonomy_preconditions", []):
            structural_preconditions.add(pc)

        # Also check node-level structural data for precondition hints
        for node_id, node_data in graph.get("nodes", {}).items():
            structural = node_data.get("structural", {})
            for pc in structural.get("taxonomy_preconditions", []):
                structural_preconditions.add(pc)

        # Determine which failure types manifested at runtime
        runtime_failures: dict[str, dict[str, Any]] = {}
        for node_id, node_data in graph.get("nodes", {}).items():
            behavioral = node_data.get("behavioral", {})
            error_beh = behavioral.get("error_behavior", {})
            throughput = behavioral.get("throughput", {})

            if error_beh.get("errors_occurred", 0) > 0:
                for handling in error_beh.get("observed_error_handling", []):
                    runtime_failures[handling] = {
                        "node_id": node_id,
                        "errors": error_beh.get("errors_occurred", 0),
                        "propagation_rate": error_beh.get("propagation_rate", 0.0),
                    }

            if throughput.get("failure_rate", 0.0) > 0:
                runtime_failures[f"throughput_failure_{node_id}"] = {
                    "node_id": node_id,
                    "failure_rate": throughput["failure_rate"],
                }

        repo_data.append({
            "repo_id": repo_id,
            "preconditions": structural_preconditions,
            "had_runtime_failures": len(runtime_failures) > 0,
            "runtime_failures": runtime_failures,
            "total_errors": sum(
                node_data.get("behavioral", {}).get("error_behavior", {}).get(
                    "errors_occurred", 0
                )
                for node_data in graph.get("nodes", {}).values()
            ),
        })

    # For each taxonomy precondition, compute probability
    for precondition_id in TAXONOMY_PRECONDITIONS:
        repos_with_precondition = [
            rd for rd in repo_data
            if precondition_id in rd["preconditions"]
        ]

        n = len(repos_with_precondition)
        if n == 0:
            results[precondition_id] = {
                "probability": None,
                "confidence_interval": [None, None],
                "sample_size": 0,
                "severity_when_manifested": None,
                "repos_with_precondition": 0,
            }
            continue

        # Count repos where the precondition led to actual runtime failure
        manifested = sum(
            1 for rd in repos_with_precondition if rd["had_runtime_failures"]
        )

        probability = manifested / n

        # Wilson score confidence interval
        ci_low, ci_high = _wilson_ci(manifested, n)

        # Severity: average total errors when failure manifested
        error_counts = [
            rd["total_errors"]
            for rd in repos_with_precondition
            if rd["had_runtime_failures"] and rd["total_errors"] > 0
        ]
        avg_severity = float(np.mean(error_counts)) if error_counts else 0.0

        # Classify severity
        if avg_severity > 10:
            severity_label = "high"
        elif avg_severity > 3:
            severity_label = "medium"
        elif avg_severity > 0:
            severity_label = "low"
        else:
            severity_label = "none"

        results[precondition_id] = {
            "probability": round(probability, 4),
            "confidence_interval": [round(ci_low, 4), round(ci_high, 4)],
            "sample_size": n,
            "manifested_count": manifested,
            "repos_with_precondition": n,
            "severity_when_manifested": {
                "avg_errors": round(avg_severity, 2),
                "severity_label": severity_label,
            },
        }

    return results


# ---------------------------------------------------------------------------
# Structural metric correlations
# ---------------------------------------------------------------------------

def compute_structural_metric_correlations(
    enriched_graphs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Correlate structural metrics with runtime failure rates.

    Computes correlations between:
    - Betweenness centrality vs cascade failures
    - Control coverage vs error propagation rates
    - Delegation depth vs timeout rate
    - Node count vs total errors

    Parameters
    ----------
    enriched_graphs:
        List of enriched graph dicts.

    Returns
    -------
    Dict mapping correlation_name -> {correlation, p_value, sample_size,
    interpretation}.
    """
    if len(enriched_graphs) < 3:
        return {
            "note": "Insufficient data for correlation analysis (need >= 3 repos).",
        }

    # Extract per-repo structural metrics and runtime failure metrics
    metrics: list[dict[str, float]] = []

    for graph in enriched_graphs:
        nodes = graph.get("nodes", {})
        edges = graph.get("edges", {})

        node_count = len(nodes)
        edge_count = len(edges)

        # Structural metrics
        delegation_edges = [
            e for e in edges.values()
            if e.get("structural", e).get("edge_type") == "delegates_to"
        ]
        delegation_depth = _estimate_delegation_depth(edges)

        # Betweenness centrality proxy: ratio of edges to nodes
        # (true betweenness requires full graph analysis)
        betweenness_proxy = edge_count / max(node_count, 1)

        # Control coverage: fraction of nodes with guardrail connections
        nodes_with_guardrails = 0
        for node_id, node_data in nodes.items():
            structural = node_data.get("structural", {})
            if structural.get("node_type") == "guardrail":
                nodes_with_guardrails += 1
        # Also count edges to guardrails
        guardrail_edges = sum(
            1 for e in edges.values()
            if e.get("structural", e).get("edge_type") in ("filtered_by", "gated_by")
        )
        control_coverage = guardrail_edges / max(node_count, 1)

        # Runtime metrics
        total_errors = 0
        cascade_failures = 0
        total_propagation_rate = 0.0
        propagation_count = 0
        timeout_events = 0

        for node_id, node_data in nodes.items():
            behavioral = node_data.get("behavioral", {})
            err = behavioral.get("error_behavior", {})
            total_errors += err.get("errors_occurred", 0)
            cascade_failures += err.get("propagated_downstream", 0)

            pr = err.get("propagation_rate", 0.0)
            if pr > 0:
                total_propagation_rate += pr
                propagation_count += 1

        avg_propagation_rate = (
            total_propagation_rate / propagation_count
            if propagation_count > 0
            else 0.0
        )

        metrics.append({
            "node_count": float(node_count),
            "edge_count": float(edge_count),
            "delegation_depth": float(delegation_depth),
            "betweenness_proxy": betweenness_proxy,
            "control_coverage": control_coverage,
            "total_errors": float(total_errors),
            "cascade_failures": float(cascade_failures),
            "avg_propagation_rate": avg_propagation_rate,
        })

    if len(metrics) < 3:
        return {"note": "Insufficient data for correlation analysis."}

    # Compute correlations
    results: dict[str, dict[str, Any]] = {}

    correlation_pairs = [
        ("betweenness_proxy", "cascade_failures", "Betweenness centrality vs cascade failures"),
        ("control_coverage", "avg_propagation_rate", "Control coverage vs error propagation rate"),
        ("delegation_depth", "total_errors", "Delegation depth vs total errors"),
        ("node_count", "total_errors", "Node count vs total errors"),
        ("edge_count", "cascade_failures", "Edge count vs cascade failures"),
    ]

    for x_key, y_key, description in correlation_pairs:
        x_vals = np.array([m[x_key] for m in metrics])
        y_vals = np.array([m[y_key] for m in metrics])

        # Skip if no variance
        if np.std(x_vals) == 0 or np.std(y_vals) == 0:
            results[f"{x_key}_vs_{y_key}"] = {
                "description": description,
                "correlation": None,
                "p_value": None,
                "sample_size": len(metrics),
                "interpretation": "Insufficient variance for correlation.",
            }
            continue

        corr, p_value = stats.pearsonr(x_vals, y_vals)

        # Interpret
        if abs(corr) > 0.7:
            strength = "strong"
        elif abs(corr) > 0.4:
            strength = "moderate"
        elif abs(corr) > 0.2:
            strength = "weak"
        else:
            strength = "negligible"

        direction = "positive" if corr > 0 else "negative"
        sig = "significant" if p_value < 0.05 else "not significant"

        interpretation = (
            f"{strength.capitalize()} {direction} correlation ({sig} at p<0.05)."
        )

        results[f"{x_key}_vs_{y_key}"] = {
            "description": description,
            "correlation": round(float(corr), 4),
            "p_value": round(float(p_value), 6),
            "sample_size": len(metrics),
            "interpretation": interpretation,
        }

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _wilson_ci(
    successes: int,
    n: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score confidence interval."""
    if n == 0:
        return (0.0, 0.0)

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / n
    denom = 1 + z**2 / n

    center = (p_hat + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom

    lower = max(0.0, float(center - spread))
    upper = min(1.0, float(center + spread))

    return (lower, upper)


def _estimate_delegation_depth(edges: dict[str, dict[str, Any]]) -> int:
    """Estimate the maximum delegation depth from edges.

    Builds a delegation adjacency list and finds the longest path.
    """
    adj: dict[str, list[str]] = defaultdict(list)
    for edge_data in edges.values():
        structural = edge_data.get("structural", edge_data)
        if structural.get("edge_type") == "delegates_to":
            source = structural.get("source", "")
            target = structural.get("target", "")
            if source and target:
                adj[source].append(target)

    if not adj:
        return 0

    # Find all roots (nodes that delegate but aren't delegated to)
    all_sources = set(adj.keys())
    all_targets = set()
    for targets in adj.values():
        all_targets.update(targets)

    roots = all_sources - all_targets
    if not roots:
        roots = all_sources  # If there's a cycle, start from any source

    # BFS/DFS to find longest path
    max_depth = 0

    def _dfs(node: str, depth: int, visited: set[str]) -> None:
        nonlocal max_depth
        max_depth = max(max_depth, depth)
        for target in adj.get(node, []):
            if target not in visited:
                visited.add(target)
                _dfs(target, depth + 1, visited)
                visited.discard(target)

    for root in roots:
        _dfs(root, 1, {root})

    return max_depth
