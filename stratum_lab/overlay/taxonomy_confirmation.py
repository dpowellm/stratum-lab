"""Taxonomy precondition confirmation module.

Compares structural taxonomy precondition flags against runtime behavioral
evidence to confirm or deny whether each precondition actually manifested.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


# ---------------------------------------------------------------------------
# Confirmation rules: precondition_id -> detection logic
# ---------------------------------------------------------------------------

def _observed_delegation_chains(enriched_graph: dict[str, Any]) -> list[list[str]]:
    """Extract observed delegation chains from enriched graph edge traversals."""
    edges = enriched_graph.get("edges", {})
    adj: dict[str, list[str]] = defaultdict(list)
    for edge_id, edge_data in edges.items():
        structural = edge_data.get("structural", edge_data)
        behavioral = edge_data.get("behavioral", {})
        if structural.get("edge_type") == "delegates_to" and behavioral.get("traversal_count", 0) > 0:
            adj[structural.get("source", "")].append(structural.get("target", ""))

    chains: list[list[str]] = []
    visited: set[str] = set()
    for start in adj:
        if start in visited:
            continue
        chain = [start]
        visited.add(start)
        current = start
        while True:
            targets = [t for t in adj.get(current, []) if t not in visited]
            if len(targets) != 1:
                break
            chain.append(targets[0])
            visited.add(targets[0])
            current = targets[0]
        if len(chain) >= 2:
            chains.append(chain)
    return chains


def _has_approval_events(enriched_graph: dict[str, Any]) -> bool:
    """Check if any human approval/checkpoint events exist."""
    for node_id, node_data in enriched_graph.get("nodes", {}).items():
        structural = node_data.get("structural", {})
        if structural.get("node_type") in ("human_checkpoint", "approval_gate"):
            behavioral = node_data.get("behavioral", {})
            if behavioral.get("activation_count", 0) > 0:
                return True
    return False


def _reached_irreversible(enriched_graph: dict[str, Any]) -> bool:
    """Check if error propagation reached irreversible capabilities."""
    for node_id, node_data in enriched_graph.get("nodes", {}).items():
        structural = node_data.get("structural", {})
        if structural.get("irreversible", False):
            behavioral = node_data.get("behavioral", {})
            if behavioral.get("error_behavior", {}).get("errors_occurred", 0) > 0:
                return True
    return False


def _has_silent_error_handling(enriched_graph: dict[str, Any]) -> bool:
    """Check for caught_silent or caught_default error handling."""
    for node_id, node_data in enriched_graph.get("nodes", {}).items():
        behavioral = node_data.get("behavioral", {})
        handling = behavioral.get("error_behavior", {}).get("observed_error_handling", [])
        if any(h in ("caught_silent", "caught_default", "fail_silent") for h in handling):
            return True
    return False


def _has_tool_failures_without_fallback(enriched_graph: dict[str, Any]) -> bool:
    """Check for tool failures with no retry or fallback."""
    for node_id, node_data in enriched_graph.get("nodes", {}).items():
        behavioral = node_data.get("behavioral", {})
        ms = behavioral.get("model_sensitivity", {})
        if ms.get("tool_call_failures", 0) > 0 and ms.get("retry_activations", 0) == 0:
            return True
    return False


def _has_shared_state_conflicts(enriched_graph: dict[str, Any]) -> bool:
    """Check for concurrent writes to shared state."""
    for node_id, node_data in enriched_graph.get("nodes", {}).items():
        structural = node_data.get("structural", {})
        if structural.get("node_type") == "data_store":
            behavioral = node_data.get("behavioral", {})
            if behavioral.get("activation_count", 0) > 1:
                return True
    return False


def _has_timeout_events(enriched_graph: dict[str, Any]) -> bool:
    """Check for timeout errors."""
    for node_id, node_data in enriched_graph.get("nodes", {}).items():
        behavioral = node_data.get("behavioral", {})
        handling = behavioral.get("error_behavior", {}).get("observed_error_handling", [])
        if any("timeout" in h.lower() for h in handling):
            return True
    return False


def _has_output_validation_failures(enriched_graph: dict[str, Any]) -> bool:
    """Check for output validation failures or guardrail triggers."""
    for node_id, node_data in enriched_graph.get("nodes", {}).items():
        behavioral = node_data.get("behavioral", {})
        ge = behavioral.get("guardrail_effectiveness")
        if ge and ge.get("trigger_count", 0) > 0:
            return True
    return False


def _has_unvalidated_chain(enriched_graph: dict[str, Any]) -> bool:
    """Check for delegation edges without an intervening guardrail node."""
    edges = enriched_graph.get("edges", {})
    nodes = enriched_graph.get("nodes", {})
    guardrails = {
        nid for nid, n in nodes.items()
        if n.get("structural", {}).get("node_type") == "guardrail"
    }

    for edge_id, edge_data in edges.items():
        structural = edge_data.get("structural", edge_data)
        if structural.get("edge_type") != "delegates_to":
            continue
        behavioral = edge_data.get("behavioral", {})
        if behavioral.get("traversal_count", 0) == 0:
            continue
        source = structural.get("source", "")
        target = structural.get("target", "")
        # Check if any guardrail sits between source and target
        validated = False
        for gid in guardrails:
            incoming = any(
                e.get("structural", e).get("source") == source
                and e.get("structural", e).get("target") == gid
                for e in edges.values()
            )
            outgoing = any(
                e.get("structural", e).get("source") == gid
                and e.get("structural", e).get("target") == target
                for e in edges.values()
            )
            if incoming and outgoing:
                validated = True
                break
        if not validated:
            return True
    return False


def _has_unvalidated_classification(enriched_graph: dict[str, Any]) -> bool:
    """Check for classification/routing output consumed downstream without validation."""
    semantic = enriched_graph.get("semantic_lineage", {})
    if semantic.get("classification_injection_count", 0) > 0:
        if semantic.get("unvalidated_count", 0) > 0:
            return True
    return False


# Main confirmation rules mapping
CONFIRMATION_RULES: dict[str, dict[str, Any]] = {
    "STRAT-DC-001": {
        "name": "deep_delegation_without_checkpoint",
        "precondition_ids": ["unbounded_delegation_depth"],
        "confirm": lambda g: (
            any(len(c) >= 3 for c in _observed_delegation_chains(g))
            and not _has_approval_events(g)
        ),
        "severity_multiplier": lambda g: 2.0 if _reached_irreversible(g) else 1.0,
    },
    "STRAT-SI-001": {
        "name": "error_laundering",
        "precondition_ids": ["no_error_propagation_strategy"],
        "confirm": lambda g: _has_silent_error_handling(g),
        "severity_multiplier": lambda g: 1.5,
    },
    "STRAT-OC-001": {
        "name": "competing_writers",
        "precondition_ids": ["shared_state_no_arbitration"],
        "confirm": lambda g: _has_shared_state_conflicts(g),
        "severity_multiplier": lambda g: 1.0,
    },
    "STRAT-TF-001": {
        "name": "unhandled_tool_failure",
        "precondition_ids": ["unhandled_tool_failure"],
        "confirm": lambda g: _has_tool_failures_without_fallback(g),
        "severity_multiplier": lambda g: 1.0,
    },
    "STRAT-TO-001": {
        "name": "missing_timeout",
        "precondition_ids": ["no_timeout_on_delegation"],
        "confirm": lambda g: _has_timeout_events(g),
        "severity_multiplier": lambda g: 1.0,
    },
    "STRAT-OV-001": {
        "name": "no_output_validation",
        "precondition_ids": ["no_output_validation", "no_guardrail_on_output"],
        "confirm": lambda g: not _has_output_validation_failures(g),
        "severity_multiplier": lambda g: 1.0,
    },
    "STRAT-SC-001": {
        "name": "unvalidated_semantic_chain",
        "precondition_ids": ["unvalidated_semantic_chain"],
        "confirm": lambda g: _has_unvalidated_chain(g),
        "severity_multiplier": lambda g: 2.0,
    },
    "STRAT-SC-002": {
        "name": "classification_without_validation",
        "precondition_ids": ["classification_without_validation"],
        "confirm": lambda g: _has_unvalidated_classification(g),
        "severity_multiplier": lambda g: 2.5,
    },
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def confirm_taxonomy_preconditions(
    enriched_graph: dict[str, Any],
    structural_preconditions: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Confirm or deny taxonomy preconditions using runtime behavioral data.

    Parameters
    ----------
    enriched_graph:
        An enriched graph with behavioral overlays.
    structural_preconditions:
        List of precondition IDs flagged by the structural scan.
        If None, checks graph-level and node-level taxonomy_preconditions.

    Returns
    -------
    List of confirmation result dicts.
    """
    if structural_preconditions is None:
        structural_preconditions = list(enriched_graph.get("taxonomy_preconditions", []))
        for node_data in enriched_graph.get("nodes", {}).values():
            for pc in node_data.get("structural", {}).get("taxonomy_preconditions", []):
                if pc not in structural_preconditions:
                    structural_preconditions.append(pc)

    results: list[dict[str, Any]] = []

    for rule_id, rule in CONFIRMATION_RULES.items():
        # Check if any of the rule's precondition_ids were flagged
        flagged = any(
            pid in structural_preconditions
            for pid in rule.get("precondition_ids", [])
        )
        if not flagged:
            continue

        try:
            confirmed = rule["confirm"](enriched_graph)
        except Exception:
            confirmed = None

        if confirmed is None:
            status = "insufficient_data"
        elif confirmed:
            status = "confirmed"
        else:
            status = "not_observed"

        severity_mult = 1.0
        try:
            severity_mult = rule.get("severity_multiplier", lambda g: 1.0)(enriched_graph)
        except Exception:
            pass

        results.append({
            "precondition_id": rule_id,
            "precondition_name": rule["name"],
            "structural_status": "flagged",
            "runtime_status": status,
            "severity_multiplier": severity_mult,
        })

    return results
