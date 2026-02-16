"""Per-repo behavioral record assembly.

Produces the exact schema that stratum-graph expects for each repo.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _normalize_downstream_impact(impact) -> Dict:
    """Convert downstream_impact to canonical dict form."""
    if isinstance(impact, dict):
        return {
            "nodes_affected": impact.get("nodes_affected", 0),
            "downstream_errors": impact.get("downstream_errors", 0),
            "downstream_tasks_failed": impact.get("downstream_tasks_failed", 0),
            "cascade_depth": impact.get("cascade_depth", 0),
        }
    elif isinstance(impact, bool):
        return {
            "nodes_affected": 1 if impact else 0,
            "downstream_errors": 1 if impact else 0,
            "downstream_tasks_failed": 0,
            "cascade_depth": 1 if impact else 0,
        }
    return {"nodes_affected": 0, "downstream_errors": 0, "downstream_tasks_failed": 0, "cascade_depth": 0}


def _normalize_error_trace(trace: Dict) -> Dict:
    """Normalize error propagation trace to canonical rich schema."""
    return {
        "trace_id": trace.get("trace_id", ""),
        "error_source_node": trace.get("error_source_node", trace.get("origin_node", "")),
        "error_type": trace.get("error_type", trace.get("origin_error_type", "")),
        "structural_predicted_path": trace.get("structural_predicted_path", trace.get("propagation_path", [])),
        "actual_observed_path": trace.get("actual_observed_path", trace.get("propagation_path", [])),
        "propagation_stopped_by": trace.get("propagation_stopped_by", ""),
        "stop_mechanism": trace.get("stop_mechanism", trace.get("error_handling_observed", "")),
        "downstream_impact": _normalize_downstream_impact(trace.get("downstream_impact")),
        "structural_prediction_match": trace.get("structural_prediction_match", False),
        "run_id": trace.get("run_id", ""),
    }


def build_behavioral_record(
    repo_full_name: str,
    execution_metadata: Dict,
    edge_validation: Dict,
    emergent_edges: List[Dict],
    node_activation: Dict,
    error_propagation: List[Dict],
    failure_modes: List[Dict],
    monitoring_baselines: List[Dict],
) -> Dict[str, Any]:
    """Assemble the per-repo behavioral record for stratum-graph.

    Returns the exact schema stratum-graph expects.
    Normalizes error_propagation entries to canonical rich schema.
    """
    return {
        "repo_full_name": repo_full_name,
        "schema_version": "v6",
        "execution_metadata": execution_metadata,
        "edge_validation": edge_validation,
        "emergent_edges": emergent_edges,
        "node_activation": node_activation,
        "error_propagation": [_normalize_error_trace(t) for t in error_propagation],
        "failure_modes": failure_modes,
        "monitoring_baselines": monitoring_baselines,
    }


def validate_behavioral_record(record: Dict) -> tuple[bool, list[str]]:
    """Validate a behavioral record has all required fields.

    Requires canonical rich schema for error_propagation entries:
    - error_source_node (str)
    - structural_predicted_path (list)
    - actual_observed_path (list)
    - stop_mechanism (str)
    - downstream_impact (dict with nodes_affected key)
    """
    required_fields = {
        "repo_full_name",
        "schema_version",
        "execution_metadata",
        "edge_validation",
        "emergent_edges",
        "node_activation",
        "error_propagation",
        "failure_modes",
        "monitoring_baselines",
    }
    missing = required_fields - set(record.keys())
    errors = list(missing)

    # Validate error_propagation canonical schema
    ep = record.get("error_propagation", [])
    if isinstance(ep, list):
        for i, trace in enumerate(ep):
            if not isinstance(trace, dict):
                continue
            # Require canonical field names
            if "error_source_node" not in trace:
                errors.append(f"error_propagation[{i}] missing 'error_source_node'")
            if not isinstance(trace.get("structural_predicted_path"), list):
                errors.append(f"error_propagation[{i}].structural_predicted_path must be list")
            if not isinstance(trace.get("actual_observed_path"), list):
                errors.append(f"error_propagation[{i}].actual_observed_path must be list")
            if not isinstance(trace.get("stop_mechanism"), str):
                errors.append(f"error_propagation[{i}].stop_mechanism must be str")
            di = trace.get("downstream_impact")
            if not isinstance(di, dict) or "nodes_affected" not in di:
                errors.append(f"error_propagation[{i}].downstream_impact must be dict with 'nodes_affected'")
            elif isinstance(di, dict):
                for key in ("nodes_affected", "downstream_errors", "downstream_tasks_failed", "cascade_depth"):
                    if key not in di:
                        errors.append(f"error_propagation[{i}].downstream_impact missing '{key}'")

    return len(errors) == 0 and len(missing) == 0, errors
