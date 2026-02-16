"""Feedback export module.

Produces 5 artifact files that the reliability scanner can import:
1. emergent_heuristics.json — detection heuristics for emergent edges
2. edge_confidence_weights.json — activation rates per edge type
3. failure_mode_catalog.json — finding behavioral manifestations
4. monitoring_baselines.json — per-finding metric baselines
5. prediction_match_report.json — structural vs runtime prediction accuracy
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from stratum_lab.config import PRECONDITION_TO_FINDING, FINDING_NAMES


# Valid discovery_type taxonomy (v6)
VALID_DISCOVERY_TYPES = frozenset({
    "error_triggered_fallback",
    "dynamic_delegation",
    "implicit_data_sharing",
    "framework_internal_routing",
})


def _derive_model_context(behavioral_records: List[Dict]) -> Dict[str, Any]:
    """Derive model context from behavioral records' execution_metadata."""
    model = "unknown"
    model_tier = "weak"

    for record in behavioral_records:
        meta = record.get("execution_metadata", {})
        if meta.get("model_name"):
            model = meta["model_name"]
            break
        if meta.get("model"):
            model = meta["model"]
            break

    return {
        "model": model,
        "model_tier": model_tier,
        "purpose": "stress_test",
        "transferability": {
            "propagation_paths": "structural — transfers to production",
            "failure_modes": "structural — transfers to production",
            "trigger_frequencies": "model_dependent — does NOT transfer",
            "absolute_baselines": "model_dependent — use as relative thresholds",
        },
    }


def _to_strat_id(finding_id: str) -> str:
    """Convert a finding_id to STRAT-XX-NNN format if it isn't already."""
    if finding_id.startswith("STRAT-"):
        return finding_id
    return PRECONDITION_TO_FINDING.get(finding_id, finding_id)


def export_feedback(
    behavioral_records: List[Dict],
    output_dir: str | Path,
) -> Dict[str, str]:
    """Export feedback artifacts from behavioral records.

    Parameters
    ----------
    behavioral_records:
        List of per-repo behavioral record dicts.
    output_dir:
        Directory to write the 5 feedback files.

    Returns
    -------
    Dict mapping filename to filepath written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_context = _derive_model_context(behavioral_records)
    files_written = {}

    # 1. Emergent heuristics
    heuristics = _aggregate_emergent_heuristics(behavioral_records)
    heuristics["model_context"] = model_context
    path = output_dir / "emergent_heuristics.json"
    _write_json(path, heuristics)
    files_written["emergent_heuristics.json"] = str(path)

    # 2. Edge confidence weights
    weights = _aggregate_edge_confidence(behavioral_records)
    weights["model_context"] = model_context
    path = output_dir / "edge_confidence_weights.json"
    _write_json(path, weights)
    files_written["edge_confidence_weights.json"] = str(path)

    # 3. Failure mode catalog
    catalog = _aggregate_failure_modes(behavioral_records)
    catalog["model_context"] = model_context
    path = output_dir / "failure_mode_catalog.json"
    _write_json(path, catalog)
    files_written["failure_mode_catalog.json"] = str(path)

    # 4. Monitoring baselines
    baselines = _aggregate_monitoring_baselines(behavioral_records)
    baselines["model_context"] = model_context
    path = output_dir / "monitoring_baselines.json"
    _write_json(path, baselines)
    files_written["monitoring_baselines.json"] = str(path)

    # 5. Prediction match report
    predictions = _aggregate_prediction_match(behavioral_records)
    predictions["model_context"] = model_context
    path = output_dir / "prediction_match_report.json"
    _write_json(path, predictions)
    files_written["prediction_match_report.json"] = str(path)

    return files_written


def _write_json(path: Path, data: Any) -> None:
    """Write data as formatted JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)


def _aggregate_emergent_heuristics(records: List[Dict]) -> Dict:
    """Aggregate emergent edge heuristics across all repos.

    Passes through discovery_type from overlay edges — does NOT re-classify.
    """
    all_heuristics: List[Dict] = []
    by_type: Dict[str, int] = {}

    for record in records:
        for edge in record.get("emergent_edges", []):
            # Pass through discovery_type directly from the overlay
            dtype = edge.get("discovery_type", "unknown")
            by_type[dtype] = by_type.get(dtype, 0) + 1
            all_heuristics.append({
                "repo": record.get("repo_full_name", ""),
                "source_node": edge.get("source_node", ""),
                "target_node": edge.get("target_node", ""),
                "discovery_type": dtype,
                "detection_heuristic": edge.get("detection_heuristic", ""),
                "activation_rate": edge.get("activation_rate", 0),
            })

    return {
        "total_emergent_edges": len(all_heuristics),
        "by_discovery_type": by_type,
        "heuristics": all_heuristics,
    }


def _aggregate_edge_confidence(records: List[Dict]) -> Dict:
    """Aggregate edge activation rates by type."""
    type_activations: Dict[str, List[float]] = {}

    for record in records:
        ev = record.get("edge_validation", {})
        for rate_type, rate_val in ev.get("activation_rates", {}).items():
            type_activations.setdefault(rate_type, []).append(rate_val)

    weights = {}
    for etype, rates in type_activations.items():
        mean_rate = sum(rates) / max(len(rates), 1)
        weights[etype] = {
            "mean_activation_rate": round(mean_rate, 4),
            "sample_size": len(rates),
            "min": round(min(rates), 4) if rates else 0,
            "max": round(max(rates), 4) if rates else 0,
        }

    return {
        "edge_types": weights,
        "repos_analyzed": len(records),
    }


def _aggregate_failure_modes(records: List[Dict]) -> Dict:
    """Aggregate failure mode observations into an enriched catalog.

    Emits STRAT IDs via PRECONDITION_TO_FINDING mapping.
    Includes per-example error_propagation traces where available.
    """
    by_finding: Dict[str, Dict] = {}

    for record in records:
        repo_name = record.get("repo_full_name", "")
        framework = record.get("execution_metadata", {}).get("framework", "")
        error_traces = record.get("error_propagation", [])

        for fm in record.get("failure_modes", []):
            raw_fid = fm.get("finding_id", "")
            fid = _to_strat_id(raw_fid)

            if fid not in by_finding:
                by_finding[fid] = {
                    "finding_id": fid,
                    "finding_name": FINDING_NAMES.get(fid, fm.get("finding_name", fid)),
                    "total_manifestations": 0,
                    "repos_affected": 0,
                    "repos_analyzed_with_finding": 0,
                    "failure_types_observed": [],
                    "examples": [],
                }

            by_finding[fid]["repos_analyzed_with_finding"] += 1

            if fm.get("manifestation_observed"):
                by_finding[fid]["repos_affected"] += 1
                by_finding[fid]["total_manifestations"] += fm.get("occurrences", 0)

                failure_type = fm.get("failure_type", "")
                if failure_type and failure_type not in by_finding[fid]["failure_types_observed"]:
                    by_finding[fid]["failure_types_observed"].append(failure_type)

                # Build enriched example with error_propagation trace
                if len(by_finding[fid]["examples"]) < 15:
                    example = {
                        "repo": repo_name,
                        "framework": framework,
                        "description": fm.get("failure_description", ""),
                    }

                    # Find matching error_propagation trace
                    if error_traces:
                        best_trace = _find_best_trace(fm, error_traces)
                        if best_trace:
                            example["error_propagation"] = _extract_trace_for_catalog(best_trace)

                    by_finding[fid]["examples"].append(example)

    # Build catalog with metadata
    catalog_entries = list(by_finding.values())

    # Add manifestation_rate_note to each finding
    for entry in catalog_entries:
        repos_with = entry["repos_affected"]
        repos_analyzed = entry["repos_analyzed_with_finding"]
        entry["manifestation_rate_note"] = (
            f"Observed in {repos_with}/{repos_analyzed} repos with this structural finding. "
            f"Rate is model-dependent; structural pattern is not."
        )

    return {
        "metadata": {
            "model_tier": "weak",
            "model": "unknown",
            "total_repos_analyzed": len(records),
            "caveat": (
                "Failure modes are structural (architecture-dependent). "
                "Propagation paths transfer to production. "
                "Trigger frequencies are model-dependent and do not transfer."
            ),
        },
        "catalog": catalog_entries,
        "repos_analyzed": len(records),
        # Keep backward-compatible 'findings' key
        "findings": catalog_entries,
    }


def _find_best_trace(failure_mode: Dict, error_traces: List[Dict]) -> Dict | None:
    """Find the best matching error_propagation trace for a failure mode."""
    if not error_traces:
        return None

    # Try to match by finding_id
    fid = failure_mode.get("finding_id", "")
    for trace in error_traces:
        if trace.get("finding_id") == fid:
            return trace

    # Return first trace with actual content
    for trace in error_traces:
        if trace.get("error_source_node") or trace.get("origin_node"):
            return trace

    return error_traces[0] if error_traces else None


def _extract_trace_for_catalog(trace: Dict) -> Dict:
    """Extract error propagation trace fields for the catalog."""
    return {
        "error_source_node": trace.get("error_source_node", trace.get("origin_node", "")),
        "error_type": trace.get("error_type", trace.get("origin_error_type", "")),
        "structural_predicted_path": trace.get("structural_predicted_path",
                                               trace.get("propagation_path", [])),
        "actual_observed_path": trace.get("actual_observed_path",
                                          trace.get("propagation_path", [])),
        "propagation_stopped_by": trace.get("propagation_stopped_by"),
        "stop_mechanism": trace.get("stop_mechanism", trace.get("error_handling_observed", "")),
        "downstream_impact": trace.get("downstream_impact", {}),
    }


def _aggregate_monitoring_baselines(records: List[Dict]) -> Dict:
    """Aggregate monitoring baselines across repos."""
    by_metric: Dict[str, List[Dict]] = {}

    for record in records:
        for baseline in record.get("monitoring_baselines", []):
            metric = baseline.get("metric", "")
            by_metric.setdefault(metric, []).append(baseline)

    aggregated = []
    for metric, baselines in by_metric.items():
        values = [b.get("observed_baseline", 0) for b in baselines]
        mean_val = sum(values) / max(len(values), 1)

        # Use STRAT ID from the first baseline
        finding_id = baselines[0].get("finding_id", "") if baselines else ""
        finding_id = _to_strat_id(finding_id)

        aggregated.append({
            "metric": metric,
            "finding_id": finding_id,
            "scanner_metric": baselines[0].get("scanner_metric", "") if baselines else "",
            "mean_baseline": round(mean_val, 4),
            "sample_repos": len(baselines),
            "per_repo_baselines": baselines,
        })

    return {
        "baselines": aggregated,
        "repos_analyzed": len(records),
    }


def _aggregate_prediction_match(records: List[Dict]) -> Dict:
    """Aggregate structural prediction accuracy across repos."""
    total_edges = 0
    activated_edges = 0
    dead_edges = 0
    match_rates: List[float] = []

    for record in records:
        ev = record.get("edge_validation", {})
        total_edges += ev.get("structural_edges_total", 0)
        activated_edges += ev.get("structural_edges_activated", 0)
        dead_edges += ev.get("structural_edges_dead", 0)

        na = record.get("node_activation", {})
        mr = na.get("structural_prediction_match_rate")
        if mr is not None:
            match_rates.append(mr)

    mean_match = sum(match_rates) / max(len(match_rates), 1) if match_rates else 0

    return {
        "total_structural_edges": total_edges,
        "activated_edges": activated_edges,
        "dead_edges": dead_edges,
        "overall_edge_activation_rate": round(activated_edges / max(total_edges, 1), 4),
        "mean_node_prediction_match_rate": round(mean_match, 4),
        "repos_analyzed": len(records),
    }
