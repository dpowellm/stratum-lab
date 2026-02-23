#!/usr/bin/env python3
"""Convert raw stratum_events.jsonl files from multiple runs into
v6 behavioral records for the stratum-graph convergence backend.

Replaces graph_builder.py, risk_analyzer.py, and ecosystem_report.py
with a single pipeline that produces v6 behavioral records.

Usage:
    python build_behavioral_records.py \
        --results-dir results/ \
        --output-dir behavioral_records/ \
        --structural-graphs structural_graphs/  # optional, from security scanner
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from glob import glob
from pathlib import Path

# Allow importing output_classifier from the same scripts directory
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from output_classifier import classify_output

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safe JSON loader
# ---------------------------------------------------------------------------
def load_json_safe(path: str) -> dict:
    """Load a JSON file, returning {} on missing/error."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# Helpers for semantic/defensive merging
# ---------------------------------------------------------------------------
def safe_mean(values: list[float]) -> float:
    """Mean of a list, returning 0.0 if empty."""
    return round(sum(values) / len(values), 3) if values else 0.0


def most_common(items: list[str]) -> str:
    """Return the most common string in a list."""
    if not items:
        return "unknown"
    from collections import Counter
    counts = Counter(items)
    return counts.most_common(1)[0][0]


def merge_edge_semantic_data(structural_edges: list, semantic_edges: list) -> list:
    """Merge structural edge data with semantic delegation fidelity data."""
    if not isinstance(semantic_edges, list):
        return structural_edges
    merged = []
    for s_edge in structural_edges:
        edge_key = tuple(s_edge.get("edge", [s_edge.get("source", ""), s_edge.get("target", "")]))
        matching_semantic = [
            se for se in semantic_edges
            if tuple(se.get("edge", [])) == edge_key
        ]
        merged_edge = {**s_edge}
        if matching_semantic:
            merged_edge["semantic"] = {
                "hedging_preservation_rate": safe_mean([
                    1.0 if m.get("hedging_preserved") else 0.0
                    for m in matching_semantic
                ]),
                "factual_addition_rate": safe_mean([
                    1.0 if m.get("factual_additions_detected") else 0.0
                    for m in matching_semantic
                ]),
                "dominant_mast_failure": most_common([
                    m.get("mast_failure_mode", "unknown")
                    for m in matching_semantic
                ]),
                "dominant_uncertainty_transfer": most_common([
                    m.get("uncertainty_transfer", "unknown")
                    for m in matching_semantic
                ]),
                "evaluation_count": len(matching_semantic),
            }
        merged.append(merged_edge)
    return merged


def merge_node_semantic_data(
    structural_nodes: list, stability_data: list,
    vulnerability_data: list, escalation_data: list,
) -> list:
    """Merge structural node data with all semantic node-level data."""
    merged = []
    stability_by_node = {s["node_id"]: s for s in stability_data} if isinstance(stability_data, list) else {}
    vuln_by_node = {v["node_id"]: v for v in vulnerability_data} if isinstance(vulnerability_data, list) else {}

    escalation_by_node: dict[str, list] = {}
    if isinstance(escalation_data, list):
        for e in escalation_data:
            nid = e.get("node_id", "")
            if nid not in escalation_by_node:
                escalation_by_node[nid] = []
            escalation_by_node[nid].append(e)

    for s_node in structural_nodes:
        nid = s_node.get("node_id", "")
        merged_node = {**s_node}

        if nid in stability_by_node:
            merged_node["stability"] = stability_by_node[nid]

        if nid in vuln_by_node:
            merged_node["vulnerability"] = vuln_by_node[nid]

        if nid in escalation_by_node:
            esc_list = escalation_by_node[nid]
            fab_risk_order = {"high": 3, "medium": 2, "low": 1, "none": 0}
            merged_node["confidence_escalation"] = {
                "escalation_rate": safe_mean([
                    1.0 if e.get("confidence_trajectory") == "escalating" else 0.0
                    for e in esc_list
                ]),
                "max_fabrication_risk": max(
                    (e.get("fabrication_risk", "none") for e in esc_list),
                    key=lambda x: fab_risk_order.get(x, 0),
                ),
                "compensatory_assertion_observed": any(
                    e.get("compensatory_assertion") for e in esc_list
                ),
            }

        merged.append(merged_node)
    return merged


def link_patterns_to_edges(patterns: list, structural_edges: list) -> list:
    """Link defensive patterns to structural edges by file proximity."""
    if not isinstance(patterns, list):
        return []
    return [p for p in patterns if p.get("near_delegation_boundary")]


def detect_semantic_findings(semantic_data: dict, defensive_data: dict) -> list:
    """Detect the 5 new semantically-derived STRAT- findings."""
    findings: list[dict] = []
    agg = semantic_data.get("aggregate_scores", {})

    # STRAT-SD-001: Semantic Drift at Trust Boundary
    df = semantic_data.get("delegation_fidelity", [])
    if isinstance(df, list):
        edges_by_key: dict[tuple, list] = {}
        for entry in df:
            key = tuple(entry.get("edge", []))
            if key not in edges_by_key:
                edges_by_key[key] = []
            edges_by_key[key].append(entry)

        for edge_key, entries in edges_by_key.items():
            valid = [e for e in entries if "parse_error" not in e]
            if len(valid) < 2:
                continue
            drift_rate = sum(
                1 for e in valid
                if not e.get("hedging_preserved") and e.get("factual_additions_detected")
            ) / len(valid)

            if drift_rate > 0.5:
                dominant_mode = most_common([e.get("mast_failure_mode", "unknown") for e in valid])
                findings.append({
                    "finding_id": "STRAT-SD-001",
                    "finding_name": "Semantic Drift at Trust Boundary",
                    "severity": "high",
                    "edge": list(edge_key),
                    "drift_rate": round(drift_rate, 2),
                    "mast_alignment": f"inter_agent_misalignment.{dominant_mode}",
                    "dominant_mast_failure_mode": dominant_mode,
                    "measurement_count": len(valid),
                    "manifestation_observed": True,
                    "remediation_evidence": None,
                })

    # STRAT-HC-001: Hallucination Propagation Chain
    uc = semantic_data.get("uncertainty_chains", [])
    if isinstance(uc, list):
        for chain_result in uc:
            if chain_result.get("information_accretion") and float(chain_result.get("chain_fidelity", 1.0)) < 0.5:
                findings.append({
                    "finding_id": "STRAT-HC-001",
                    "finding_name": "Hallucination Propagation Chain",
                    "severity": "critical",
                    "chain": chain_result.get("chain", []),
                    "accretion_boundary": chain_result.get("accretion_boundary"),
                    "elevation_boundary": chain_result.get("elevation_boundary"),
                    "chain_fidelity": chain_result.get("chain_fidelity"),
                    "run_id": chain_result.get("run_id"),
                    "manifestation_observed": True,
                    "remediation_evidence": None,
                })

    # STRAT-CE-001: Confidence Escalation Under Failure
    ce = semantic_data.get("confidence_escalation", [])
    if isinstance(ce, list):
        for entry in ce:
            if (entry.get("compensatory_assertion")
                    and entry.get("fabrication_risk") == "high"
                    and entry.get("had_preceding_errors")):
                findings.append({
                    "finding_id": "STRAT-CE-001",
                    "finding_name": "Confidence Escalation Under Failure",
                    "severity": "high",
                    "node_id": entry.get("node_id"),
                    "run_id": entry.get("run_id"),
                    "confidence_trajectory": entry.get("confidence_trajectory"),
                    "call_count": entry.get("call_count"),
                    "manifestation_observed": True,
                    "remediation_evidence": None,
                })

    # STRAT-SC-001: Semantic Inconsistency Across Runs
    crc = semantic_data.get("cross_run_consistency", {})
    if isinstance(crc, dict):
        for node in crc.get("node_stability", []):
            if node.get("stability_score", 1.0) < 0.5:
                findings.append({
                    "finding_id": "STRAT-SC-001",
                    "finding_name": "Semantic Inconsistency Across Runs",
                    "severity": "medium",
                    "node_id": node.get("node_id"),
                    "stability_score": node.get("stability_score"),
                    "mean_novel_claims": node.get("mean_novel_claims"),
                    "mean_dropped_claims": node.get("mean_dropped_claims"),
                    "manifestation_observed": True,
                    "remediation_evidence": None,
                })

    # STRAT-TV-001: Topological Vulnerability Concentration
    tv = semantic_data.get("topological_vulnerability", [])
    if isinstance(tv, list):
        for node in tv:
            if (node.get("vulnerability_score", 0) > 0.7
                    and node.get("has_defenses", {}).get("defense_count", 0) == 0):
                findings.append({
                    "finding_id": "STRAT-TV-001",
                    "finding_name": "Topological Vulnerability Concentration",
                    "severity": "high",
                    "node_id": node.get("node_id"),
                    "position_class": node.get("position_class"),
                    "vulnerability_score": node.get("vulnerability_score"),
                    "fan_in": node.get("fan_in"),
                    "error_propagation_reach": node.get("error_propagation_reach"),
                    "missing_defenses": [
                        k.replace("has_", "")
                        for k, v in node.get("has_defenses", {}).items()
                        if k.startswith("has_") and not v
                    ],
                    "manifestation_observed": True,
                    "remediation_evidence": None,
                })

    return findings

# ---------------------------------------------------------------------------
# STRAT failure mode signal definitions
# ---------------------------------------------------------------------------
FAILURE_MODE_SIGNALS = {
    "STRAT-DC-001": {
        "name": "Unsupervised delegation chain",
        "failure_type": "Cascading errors through unsupervised delegation chain",
        "signals": ["no_human_gate_activation", "delegation_depth_>=3"],
    },
    "STRAT-SI-001": {
        "name": "Error laundering",
        "failure_type": "Error silently swallowed, downstream agents process corrupted data",
        "signals": ["silent_error_handling", "error_swallowed"],
    },
    "STRAT-AB-001": {
        "name": "Autonomous behavior drift",
        "failure_type": "Agent behavior drifts from intended scope without detection",
        "signals": ["output_divergence", "unexpected_tool_usage"],
    },
    "STRAT-OC-002": {
        "name": "Concurrent state conflict",
        "failure_type": "Multiple agents write to shared state without coordination",
        "signals": ["concurrent_state_writes", "state_overwrite"],
    },
    "STRAT-EA-001": {
        "name": "External API dependency failure",
        "failure_type": "Agent fails due to external tool/API unavailability",
        "signals": ["tool_call_failure", "api_timeout"],
    },
}

# Mapping from monitoring metrics to STRAT finding IDs
METRIC_TO_FINDING = {
    "max_delegation_depth": "STRAT-DC-001",
    "error_swallow_rate": "STRAT-SI-001",
    "total_llm_calls_per_run": "STRAT-AB-001",
    "concurrent_state_write_rate": "STRAT-OC-002",
    "tool_call_failure_rate": "STRAT-EA-001",
    "delegation_latency_p95_ms": "STRAT-DC-002",
    "schema_mismatch_rate": "STRAT-SI-004",
}


# ---------------------------------------------------------------------------
# Main per-repo pipeline
# ---------------------------------------------------------------------------
def build_behavioral_record(
    repo_hash: str,
    results_dir: str,
    structural_graph: dict | None = None,
) -> dict | None:
    """Build a v6 behavioral record from multiple run event files."""

    # 1. Load all run events
    runs = []
    pattern = os.path.join(results_dir, repo_hash, "events_run_*.jsonl")
    for run_file in sorted(glob(pattern)):
        try:
            with open(run_file, "r", encoding="utf-8") as fh:
                events = [json.loads(line) for line in fh if line.strip()]
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping corrupt event file %s: %s", run_file, exc)
            continue

        meta_file = run_file.replace("events_run_", "run_metadata_").replace(
            ".jsonl", ".json"
        )
        try:
            with open(meta_file, "r", encoding="utf-8") as fh:
                run_meta = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping run (no metadata) %s: %s", meta_file, exc)
            continue

        runs.append({"events": events, "metadata": run_meta})

    if not runs:
        return None

    # 2. Build execution_metadata
    execution_metadata = build_execution_metadata(runs)

    # 3. Extract all unique nodes across runs
    all_nodes = extract_all_nodes(runs)

    # 4. Build structural graph (from scanner or inferred from first run)
    if structural_graph is None:
        structural_graph = infer_structural_graph(runs[0]["events"])

    # 5. Compute edge activation rates
    edge_validation = compute_edge_validation(structural_graph, runs)

    # 6. Detect emergent edges
    emergent_edges = detect_emergent_edges(structural_graph, runs)

    # 7. Classify node activation
    node_activation = classify_node_activation(all_nodes, runs)

    # 8. Trace error propagation
    error_propagation = trace_error_propagation(structural_graph, runs)

    # 9. Classify failure modes
    failure_modes = classify_failure_modes(runs, edge_validation, error_propagation)

    # 10. Compute monitoring baselines
    monitoring_baselines = compute_monitoring_baselines(runs, failure_modes)

    # 11. Load semantic analysis data (v3)
    repo_dir = os.path.join(results_dir, repo_hash)
    semantic_data = load_json_safe(os.path.join(repo_dir, "semantic_analysis.json"))

    # 12. Load defensive patterns data (v3)
    defensive_data = load_json_safe(os.path.join(repo_dir, "defensive_patterns.json"))

    # 13. Detect semantic findings and merge into failure modes
    semantic_findings = detect_semantic_findings(semantic_data, defensive_data)
    all_findings = failure_modes + semantic_findings
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    all_findings.sort(key=lambda f: severity_order.get(f.get("severity", "low"), 4))

    # Build semantic_analysis top-level key
    semantic_analysis_section = {
        "version": semantic_data.get("semantic_analysis_version", "unknown"),
        "model_used": semantic_data.get("model_used", "unknown"),
        "total_llm_calls": semantic_data.get("total_llm_calls", 0),
        "aggregate_scores": semantic_data.get("aggregate_scores", {}),
        "model_confidence": "corpus-calibrated",
        "edges": merge_edge_semantic_data(
            structural_graph.get("edges", []),
            semantic_data.get("delegation_fidelity", []),
        ),
        "nodes": merge_node_semantic_data(
            structural_graph.get("nodes", []),
            semantic_data.get("cross_run_consistency", {}).get("node_stability", [])
            if isinstance(semantic_data.get("cross_run_consistency"), dict) else [],
            semantic_data.get("topological_vulnerability", []),
            semantic_data.get("confidence_escalation", []),
        ),
        "uncertainty_chains": semantic_data.get("uncertainty_chains", []),
    }

    # Build defensive_patterns top-level key
    defensive_patterns_section = {
        "scan_version": defensive_data.get("scan_version", "unknown"),
        "files_scanned": defensive_data.get("files_scanned", 0),
        "total_patterns_found": defensive_data.get("total_patterns_found", 0),
        "delegation_boundaries_found": defensive_data.get("delegation_boundaries_found", 0),
        "summary": defensive_data.get("summary", {}),
        "patterns_at_edges": link_patterns_to_edges(
            defensive_data.get("patterns", []),
            structural_graph.get("edges", []),
        ),
    }

    # 14. Load research enrichments data (v4)
    enrichments_data = load_json_safe(os.path.join(repo_dir, "enrichments.json"))
    research_enrichments = enrichments_data if enrichments_data else None

    return {
        "repo_full_name": runs[0]["metadata"].get("repo_full_name", repo_hash),
        "repo_hash": runs[0]["metadata"].get("repo_full_name", repo_hash),
        "schema_version": "v6",
        "execution_metadata": execution_metadata,
        "edge_validation": edge_validation,
        "emergent_edges": emergent_edges,
        "node_activation": node_activation,
        "error_propagation": error_propagation,
        "failure_modes": all_findings,
        "monitoring_baselines": monitoring_baselines,
        "semantic_analysis": semantic_analysis_section,
        "defensive_patterns": defensive_patterns_section,
        "research_enrichments": research_enrichments,
    }


# ---------------------------------------------------------------------------
# Execution metadata
# ---------------------------------------------------------------------------
def build_execution_metadata(runs: list[dict]) -> dict:
    """Aggregate run metadata into a single execution_metadata block."""
    total_runs = len(runs)
    total_events = sum(len(r["events"]) for r in runs)

    durations = []
    frameworks = set()
    statuses = []

    for run in runs:
        meta = run["metadata"]
        if "duration_seconds" in meta:
            durations.append(meta["duration_seconds"])
        if "framework" in meta:
            frameworks.add(meta["framework"])
        if "status" in meta:
            statuses.append(meta["status"])

    success_count = sum(1 for s in statuses if s in ("SUCCESS", "PARTIAL_SUCCESS", "TIER2_SUCCESS"))

    return {
        "total_runs": total_runs,
        "total_events": total_events,
        "success_rate": round(success_count / max(total_runs, 1), 4),
        "mean_duration_seconds": round(sum(durations) / max(len(durations), 1), 2) if durations else None,
        "frameworks_detected": sorted(frameworks),
        "statuses": statuses,
    }


# ---------------------------------------------------------------------------
# Node extraction
# ---------------------------------------------------------------------------
def extract_all_nodes(runs: list[dict]) -> set[str]:
    """Extract all unique node IDs observed across all runs."""
    nodes: set[str] = set()
    for run in runs:
        for event in run["events"]:
            sn = event.get("source_node", {})
            if sn.get("node_id"):
                nodes.add(sn["node_id"])
            tn = event.get("target_node", {})
            if tn and tn.get("node_id"):
                nodes.add(tn["node_id"])
    return nodes


# ---------------------------------------------------------------------------
# Structural graph inference
# ---------------------------------------------------------------------------
def infer_structural_graph(events: list[dict]) -> dict:
    """Build a structural graph from runtime events.

    Used when the security scanner's structural graph is not available.
    Takes the first run's events and treats all observed connections as
    structural.
    """
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    edge_id_counter = 0

    for event in events:
        sn = event.get("source_node", {})
        if sn.get("node_id") and sn["node_id"] not in nodes:
            nodes[sn["node_id"]] = {
                "node_id": sn["node_id"],
                "node_type": classify_node_type(sn, event),
                "node_name": sn.get("node_name", ""),
            }

    # Extract all edges from events
    runtime_edges = extract_runtime_edges(events)

    # Deduplicate and assign edge IDs
    seen: set[tuple[str, str, str]] = set()
    for re_edge in runtime_edges:
        key = (re_edge["source"], re_edge["target"], re_edge["edge_type"])
        if key not in seen:
            seen.add(key)
            edge_id_counter += 1
            edges.append({
                "edge_id": f"e{edge_id_counter}",
                "source": re_edge["source"],
                "target": re_edge["target"],
                "edge_type": re_edge["edge_type"],
            })

    return {"nodes": list(nodes.values()), "edges": edges}


def classify_node_type(source_node: dict, event: dict) -> str:
    """Classify node into stratum-graph categories."""
    nt = source_node.get("node_type", "")
    et = event.get("event_type", "")
    name = source_node.get("node_name", "").lower()

    if "crew" in name or et == "execution.start":
        return "orchestrator"
    if nt == "capability" or et.startswith("llm."):
        return "capability"
    if "tool" in name or et.startswith("tool."):
        return "capability"
    if et.startswith("agent."):
        return "agent"
    if "state" in name or "store" in name:
        return "data_store"
    if "guard" in name or "review" in name:
        return "guard"
    return "agent"  # default


# ---------------------------------------------------------------------------
# Runtime edge extraction from events
# ---------------------------------------------------------------------------
def extract_runtime_edges(events: list[dict]) -> list[dict]:
    """Extract edges that actually fired from runtime events."""
    edges: list[dict] = []

    for event in events:
        et = event["event_type"]
        sn = event.get("source_node", {})
        payload = event.get("payload", {})

        # Agent -> Crew (orchestration edge)
        if et == "agent.task_start" and payload.get("parent_node_id"):
            edges.append({
                "source": payload["parent_node_id"],
                "target": sn.get("node_id", ""),
                "edge_type": "delegates_to",
            })

        # Agent -> previous agent (data flow edge)
        if et == "agent.task_start" and payload.get("input_source"):
            edges.append({
                "source": payload["input_source"],
                "target": sn.get("node_id", ""),
                "edge_type": "reads_from",
            })

        # Agent -> LLM (capability call edge)
        if et == "llm.call_end":
            stack = payload.get("active_node_stack", [])
            if stack:
                edges.append({
                    "source": stack[-1],
                    "target": sn.get("node_id", ""),
                    "edge_type": "calls",
                })

        # Tool invocation edges
        if et == "tool.invoked":
            edges.append({
                "source": sn.get("node_id", ""),
                "target": payload.get("tool_node_id", ""),
                "edge_type": "uses",
            })

        # State access edges
        if et == "state.access":
            access_type = payload.get("access_type", "")
            edges.append({
                "source": payload.get("accessor_node", ""),
                "target": payload.get("state_key", ""),
                "edge_type": "writes_to" if access_type == "write" else "reads_from",
            })

        # Delegation edges
        if et == "delegation.initiated":
            tn = event.get("target_node", {})
            if tn:
                edges.append({
                    "source": sn.get("node_id", ""),
                    "target": tn.get("node_id", ""),
                    "edge_type": "delegates_to",
                })

        # Routing decision edges
        if et == "routing.decision":
            edges.append({
                "source": payload.get("source_node", ""),
                "target": payload.get("selected_target", ""),
                "edge_type": "delegates_to",
            })

    return edges


# ---------------------------------------------------------------------------
# Edge validation
# ---------------------------------------------------------------------------
def compute_edge_validation(structural_graph: dict, runs: list[dict]) -> dict:
    """Compare structural edges against runtime activation."""
    structural_edges = structural_graph.get("edges", [])
    activation_counts = {e["edge_id"]: 0 for e in structural_edges}
    total_runs = len(runs)

    for run in runs:
        runtime_edges = extract_runtime_edges(run["events"])
        for se in structural_edges:
            if edge_matches_runtime(se, runtime_edges):
                activation_counts[se["edge_id"]] += 1

    activation_rates: dict[str, dict] = {}
    dead_edges: list[dict] = []

    for se in structural_edges:
        rate = activation_counts[se["edge_id"]] / max(total_runs, 1)
        activation_rates[se["edge_id"]] = {
            "traversal_count": activation_counts[se["edge_id"]],
            "activation_rate": round(rate, 4),
            "never_activated": rate == 0.0,
        }
        if rate == 0.0:
            dead_edges.append({
                "edge_id": se["edge_id"],
                "source": se["source"],
                "target": se["target"],
                "edge_type": se.get("edge_type", "unknown"),
                "runs_observed": total_runs,
                "possible_reasons": infer_dead_edge_reason(se, runs),
            })

    return {
        "structural_edges_total": len(structural_edges),
        "structural_edges_activated": sum(
            1 for r in activation_rates.values() if not r["never_activated"]
        ),
        "dead_edges": dead_edges,
        "activation_rates": activation_rates,
    }


def edge_matches_runtime(structural_edge: dict, runtime_edges: list[dict]) -> bool:
    """Check if a structural edge was traversed in a set of runtime edges."""
    s_src = structural_edge["source"]
    s_tgt = structural_edge["target"]
    s_type = structural_edge.get("edge_type", "")

    for re_edge in runtime_edges:
        if re_edge["source"] == s_src and re_edge["target"] == s_tgt:
            # Exact match on source/target; type match is a bonus
            if not s_type or re_edge["edge_type"] == s_type:
                return True
    return False


def infer_dead_edge_reason(structural_edge: dict, runs: list[dict]) -> list[str]:
    """Heuristic reasons why a structural edge was never activated."""
    reasons: list[str] = []
    src = structural_edge["source"]
    tgt = structural_edge["target"]

    src_seen = False
    tgt_seen = False
    for run in runs:
        for event in run["events"]:
            sn_id = event.get("source_node", {}).get("node_id", "")
            tn_id = event.get("target_node", {}).get("node_id", "")
            if sn_id == src or tn_id == src:
                src_seen = True
            if sn_id == tgt or tn_id == tgt:
                tgt_seen = True
            if src_seen and tgt_seen:
                break
        if src_seen and tgt_seen:
            break

    if not src_seen:
        reasons.append(f"source_node_never_activated:{src}")
    if not tgt_seen:
        reasons.append(f"target_node_never_activated:{tgt}")
    if src_seen and tgt_seen:
        reasons.append("nodes_active_but_edge_never_traversed")

    edge_type = structural_edge.get("edge_type", "")
    if edge_type in ("guards", "reviews"):
        reasons.append("guard_edge_may_be_bypass_only")

    return reasons if reasons else ["unknown"]


# ---------------------------------------------------------------------------
# Emergent edge detection
# ---------------------------------------------------------------------------
def detect_emergent_edges(structural_graph: dict, runs: list[dict]) -> list[dict]:
    """Find runtime edges that do NOT exist in the structural graph."""
    structural_keys: set[tuple[str, str]] = set()
    for se in structural_graph.get("edges", []):
        structural_keys.add((se["source"], se["target"]))

    emergent: dict[tuple[str, str, str], int] = {}

    for run in runs:
        runtime_edges = extract_runtime_edges(run["events"])
        seen_this_run: set[tuple[str, str, str]] = set()
        for re_edge in runtime_edges:
            key = (re_edge["source"], re_edge["target"])
            full_key = (re_edge["source"], re_edge["target"], re_edge["edge_type"])
            if key not in structural_keys and full_key not in seen_this_run:
                seen_this_run.add(full_key)
                emergent[full_key] = emergent.get(full_key, 0) + 1

    total_runs = len(runs)
    results: list[dict] = []
    for (src, tgt, etype), count in sorted(emergent.items(), key=lambda x: -x[1]):
        results.append({
            "source": src,
            "target": tgt,
            "edge_type": etype,
            "observed_in_runs": count,
            "observation_rate": round(count / max(total_runs, 1), 4),
            "classification": "consistent" if count == total_runs else "intermittent",
        })

    return results


# ---------------------------------------------------------------------------
# Node activation classification
# ---------------------------------------------------------------------------
def classify_node_activation(all_nodes: set[str], runs: list[dict]) -> dict:
    """Classify nodes as always_active, conditional, or never_active."""
    node_run_presence: dict[str, int] = {node: 0 for node in all_nodes}
    total_runs = len(runs)

    for run in runs:
        active_nodes: set[str] = set()
        for event in run["events"]:
            sn = event.get("source_node", {})
            if sn.get("node_id"):
                active_nodes.add(sn["node_id"])
            tn = event.get("target_node", {})
            if tn and tn.get("node_id"):
                active_nodes.add(tn["node_id"])
        for node in active_nodes:
            if node in node_run_presence:
                node_run_presence[node] += 1

    always_active: list[str] = []
    conditional: list[str] = []
    never_active: list[str] = []

    for node, count in node_run_presence.items():
        if count == total_runs:
            always_active.append(node)
        elif count > 0:
            conditional.append(node)
        else:
            never_active.append(node)

    structural_node_count = len(all_nodes)
    runtime_node_count = len(always_active) + len(conditional)
    match_rate = runtime_node_count / max(structural_node_count, 1)

    return {
        "always_active": sorted(always_active),
        "conditional": sorted(conditional),
        "never_active": sorted(never_active),
        "structural_prediction_match_rate": round(match_rate, 4),
    }


# ---------------------------------------------------------------------------
# Error propagation tracing
# ---------------------------------------------------------------------------
def trace_error_propagation(structural_graph: dict, runs: list[dict]) -> list[dict]:
    """Trace how errors propagate through the agent topology."""
    traces: list[dict] = []

    for run in runs:
        events = run["events"]
        error_events = [e for e in events if e["event_type"] == "error.occurred"]

        for error in error_events:
            source_node_name = error.get("source_node", {}).get("node_name", "unknown")
            error_type = error.get("payload", {}).get("error_type", "unknown")
            source_node_id = error.get("source_node", {}).get("node_id", "")

            # Find structural predicted path (BFS from error source)
            predicted_path = bfs_from_node(structural_graph, source_node_id)

            # Find actual observed path
            actual_path = trace_actual_error_path(events, error)

            # Determine what stopped propagation
            stopped_by = actual_path[-1] if actual_path else "unknown"

            # Determine stop mechanism
            stop_mechanism = _infer_stop_mechanism(events, error, actual_path)

            traces.append({
                "error_source_node": source_node_name,
                "error_type": error_type,
                "structural_predicted_path": predicted_path,
                "actual_observed_path": actual_path,
                "propagation_stopped_by": stopped_by,
                "stop_mechanism": stop_mechanism,
                "downstream_impact": {
                    "nodes_affected": max(len(actual_path) - 1, 0),
                    "downstream_errors": count_downstream_errors(events, error),
                    "downstream_tasks_failed": count_downstream_failures(events, error),
                    "cascade_depth": max(len(actual_path) - 1, 0),
                },
                "swallowed": is_error_swallowed(events, error),
            })

    return traces


def bfs_from_node(structural_graph: dict, start_node_id: str) -> list[str]:
    """BFS traversal from a node in the structural graph.

    Returns the list of node IDs reachable from *start_node_id* following
    outgoing edges, in BFS order.
    """
    if not start_node_id:
        return []

    adjacency: dict[str, list[str]] = {}
    for edge in structural_graph.get("edges", []):
        adjacency.setdefault(edge["source"], []).append(edge["target"])

    visited: list[str] = []
    visited_set: set[str] = set()
    queue = [start_node_id]

    while queue:
        node = queue.pop(0)
        if node in visited_set:
            continue
        visited_set.add(node)
        visited.append(node)
        for neighbor in adjacency.get(node, []):
            if neighbor not in visited_set:
                queue.append(neighbor)

    return visited


def trace_actual_error_path(events: list[dict], error_event: dict) -> list[str]:
    """Trace the actual observed error propagation path.

    Starting from the error event, walks forward through subsequent events
    looking for error-related activity or degraded outputs.
    """
    error_ts = error_event.get("timestamp", "")
    error_source_id = error_event.get("source_node", {}).get("node_id", "")

    path: list[str] = []
    if error_source_id:
        path.append(error_source_id)

    seen: set[str] = {error_source_id}

    # Walk events after the error looking for downstream effects
    past_error = False
    for event in events:
        if event is error_event:
            past_error = True
            continue
        if not past_error:
            continue

        et = event.get("event_type", "")
        node_id = event.get("source_node", {}).get("node_id", "")

        # Downstream error or task failure
        if et in ("error.occurred", "agent.task_end") and node_id and node_id not in seen:
            payload = event.get("payload", {})
            status = payload.get("status", "")
            if et == "error.occurred" or status in ("failed", "error"):
                path.append(node_id)
                seen.add(node_id)

    return path


def _infer_stop_mechanism(
    events: list[dict], error_event: dict, actual_path: list[str]
) -> str:
    """Infer how error propagation was stopped."""
    if not actual_path:
        return "no_propagation"

    last_node = actual_path[-1]

    # Check if there's a retry that succeeded after the error
    past_error = False
    for event in events:
        if event is error_event:
            past_error = True
            continue
        if not past_error:
            continue
        sn_id = event.get("source_node", {}).get("node_id", "")
        if sn_id == last_node:
            et = event.get("event_type", "")
            if et in ("agent.task_end",) and event.get("payload", {}).get("status") == "success":
                return "retry_succeeded"
            if "guard" in sn_id.lower() or "review" in sn_id.lower():
                return "error_boundary"

    return "end_of_chain"


def count_downstream_errors(events: list[dict], error_event: dict) -> int:
    """Count error events that occur after *error_event*."""
    count = 0
    past_error = False
    for event in events:
        if event is error_event:
            past_error = True
            continue
        if past_error and event.get("event_type") == "error.occurred":
            count += 1
    return count


def count_downstream_failures(events: list[dict], error_event: dict) -> int:
    """Count task-end events with a failed status after *error_event*."""
    count = 0
    past_error = False
    for event in events:
        if event is error_event:
            past_error = True
            continue
        if past_error and event.get("event_type") == "agent.task_end":
            status = event.get("payload", {}).get("status", "")
            if status in ("failed", "error"):
                count += 1
    return count


def is_error_swallowed(events: list[dict], error_event: dict) -> bool:
    """Check if an error was swallowed (no downstream error propagation).

    An error is considered swallowed if:
    - The error occurred
    - No subsequent error.occurred events reference related nodes
    - Downstream tasks continued as if nothing happened
    """
    error_source_id = error_event.get("source_node", {}).get("node_id", "")
    past_error = False
    saw_downstream_error = False
    saw_downstream_success = False

    for event in events:
        if event is error_event:
            past_error = True
            continue
        if not past_error:
            continue

        et = event.get("event_type", "")
        if et == "error.occurred":
            saw_downstream_error = True
        if et == "agent.task_end" and event.get("payload", {}).get("status") == "success":
            saw_downstream_success = True

    # Error is swallowed if it occurred but no downstream errors propagated
    # and downstream tasks continued successfully
    return not saw_downstream_error and saw_downstream_success


# ---------------------------------------------------------------------------
# Failure mode classification (STRAT- taxonomy)
# ---------------------------------------------------------------------------
def classify_failure_modes(
    runs: list[dict],
    edge_validation: dict,
    error_propagation: list[dict],
) -> list[dict]:
    """Detect STRAT- failure modes from behavioral signals."""
    findings: list[dict] = []

    for finding_id, spec in FAILURE_MODE_SIGNALS.items():
        signals_detected = detect_signals(
            spec["signals"], runs, edge_validation, error_propagation
        )
        occurrences = sum(s["count"] for s in signals_detected)

        if occurrences > 0:
            findings.append({
                "finding_id": finding_id,
                "finding_name": spec["name"],
                "manifestation_observed": True,
                "failure_type": spec["failure_type"],
                "occurrences": occurrences,
                "downstream_impact": any(
                    has_downstream_impact(s) for s in signals_detected
                ),
                "failure_description": (
                    f"Observed {occurrences} instances across signals: "
                    f"{', '.join(s['signal'] for s in signals_detected if s['count'] > 0)}"
                ),
            })
        else:
            findings.append({
                "finding_id": finding_id,
                "finding_name": spec["name"],
                "manifestation_observed": False,
                "failure_type": spec["failure_type"],
                "occurrences": 0,
                "downstream_impact": False,
            })

    return findings


def detect_signals(
    signal_names: list[str],
    runs: list[dict],
    edge_validation: dict,
    error_propagation: list[dict],
) -> list[dict]:
    """Detect specific behavioral signals across runs."""
    results: list[dict] = []

    for signal in signal_names:
        count = 0

        if signal == "no_human_gate_activation":
            # Check if any guard/review node was never activated
            for run in runs:
                has_guard = False
                for event in run["events"]:
                    sn_name = event.get("source_node", {}).get("node_name", "").lower()
                    et = event.get("event_type", "")
                    if ("guard" in sn_name or "review" in sn_name or "human" in sn_name) \
                            and et not in ("error.occurred",):
                        has_guard = True
                        break
                if not has_guard:
                    count += 1

        elif signal == "delegation_depth_>=3":
            for run in runs:
                depth = _max_delegation_depth(run["events"])
                if depth >= 3:
                    count += 1

        elif signal == "silent_error_handling":
            for trace in error_propagation:
                if trace.get("swallowed", False):
                    count += 1

        elif signal == "error_swallowed":
            for trace in error_propagation:
                if trace.get("swallowed", False):
                    count += 1

        elif signal == "output_divergence":
            # Use output_classifier to detect output divergence across runs
            count += _detect_output_divergence(runs)

        elif signal == "unexpected_tool_usage":
            # Tools used in some runs but not the first run
            count += _detect_unexpected_tools(runs)

        elif signal == "concurrent_state_writes":
            for run in runs:
                count += _count_concurrent_state_writes(run["events"])

        elif signal == "state_overwrite":
            for run in runs:
                count += _count_state_overwrites(run["events"])

        elif signal == "tool_call_failure":
            for run in runs:
                for event in run["events"]:
                    if event.get("event_type") == "tool.invoked":
                        status = event.get("payload", {}).get("status", "")
                        if status in ("failed", "error", "timeout"):
                            count += 1

        elif signal == "api_timeout":
            for run in runs:
                for event in run["events"]:
                    if event.get("event_type") == "tool.invoked":
                        if event.get("payload", {}).get("status") == "timeout":
                            count += 1

        results.append({"signal": signal, "count": count})

    return results


def has_downstream_impact(signal_result: dict) -> bool:
    """Check if a detected signal has downstream impact."""
    return signal_result.get("count", 0) > 0


# ---------------------------------------------------------------------------
# Signal detection helpers
# ---------------------------------------------------------------------------
def _max_delegation_depth(events: list[dict]) -> int:
    """Compute maximum delegation chain depth from events."""
    depth = 0
    current_depth = 0
    for event in events:
        et = event.get("event_type", "")
        if et in ("delegation.initiated", "agent.task_start"):
            payload = event.get("payload", {})
            if payload.get("parent_node_id") or et == "delegation.initiated":
                current_depth += 1
                depth = max(depth, current_depth)
        elif et == "agent.task_end":
            current_depth = max(0, current_depth - 1)
    return depth


def _detect_output_divergence(runs: list[dict]) -> int:
    """Detect output divergence across runs using output_classifier."""
    if len(runs) < 2:
        return 0

    divergence_count = 0
    # Collect output hashes per node across runs
    node_outputs: dict[str, list[str]] = {}

    for run in runs:
        for event in run["events"]:
            if event.get("event_type") in ("agent.task_end", "llm.call_end"):
                node_id = event.get("source_node", {}).get("node_id", "")
                output_hash = event.get("payload", {}).get("output_hash", "")
                if node_id and output_hash:
                    node_outputs.setdefault(node_id, []).append(output_hash)

    # Check for nodes with divergent outputs across runs
    for node_id, hashes in node_outputs.items():
        unique_hashes = set(hashes)
        if len(unique_hashes) > 1:
            # Use output_classifier for enrichment if we have preview text
            divergence_count += 1

    return divergence_count


def _detect_unexpected_tools(runs: list[dict]) -> int:
    """Detect tools used in later runs but not the first run."""
    if len(runs) < 2:
        return 0

    first_run_tools: set[str] = set()
    for event in runs[0]["events"]:
        if event.get("event_type") == "tool.invoked":
            tool_id = event.get("payload", {}).get("tool_node_id", "")
            if tool_id:
                first_run_tools.add(tool_id)

    unexpected_count = 0
    for run in runs[1:]:
        for event in run["events"]:
            if event.get("event_type") == "tool.invoked":
                tool_id = event.get("payload", {}).get("tool_node_id", "")
                if tool_id and tool_id not in first_run_tools:
                    unexpected_count += 1

    return unexpected_count


def _count_concurrent_state_writes(events: list[dict]) -> int:
    """Count cases where multiple agents write to the same state key close together."""
    # Group state writes by state_key with timestamps
    state_writes: dict[str, list[dict]] = {}
    for event in events:
        if event.get("event_type") == "state.access":
            payload = event.get("payload", {})
            if payload.get("access_type") == "write":
                key = payload.get("state_key", "")
                if key:
                    state_writes.setdefault(key, []).append({
                        "accessor": payload.get("accessor_node", ""),
                        "timestamp": event.get("timestamp", ""),
                    })

    count = 0
    for key, writes in state_writes.items():
        unique_accessors = set(w["accessor"] for w in writes)
        if len(unique_accessors) > 1:
            count += 1

    return count


def _count_state_overwrites(events: list[dict]) -> int:
    """Count state overwrites (same key written multiple times)."""
    write_counts: dict[str, int] = {}
    for event in events:
        if event.get("event_type") == "state.access":
            payload = event.get("payload", {})
            if payload.get("access_type") == "write":
                key = payload.get("state_key", "")
                if key:
                    write_counts[key] = write_counts.get(key, 0) + 1

    return sum(1 for c in write_counts.values() if c > 1)


# ---------------------------------------------------------------------------
# Monitoring baselines
# ---------------------------------------------------------------------------
def compute_monitoring_baselines(
    runs: list[dict], failure_modes: list[dict]
) -> list[dict]:
    """Compute per-metric baselines across runs."""
    baselines: list[dict] = []

    for metric_name, finding_id in METRIC_TO_FINDING.items():
        values = [compute_metric(metric_name, run) for run in runs]
        values = [v for v in values if v is not None]

        if not values:
            continue

        mean_val = sum(values) / len(values)
        stddev = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5
        threshold = mean_val + 2 * stddev  # 2-sigma threshold

        baselines.append({
            "metric": metric_name,
            "finding_id": finding_id,
            "observed_baseline": round(mean_val, 4),
            "observed_stddev": round(stddev, 4),
            "suggested_threshold": round(threshold, 4),
            "confidence": "medium" if len(values) >= 3 else "low",
            "sample_size": len(values),
        })

    return baselines


def compute_metric(metric_name: str, run: dict) -> float | None:
    """Compute an individual metric for a single run."""
    events = run["events"]

    if metric_name == "max_delegation_depth":
        depth = _max_delegation_depth(events)
        return float(depth)

    elif metric_name == "error_swallow_rate":
        error_events = [e for e in events if e["event_type"] == "error.occurred"]
        if not error_events:
            return 0.0
        swallowed = sum(1 for e in error_events if is_error_swallowed(events, e))
        return swallowed / len(error_events)

    elif metric_name == "total_llm_calls_per_run":
        return float(sum(
            1 for e in events if e["event_type"] in ("llm.call_start", "llm.call_end")
        ) // 2 or sum(
            1 for e in events if e["event_type"] == "llm.call_end"
        ))

    elif metric_name == "concurrent_state_write_rate":
        total_writes = sum(
            1 for e in events
            if e.get("event_type") == "state.access"
            and e.get("payload", {}).get("access_type") == "write"
        )
        concurrent = _count_concurrent_state_writes(events)
        return concurrent / max(total_writes, 1)

    elif metric_name == "tool_call_failure_rate":
        tool_events = [e for e in events if e["event_type"] == "tool.invoked"]
        if not tool_events:
            return 0.0
        failures = sum(
            1 for e in tool_events
            if e.get("payload", {}).get("status") in ("failed", "error", "timeout")
        )
        return failures / len(tool_events)

    elif metric_name == "delegation_latency_p95_ms":
        latencies: list[float] = []
        for event in events:
            if event.get("event_type") == "delegation.initiated":
                latency = event.get("payload", {}).get("latency_ms")
                if latency is not None:
                    latencies.append(float(latency))
        if not latencies:
            return None
        latencies.sort()
        p95_idx = int(len(latencies) * 0.95)
        return latencies[min(p95_idx, len(latencies) - 1)]

    elif metric_name == "schema_mismatch_rate":
        total_outputs = 0
        mismatches = 0
        for event in events:
            if event.get("event_type") in ("agent.task_end", "llm.call_end"):
                total_outputs += 1
                payload = event.get("payload", {})
                if payload.get("schema_mismatch") or payload.get("validation_error"):
                    mismatches += 1
        return mismatches / max(total_outputs, 1)

    return None


# ---------------------------------------------------------------------------
# Output enrichment via output_classifier
# ---------------------------------------------------------------------------
def enrich_with_output_classification(events: list[dict]) -> list[dict]:
    """Enrich events that have output_preview with output classification.

    Uses output_classifier.classify_output to tag agent/LLM outputs.
    Returns a list of enrichment records.
    """
    enrichments: list[dict] = []
    for event in events:
        payload = event.get("payload", {})
        preview = payload.get("output_preview")
        if preview and event.get("event_type") in ("agent.task_end", "llm.call_end"):
            output_type = payload.get("output_type", "str")
            size_bytes = payload.get("output_size_bytes", len(preview.encode("utf-8")))
            classification = classify_output(preview, output_type, size_bytes)
            enrichments.append({
                "node_id": event.get("source_node", {}).get("node_id", ""),
                "event_type": event["event_type"],
                "classification": classification,
            })
    return enrichments


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw stratum_events.jsonl files into v6 behavioral records."
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing per-repo result folders.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write v6 behavioral record JSON files.",
    )
    parser.add_argument(
        "--structural-graphs",
        default=None,
        help="Optional directory containing structural graph JSON files from the security scanner.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    results_dir = args.results_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Discover repo hashes (subdirectories of results_dir)
    repo_hashes: list[str] = []
    for entry in sorted(os.listdir(results_dir)):
        full_path = os.path.join(results_dir, entry)
        if os.path.isdir(full_path):
            # Verify it has at least one event file
            if glob(os.path.join(full_path, "events_run_*.jsonl")):
                repo_hashes.append(entry)

    logger.info("Found %d repos to process in %s", len(repo_hashes), results_dir)

    success_count = 0
    error_count = 0

    for repo_hash in repo_hashes:
        logger.info("Processing %s ...", repo_hash)

        # Load structural graph if available
        structural_graph = None
        if args.structural_graphs:
            sg_path = os.path.join(args.structural_graphs, f"{repo_hash}.json")
            if os.path.isfile(sg_path):
                try:
                    with open(sg_path, "r", encoding="utf-8") as fh:
                        structural_graph = json.load(fh)
                    logger.debug("Loaded structural graph for %s", repo_hash)
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning(
                        "Failed to load structural graph for %s: %s", repo_hash, exc
                    )

        try:
            record = build_behavioral_record(repo_hash, results_dir, structural_graph)
        except Exception:
            logger.exception("Error processing %s", repo_hash)
            error_count += 1
            continue

        if record is None:
            logger.warning("No valid runs found for %s, skipping.", repo_hash)
            error_count += 1
            continue

        out_path = os.path.join(output_dir, f"{repo_hash}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)

        success_count += 1
        logger.info(
            "  -> %s: %d events, %d runs, %d edges, %d emergent",
            record["repo_full_name"],
            record["execution_metadata"]["total_events"],
            record["execution_metadata"]["total_runs"],
            record["edge_validation"]["structural_edges_total"],
            len(record["emergent_edges"]),
        )

    logger.info(
        "Done. %d records written, %d errors. Output: %s",
        success_count,
        error_count,
        output_dir,
    )


if __name__ == "__main__":
    main()
