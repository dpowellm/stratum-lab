#!/usr/bin/env python3
"""Per-repo risk analysis for AI agent behavioral graphs.

Takes a graph JSON (from graph_builder.py) and produces a risk report
with 7 risk scores, topology classification, transition risk analysis,
and human-readable findings.

Usage:
    python risk_analyzer.py <graph_json> [-o output.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import reduce
from pathlib import Path

# Import classify_output from sibling module
try:
    from output_classifier import classify_output
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    from output_classifier import classify_output


# ---------------------------------------------------------------------------
# Risky output-category transitions
# ---------------------------------------------------------------------------

RISKY_TRANSITIONS = {
    ("speculative", "generative"): "speculation_amplified",
    ("speculative", "factual"): "speculation_laundered",
    ("generative", "factual"): "fiction_presented_as_fact",
    ("error_empty", "generative"): "hallucination_from_nothing",
}


# ---------------------------------------------------------------------------
# Default risk-score weights (sum to 1.0)
# ---------------------------------------------------------------------------

RISK_WEIGHTS = {
    "propagation_depth": 0.20,
    "single_model_dependency": 0.15,
    "unvalidated_input_rate": 0.20,
    "fan_out_risk": 0.10,
    "concentration_risk": 0.10,
    "model_degradation_resilience": 0.10,
    "compound_chain_risk": 0.15,
}


# ---------------------------------------------------------------------------
# Helpers: normalise graph JSON field names
# ---------------------------------------------------------------------------

def _node_id(node: dict) -> str:
    """Return the node identifier regardless of schema variant."""
    return node.get("id") or node.get("node_id") or ""


def _node_type(node: dict) -> str:
    return node.get("type") or node.get("node_type") or "unknown"


def _node_name(node: dict) -> str:
    return node.get("name") or node.get("node_name") or ""


def _edge_type(edge: dict) -> str:
    return edge.get("type") or edge.get("edge_type") or "unknown"


def _edge_count(edge: dict) -> int:
    return edge.get("count", 0) or edge.get("event_count", 0) or 1


def _cf_from(entry: dict) -> str:
    return entry.get("from_node") or entry.get("producer_node") or ""


def _cf_to(entry: dict) -> str | list:
    """Return consumer(s) -- may be a single id or a list."""
    to = entry.get("to_node")
    if to:
        return to
    return entry.get("consumer_nodes") or ""


def _cf_transformation(entry: dict) -> str:
    return entry.get("transformation_type", "")


# ---------------------------------------------------------------------------
# Topology classification
# ---------------------------------------------------------------------------

def _count_outgoing(node_id: str, edges: list[dict]) -> int:
    return sum(1 for e in edges if e.get("source") == node_id)


def classify_topology(nodes: list[dict], edges: list[dict]) -> str:
    """Classify graph topology into a human-readable label."""
    agent_nodes = [n for n in nodes if _node_type(n) in ("agent", "orchestrator")]
    if len(agent_nodes) <= 1:
        return "single_agent"

    max_fan_out = (
        max(_count_outgoing(_node_id(n), edges) for n in agent_nodes)
        if agent_nodes
        else 0
    )
    has_delegation = any(_edge_type(e) == "delegates_to" for e in edges)

    if has_delegation:
        return "hierarchical_delegation"
    if max_fan_out > 2:
        return "hub_and_spoke"
    return f"sequential_{len(agent_nodes)}_agent"


# ---------------------------------------------------------------------------
# Individual risk scores
# ---------------------------------------------------------------------------

def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def compute_propagation_depth(graph: dict) -> float:
    """chain_depth * (1 - validation_rate).

    validation_rate = edges with a content transformation / total edges.
    """
    risk_indicators = graph.get("risk_indicators") or {}
    chain_depth = risk_indicators.get("max_chain_depth") or risk_indicators.get(
        "max_delegation_depth", 0
    )

    content_flow = graph.get("content_flow") or []
    edges = graph.get("edges") or []
    total_edges = len(edges)

    # Count edges that have a corresponding transformed content-flow entry
    transformed = sum(
        1
        for cf in content_flow
        if _cf_transformation(cf) and _cf_transformation(cf) != "pass_through"
    )
    validation_rate = transformed / max(total_edges, 1)

    return _clamp(chain_depth * (1 - validation_rate))


def compute_single_model_dependency(graph: dict) -> float:
    """1.0 / max(unique_model_count, 1).

    Uses risk_indicators["single_model_dependency"] as a fast-path boolean,
    or counts unique models from node metadata if available.
    """
    risk_indicators = graph.get("risk_indicators") or {}

    # Fast-path: boolean flag from graph_builder
    flag = risk_indicators.get("single_model_dependency")
    if flag is True:
        return 1.0

    # Try counting unique models from node metadata
    unique_models: set[str] = set()
    for node in graph.get("nodes") or []:
        meta = node.get("metadata") or {}
        model = meta.get("model") or meta.get("model_name")
        if model:
            unique_models.add(model)

    if unique_models:
        return _clamp(1.0 / max(len(unique_models), 1))

    # If explicitly False, multiple models exist
    if flag is False:
        return 0.5  # Unknown count, but not single

    return 1.0  # Default conservative: assume single model


def compute_unvalidated_input_rate(graph: dict) -> float:
    """pass_through content_flow entries / total content_flow entries."""
    content_flow = graph.get("content_flow") or []
    if not content_flow:
        return 0.0

    pass_through = sum(
        1 for cf in content_flow if _cf_transformation(cf) == "pass_through"
    )
    return _clamp(pass_through / len(content_flow))


def compute_fan_out_risk(graph: dict) -> float:
    """max(outgoing_edge_count per node) / max(total_nodes, 1)."""
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    if not nodes:
        return 0.0

    max_fan_out = max(
        (_count_outgoing(_node_id(n), edges) for n in nodes), default=0
    )
    return _clamp(max_fan_out / max(len(nodes), 1))


def compute_concentration_risk(graph: dict) -> float:
    """max(llm_calls_per_agent) / max(total_llm_calls, 1).

    Counts from edges of type 'calls_llm'.
    """
    edges = graph.get("edges") or []
    llm_edges = [e for e in edges if _edge_type(e) == "calls_llm"]
    if not llm_edges:
        return 0.0

    total_llm_calls = sum(_edge_count(e) for e in llm_edges)

    # Group by source (agent) node
    calls_per_agent: dict[str, int] = {}
    for e in llm_edges:
        src = e.get("source", "")
        calls_per_agent[src] = calls_per_agent.get(src, 0) + _edge_count(e)

    max_calls = max(calls_per_agent.values(), default=0)
    return _clamp(max_calls / max(total_llm_calls, 1))


def compute_model_degradation_resilience(_graph: dict) -> float:
    """Always 0.0 -- no system detected quality drop in Mistral-as-degradation test."""
    return 0.0


def compute_compound_chain_risk(graph: dict) -> float:
    """1 - product(1 - edge_risk for each edge in longest path).

    edge_risk: 0.1 for validated edges, 0.3 for pass_through, 0.5 for unvalidated.
    Uses max_chain_depth from risk_indicators.
    """
    risk_indicators = graph.get("risk_indicators") or {}
    chain_depth = risk_indicators.get("max_chain_depth") or risk_indicators.get(
        "max_delegation_depth", 0
    )
    if chain_depth <= 0:
        return 0.0

    content_flow = graph.get("content_flow") or []

    # Categorise content-flow entries
    validated_count = 0
    pass_through_count = 0
    for cf in content_flow:
        t = _cf_transformation(cf)
        if t == "pass_through":
            pass_through_count += 1
        elif t:
            validated_count += 1

    total_cf = len(content_flow)
    unvalidated_count = max(chain_depth - validated_count - pass_through_count, 0)

    # Build a list of per-edge risks along the longest path
    edge_risks: list[float] = []
    for _ in range(validated_count):
        edge_risks.append(0.1)
    for _ in range(pass_through_count):
        edge_risks.append(0.3)
    for _ in range(unvalidated_count):
        edge_risks.append(0.5)

    # Trim or pad to chain_depth
    edge_risks = edge_risks[:chain_depth]
    while len(edge_risks) < chain_depth:
        edge_risks.append(0.5)  # Assume unvalidated for unknown edges

    survival = reduce(lambda acc, r: acc * (1 - r), edge_risks, 1.0)
    return _clamp(1 - survival)


# ---------------------------------------------------------------------------
# Aggregate risk scores
# ---------------------------------------------------------------------------

def compute_risk_scores(graph: dict) -> dict:
    """Compute all 7 risk scores and the weighted overall score."""
    scores = {
        "propagation_depth": round(compute_propagation_depth(graph), 4),
        "single_model_dependency": round(compute_single_model_dependency(graph), 4),
        "unvalidated_input_rate": round(compute_unvalidated_input_rate(graph), 4),
        "fan_out_risk": round(compute_fan_out_risk(graph), 4),
        "concentration_risk": round(compute_concentration_risk(graph), 4),
        "model_degradation_resilience": round(
            compute_model_degradation_resilience(graph), 4
        ),
        "compound_chain_risk": round(compute_compound_chain_risk(graph), 4),
    }

    overall = sum(scores[k] * RISK_WEIGHTS[k] for k in RISK_WEIGHTS)
    scores["overall_risk"] = round(_clamp(overall), 4)
    return scores


# ---------------------------------------------------------------------------
# Output classification per node
# ---------------------------------------------------------------------------

def classify_node_outputs(graph: dict) -> dict:
    """Run output_classifier.classify_output on each node with output metadata.

    Returns {node_name: {"primary": ..., "confidence": ...}}.
    """
    classifications: dict[str, dict] = {}
    for node in graph.get("nodes") or []:
        meta = node.get("metadata") or {}
        preview = meta.get("output_preview")
        output_type = meta.get("output_type", "str")
        size_bytes = meta.get("output_size_bytes")

        if preview is None or size_bytes is None:
            continue

        result = classify_output(preview, output_type, size_bytes)
        name = _node_name(node) or _node_id(node)
        classifications[name] = {
            "primary": result["primary"],
            "confidence": result["confidence"],
        }

    return classifications


# ---------------------------------------------------------------------------
# Transition risk analysis
# ---------------------------------------------------------------------------

def analyse_transition_risks(
    graph: dict, classifications: dict
) -> list[dict]:
    """For each content_flow entry, check if the producer->consumer transition
    is a risky output-category pair.
    """
    content_flow = graph.get("content_flow") or []
    nodes = graph.get("nodes") or []

    # Build quick lookup: node_id -> node_name
    id_to_name: dict[str, str] = {}
    for n in nodes:
        nid = _node_id(n)
        name = _node_name(n) or nid
        id_to_name[nid] = name

    transition_risks: list[dict] = []

    for cf in content_flow:
        from_id = _cf_from(cf)
        to_raw = _cf_to(cf)
        consumer_ids = to_raw if isinstance(to_raw, list) else ([to_raw] if to_raw else [])

        from_name = id_to_name.get(from_id, from_id)
        from_class = classifications.get(from_name, {}).get("primary")

        for to_id in consumer_ids:
            to_name = id_to_name.get(to_id, to_id)
            to_class = classifications.get(to_name, {}).get("primary")

            if from_class and to_class:
                pair = (from_class, to_class)
                label = RISKY_TRANSITIONS.get(pair)
                if label:
                    transition_risks.append({
                        "from_node": from_name,
                        "to_node": to_name,
                        "from_category": from_class,
                        "to_category": to_class,
                        "risk_label": label,
                    })

    return transition_risks


# ---------------------------------------------------------------------------
# Findings generation
# ---------------------------------------------------------------------------

def generate_findings(
    risk_scores: dict, graph: dict, transition_risks: list[dict]
) -> list[str]:
    """Produce human-readable findings based on risk scores and graph data."""
    findings: list[str] = []

    if risk_scores.get("single_model_dependency", 0) >= 1.0:
        findings.append("All agents use the same LLM with no fallback")

    uir = risk_scores.get("unvalidated_input_rate", 0)
    if uir > 0.5:
        pct = round(uir * 100)
        findings.append(
            f"{pct}% of agent-to-agent data flows have no validation"
        )

    if risk_scores.get("model_degradation_resilience", 0) == 0.0:
        findings.append(
            "System continued execution under degraded LLM conditions "
            "without detection"
        )

    ccr = risk_scores.get("compound_chain_risk", 0)
    if ccr > 0.5:
        risk_indicators = graph.get("risk_indicators") or {}
        chain_depth = risk_indicators.get(
            "max_chain_depth"
        ) or risk_indicators.get("max_delegation_depth", 0)
        pct = round(ccr * 100)
        findings.append(
            f"Chain of {chain_depth} agents creates compound risk of {pct}%"
        )

    for tr in transition_risks:
        findings.append(
            f"Risky transition ({tr['risk_label']}): "
            f"{tr['from_node']} ({tr['from_category']}) -> "
            f"{tr['to_node']} ({tr['to_category']})"
        )

    return findings


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------

def analyse_graph(graph: dict) -> dict:
    """Run the full risk analysis pipeline on a graph JSON dict."""
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []

    topology = classify_topology(nodes, edges)
    agent_count = len(
        [n for n in nodes if _node_type(n) in ("agent", "orchestrator")]
    )

    risk_scores = compute_risk_scores(graph)
    classifications = classify_node_outputs(graph)
    transition_risks = analyse_transition_risks(graph, classifications)
    findings = generate_findings(risk_scores, graph, transition_risks)

    repo = graph.get("repo_id") or graph.get("repo", "")
    framework = graph.get("framework", "")
    tier = graph.get("tier")

    report: dict = {
        "repo": repo,
        "framework": framework,
    }
    if tier is not None:
        report["tier"] = tier
    report.update({
        "topology_type": topology,
        "agent_count": agent_count,
        "risk_scores": risk_scores,
        "output_classifications": classifications,
        "transition_risks": transition_risks,
        "findings": findings,
    })

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-repo risk analysis for AI agent behavioral graphs."
    )
    parser.add_argument(
        "graph_json",
        help="Path to a graph JSON file (from graph_builder.py).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Write risk report to this file instead of stdout.",
    )
    args = parser.parse_args()

    graph_path = Path(args.graph_json)
    if not graph_path.exists():
        print(f"Error: file not found: {graph_path}", file=sys.stderr)
        sys.exit(1)

    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    report = analyse_graph(graph)

    output_text = json.dumps(report, indent=2) + "\n"

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(
            f"Risk report written to {out_path} "
            f"(overall risk: {report['risk_scores']['overall_risk']})",
            file=sys.stderr,
        )
    else:
        sys.stdout.write(output_text)


if __name__ == "__main__":
    main()
