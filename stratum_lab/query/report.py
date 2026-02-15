"""Risk report generation.

Produces structured JSON or formatted markdown reports from risk predictions.
This is the customer-facing output of the product query layer.
"""

from __future__ import annotations

import json
from typing import Any


def generate_risk_report(
    prediction: Any,  # RiskPrediction dataclass
    structural_graph: dict[str, Any],
    output_format: str = "json",
) -> dict[str, Any] | str:
    """Generate the customer-facing risk report.

    Parameters
    ----------
    prediction:
        A RiskPrediction dataclass instance.
    structural_graph:
        The original structural graph dict.
    output_format:
        ``"json"`` for structured dict, ``"markdown"`` for human-readable.

    Returns
    -------
    A dict (json format) or string (markdown format).
    """
    report = _build_report_dict(prediction, structural_graph)

    if output_format == "markdown":
        return _render_markdown(report)
    return report


def _build_report_dict(
    prediction: Any,
    structural_graph: dict[str, Any],
) -> dict[str, Any]:
    """Build the structured report dict."""
    # Executive summary
    top_risks = sorted(
        prediction.predicted_risks,
        key=lambda r: r.manifestation_probability * _severity_weight(r.severity_when_manifested),
        reverse=True,
    )[:3]

    executive_summary = {
        "archetype": prediction.archetype,
        "archetype_prevalence": prediction.archetype_prevalence,
        "overall_risk_score": prediction.overall_risk_score,
        "risk_level": _score_to_level(prediction.overall_risk_score),
        "top_risks": [
            {
                "precondition_id": r.precondition_id,
                "name": r.precondition_name,
                "probability": r.manifestation_probability,
                "severity": r.severity_when_manifested,
            }
            for r in top_risks
        ],
        "positive_signals": prediction.positive_signals,
    }

    # Risk details
    risk_details = []
    for r in prediction.predicted_risks:
        risk_details.append({
            "precondition_id": r.precondition_id,
            "precondition_name": r.precondition_name,
            "manifestation_probability": r.manifestation_probability,
            "confidence_interval": list(r.confidence_interval),
            "sample_size": r.sample_size,
            "severity_when_manifested": r.severity_when_manifested,
            "behavioral_description": r.behavioral_description,
            "structural_evidence": r.structural_evidence,
            "similar_repo_outcomes": r.similar_repo_outcomes,
            "fragility_flag": r.fragility_flag,
            "remediation": r.remediation,
        })

    # Architecture analysis
    nodes = structural_graph.get("nodes", {})
    edges = structural_graph.get("edges", {})
    node_count = len(nodes)
    edge_count = len(edges)

    node_types: dict[str, int] = {}
    for nid, ndata in nodes.items():
        structural = ndata.get("structural", ndata)
        nt = structural.get("node_type", "unknown")
        node_types[nt] = node_types.get(nt, 0) + 1

    architecture_analysis = {
        "node_count": node_count,
        "edge_count": edge_count,
        "node_type_distribution": node_types,
        "fingerprint": prediction.graph_fingerprint,
    }

    # Benchmark comparison
    benchmark = {
        "archetype": prediction.archetype,
        "archetype_prevalence_pct": round(prediction.archetype_prevalence * 100, 1),
        "framework_comparison": prediction.framework_comparison,
    }

    # Methodology
    total_sample = sum(r.sample_size for r in prediction.predicted_risks) if prediction.predicted_risks else 0
    methodology = {
        "dataset_description": "Behavioral analysis of open-source AI agent repositories",
        "total_sample_across_risks": total_sample,
        "confidence_level": "95% Wilson score intervals",
        "manifestation_probability_definition": (
            "The fraction of repos with this structural precondition "
            "where the corresponding failure mode was observed at runtime."
        ),
    }

    # Semantic analysis
    semantic_analysis = getattr(prediction, "semantic_analysis", {}) or {}

    return {
        "executive_summary": executive_summary,
        "risk_details": risk_details,
        "structural_only_risks": prediction.structural_only_risks,
        "architecture_analysis": architecture_analysis,
        "semantic_analysis": semantic_analysis,
        "benchmark_comparison": benchmark,
        "dataset_coverage": prediction.dataset_coverage,
        "methodology": methodology,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    """Render the report dict as a formatted markdown string."""
    lines: list[str] = []

    # Title
    es = report["executive_summary"]
    lines.append("# Stratum Risk Report")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"**Architecture Type:** {es['archetype']}")
    lines.append(f"**Prevalence in Ecosystem:** {round(es['archetype_prevalence'] * 100, 1)}%")
    lines.append(f"**Overall Risk Score:** {es['overall_risk_score']:.1f} / 100 ({es['risk_level']})")
    lines.append("")

    if es["top_risks"]:
        lines.append("### Top Risks")
        lines.append("")
        for i, r in enumerate(es["top_risks"], 1):
            lines.append(
                f"{i}. **{r['name']}** -- "
                f"P(manifestation) = {r['probability']:.0%}, "
                f"severity: {r['severity']}"
            )
        lines.append("")

    if es["positive_signals"]:
        lines.append("### Positive Signals")
        lines.append("")
        for sig in es["positive_signals"]:
            lines.append(f"- {sig}")
        lines.append("")

    # Risk details
    lines.append("## Risk Details")
    lines.append("")
    for r in report["risk_details"]:
        lines.append(f"### {r['precondition_id']}: {r['precondition_name']}")
        lines.append("")
        lines.append(f"- **Manifestation Probability:** {r['manifestation_probability']:.0%} "
                      f"(95% CI: [{r['confidence_interval'][0]:.0%}, {r['confidence_interval'][1]:.0%}])")
        lines.append(f"- **Sample Size:** {r['sample_size']} repos")
        lines.append(f"- **Severity When Manifested:** {r['severity_when_manifested']}")
        if r.get("fragility_flag"):
            lines.append("- **Model Sensitivity:** This position is model-sensitive")
        lines.append("")
        lines.append(f"> {r['behavioral_description']}")
        lines.append("")
        if r.get("structural_evidence"):
            lines.append("**Structural Evidence:**")
            for ev in r["structural_evidence"]:
                lines.append(f"- `{ev}`")
            lines.append("")
        if r.get("remediation"):
            lines.append(f"**Remediation:** {r['remediation']}")
            lines.append("")

    # Architecture
    arch = report["architecture_analysis"]
    lines.append("## Architecture Analysis")
    lines.append("")
    lines.append(f"- **Nodes:** {arch['node_count']}")
    lines.append(f"- **Edges:** {arch['edge_count']}")
    for nt, count in arch.get("node_type_distribution", {}).items():
        lines.append(f"- {nt}: {count}")
    lines.append("")

    # Semantic analysis
    sem = report.get("semantic_analysis", {})
    if sem:
        lines.append("## Semantic Cascade Analysis")
        lines.append("")
        lines.append(f"- **Semantic Risk Score:** {sem.get('semantic_risk_score', 0):.1f} / 100")
        lines.append(f"- **Unvalidated Handoff Fraction:** {sem.get('unvalidated_handoff_fraction', 0):.0%}")
        lines.append(f"- **Semantic Chain Depth:** {sem.get('semantic_chain_depth', 0)}")
        lines.append(f"- **Max Blast Radius:** {sem.get('max_blast_radius', 0)}")
        lines.append(f"- **Classification Injection Points:** {sem.get('classification_injection_points', 0)}")
        nondet = sem.get("nondeterministic_nodes", [])
        if nondet:
            lines.append(f"- **Non-deterministic Nodes:** {', '.join(nondet)}")
        verdict = sem.get("verdict", "")
        if verdict:
            lines.append("")
            lines.append(f"> {verdict}")
        lines.append("")

    # Benchmark
    bench = report["benchmark_comparison"]
    lines.append("## Benchmark Comparison")
    lines.append("")
    lines.append(f"Architecture type **{bench['archetype']}** is found in "
                  f"**{bench['archetype_prevalence_pct']}%** of the ecosystem.")
    lines.append("")
    if bench.get("framework_comparison"):
        lines.append("**Framework-specific data:**")
        lines.append("")
        for motif, comparison in bench["framework_comparison"].items():
            if not isinstance(comparison, dict):
                continue
            per_fw = comparison.get("per_framework", {})
            if per_fw:
                lines.append(f"**{motif}:**")
                for fw_name, fw_data in per_fw.items():
                    rate = fw_data.get("failure_rate", 0)
                    count = fw_data.get("repos_count", 0)
                    lines.append(f"- {fw_name}: {rate:.0%} failure rate (n={count})")
                lines.append("")
            else:
                lines.append(f"- {motif}: no framework-specific data available")
                lines.append("")

    # Methodology
    meth = report["methodology"]
    lines.append("## Methodology")
    lines.append("")
    lines.append(f"- {meth['dataset_description']}")
    lines.append(f"- Total sample across all risks: {meth['total_sample_across_risks']}")
    lines.append(f"- Confidence intervals: {meth['confidence_level']}")
    lines.append(f"- {meth['manifestation_probability_definition']}")
    lines.append("")
    lines.append("---")
    lines.append("*Generated by Stratum Lab*")

    return "\n".join(lines)


def _severity_weight(severity: str) -> float:
    """Convert severity label to numeric weight."""
    return {"high": 3.0, "medium": 2.0, "low": 1.0, "none": 0.0}.get(severity, 1.0)


def _score_to_level(score: float) -> str:
    """Convert 0-100 risk score to a level label."""
    if score >= 70:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    elif score >= 15:
        return "LOW"
    return "MINIMAL"
