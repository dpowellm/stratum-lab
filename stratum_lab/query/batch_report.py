"""Batch report generation for all repos in the dataset.

Runs each repo's structural graph through the query layer to produce
a per-repo risk report. These reports serve two purposes:
1. Validate the query layer produces sensible output at scale
2. Generate the outreach material (teaser + full reports)
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def generate_batch_reports(
    enriched_graphs_dir: Path,
    knowledge_base_dir: Path,
    output_dir: Path,
) -> Dict:
    """Generate risk reports for every enriched graph in the dataset.

    Returns summary statistics about report generation.
    """
    enriched_graphs_dir = Path(enriched_graphs_dir)
    knowledge_base_dir = Path(knowledge_base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports: List[Dict] = []
    errors: List[Dict] = []

    graph_files = sorted(enriched_graphs_dir.glob("*.json"))

    for graph_file in graph_files:
        repo_id = graph_file.stem.replace("_enriched", "")
        try:
            with open(graph_file) as f:
                graph = json.load(f)

            # Run query pipeline on this repo's graph
            from stratum_lab.query.fingerprint import compute_graph_fingerprint
            from stratum_lab.query.matcher import match_against_dataset
            from stratum_lab.query.predictor import predict_risks
            from stratum_lab.query.report import generate_risk_report

            fingerprint = compute_graph_fingerprint(graph)

            matches = match_against_dataset(fingerprint, str(knowledge_base_dir), top_k=5)

            preconditions = list(graph.get("taxonomy_preconditions", []))
            prediction = predict_risks(graph, matches, preconditions, knowledge_base_dir)
            report = generate_risk_report(prediction, graph, output_format="json")

            report_path = output_dir / f"{repo_id}_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

            exec_summary = report.get("executive_summary", {})
            top_risks = exec_summary.get("top_risks", [])

            reports.append({
                "repo_id": repo_id,
                "risk_score": exec_summary.get("overall_risk_score", 0),
                "risk_level": exec_summary.get("risk_level", "UNKNOWN"),
                "risk_count": len(report.get("risk_details", [])),
                "archetype": exec_summary.get("archetype", "unknown"),
                "top_risk": top_risks[0]["name"] if top_risks else None,
            })
        except Exception as e:
            errors.append({"repo_id": repo_id, "error": str(e)})

    summary = {
        "total_repos": len(graph_files),
        "reports_generated": len(reports),
        "errors": len(errors),
        "error_details": errors,
        "risk_distribution": _compute_risk_distribution(reports),
        "reports": reports,
    }

    with open(output_dir / "batch_report_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def _compute_risk_distribution(reports: List[Dict]) -> Dict:
    """Aggregate risk statistics across all reports."""
    levels = Counter(r["risk_level"] for r in reports)
    scores = [r["risk_score"] for r in reports]
    archetypes = Counter(r["archetype"] for r in reports)
    top_risks = Counter(r["top_risk"] for r in reports if r["top_risk"])

    return {
        "by_risk_level": dict(levels),
        "score_mean": sum(scores) / max(len(scores), 1),
        "score_p50": sorted(scores)[len(scores) // 2] if scores else 0,
        "score_p90": sorted(scores)[int(len(scores) * 0.9)] if scores else 0,
        "by_archetype": dict(archetypes),
        "most_common_top_risk": top_risks.most_common(3),
    }
