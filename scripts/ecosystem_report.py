#!/usr/bin/env python3
"""Aggregate findings generator -- produces the final ecosystem report.

Reads all risk reports, graph files, and scan results across the results
directory and produces a structured ecosystem analysis covering topology
census, framework comparison, resilience findings, output flow patterns,
role semantics, and cross-referenced risk profiles.

Usage:
    python ecosystem_report.py <results_dir> [-o report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow running from repo root without installing stratum_lab
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from output_classifier import classify_output

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUCCESS_STATUSES = {"SUCCESS", "PARTIAL_SUCCESS", "TIER2_SUCCESS", "TIER2_PARTIAL"}

# Minimum node/edge thresholds for computing network metrics
MIN_NODES_FOR_NETWORK_METRICS = 5
MIN_SAMPLES_FOR_PER_AGENT_DIST = 10

# Topology types from graph_builder.classify_topology
KNOWN_TOPOLOGY_TYPES = {
    "single_agent",
    "hierarchical_delegation",
    "hub_and_spoke",
}


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | None:
    """Load a JSON file, returning None on failure."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _collect_repo_data(results_dir: Path) -> list[dict]:
    """Walk results_dir and load per-repo artifacts into a flat list."""
    repos: list[dict] = []

    for repo_dir in sorted(results_dir.iterdir()):
        if not repo_dir.is_dir():
            continue

        status_path = repo_dir / "status.json"
        if not status_path.exists():
            continue

        status_data = _load_json(status_path)
        if status_data is None:
            continue

        status = status_data.get("status", "UNKNOWN")
        tier = status_data.get("tier", 1)

        entry: dict[str, Any] = {
            "dir": str(repo_dir),
            "dir_name": repo_dir.name,
            "status": status,
            "tier": tier,
            "repo_url": status_data.get("repo", ""),
            "framework": status_data.get("framework", ""),
            "event_count": status_data.get("event_count", 0),
            "duration_seconds": status_data.get("duration_seconds", 0),
            "graph": None,
            "risk_report": None,
        }

        if status in SUCCESS_STATUSES:
            graph = _load_json(repo_dir / "graph.json")
            if graph is not None:
                entry["graph"] = graph
                # Infer framework from graph if status didn't have it
                if not entry["framework"]:
                    entry["framework"] = graph.get("framework", "")

            risk = _load_json(repo_dir / "risk_report.json")
            if risk is not None:
                entry["risk_report"] = risk

        repos.append(entry)

    return repos


# ---------------------------------------------------------------------------
# Topology census
# ---------------------------------------------------------------------------

def _topology_census(repos: list[dict]) -> dict:
    """Classify all topology types and count prevalence, separated by tier."""
    tier1_types: Counter[str] = Counter()
    tier2_types: Counter[str] = Counter()
    all_types: Counter[str] = Counter()

    for repo in repos:
        graph = repo.get("graph")
        if graph is None:
            continue

        topo = graph.get("topology_type", "unknown")
        all_types[topo] += 1
        if repo["tier"] == 1:
            tier1_types[topo] += 1
        else:
            tier2_types[topo] += 1

    total = sum(all_types.values()) or 1
    prevalence = {
        t: round(c / total * 100, 1) for t, c in all_types.most_common()
    }

    return {
        "total_graphs_analyzed": sum(all_types.values()),
        "topology_distribution": dict(all_types.most_common()),
        "topology_prevalence_pct": prevalence,
        "tier1_distribution": dict(tier1_types.most_common()),
        "tier2_distribution": dict(tier2_types.most_common()),
    }


# ---------------------------------------------------------------------------
# Framework comparison
# ---------------------------------------------------------------------------

def _framework_comparison(repos: list[dict]) -> dict:
    """Compare frameworks on topology complexity, event richness, error handling."""
    fw_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "count": 0,
        "topologies": Counter(),
        "total_nodes": 0,
        "total_edges": 0,
        "total_events": 0,
        "has_error_handling": 0,
        "tier1_count": 0,
        "tier2_count": 0,
    })

    for repo in repos:
        graph = repo.get("graph")
        if graph is None:
            continue

        fw = repo.get("framework") or "unknown"
        stats = fw_stats[fw]
        stats["count"] += 1

        if repo["tier"] == 1:
            stats["tier1_count"] += 1
        else:
            stats["tier2_count"] += 1

        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        node_list = list(nodes.values()) if isinstance(nodes, dict) else nodes
        edge_list = list(edges.values()) if isinstance(edges, dict) else edges

        stats["total_nodes"] += len(node_list)
        stats["total_edges"] += len(edge_list)
        stats["total_events"] += graph.get("summary", {}).get("total_events", 0)

        topo = graph.get("topology_type", "unknown")
        stats["topologies"][topo] += 1

        risk = graph.get("risk_indicators", {})
        if risk.get("has_error_boundaries"):
            stats["has_error_handling"] += 1

    comparison = {}
    for fw, stats in sorted(fw_stats.items(), key=lambda x: x[1]["count"], reverse=True):
        n = stats["count"] or 1
        comparison[fw] = {
            "repo_count": stats["count"],
            "tier1_count": stats["tier1_count"],
            "tier2_count": stats["tier2_count"],
            "avg_nodes": round(stats["total_nodes"] / n, 1),
            "avg_edges": round(stats["total_edges"] / n, 1),
            "avg_events": round(stats["total_events"] / n, 1),
            "error_handling_pct": round(stats["has_error_handling"] / n * 100, 1),
            "topology_distribution": dict(stats["topologies"].most_common()),
        }

    return comparison


# ---------------------------------------------------------------------------
# Resilience findings
# ---------------------------------------------------------------------------

def _resilience_findings(repos: list[dict]) -> dict:
    """Analyze model degradation resilience from risk indicators."""
    total_with_indicators = 0
    degradation_detected_count = 0
    continued_without_detection = 0
    all_remapped_count = 0

    for repo in repos:
        graph = repo.get("graph")
        if graph is None:
            continue

        risk = graph.get("risk_indicators")
        if risk is None:
            continue

        total_with_indicators += 1

        if risk.get("model_degradation_detected"):
            degradation_detected_count += 1

        if risk.get("all_calls_model_remapped") and risk.get("no_agent_detected_degradation"):
            continued_without_detection += 1

        if risk.get("all_calls_model_remapped"):
            all_remapped_count += 1

    denom = total_with_indicators or 1
    continued_pct = round(continued_without_detection / denom * 100, 1)

    return {
        "total_repos_analyzed": total_with_indicators,
        "all_calls_remapped_count": all_remapped_count,
        "continued_without_detection_count": continued_without_detection,
        "continued_without_detection_pct": continued_pct,
        "summary": (
            f"Under degraded LLM conditions (smaller model substituted for "
            f"expected model), {continued_pct}% of open-source agent "
            f"implementations continued execution without detecting quality loss."
        ),
        "framing_note": (
            "This reflects a resilience test using Mistral-7B-Instruct-v0.3 as "
            "a substitute, not a behavioral analysis of the agents' intended "
            "capabilities."
        ),
    }


# ---------------------------------------------------------------------------
# Output flow analysis
# ---------------------------------------------------------------------------

def _output_flow_analysis(repos: list[dict]) -> dict:
    """Analyze transformation types and pass-through prevalence per framework."""
    fw_transforms: dict[str, Counter] = defaultdict(Counter)
    total_flows = 0
    pass_through_count = 0
    output_classifications: Counter[str] = Counter()

    for repo in repos:
        graph = repo.get("graph")
        if graph is None:
            continue

        fw = repo.get("framework") or "unknown"
        content_flow = graph.get("content_flow", [])

        for flow in content_flow:
            total_flows += 1
            ttype = flow.get("transformation_type", "unknown")
            fw_transforms[fw][ttype] += 1
            if ttype == "pass_through":
                pass_through_count += 1

        # Classify outputs using the output_classifier
        nodes = graph.get("nodes", [])
        node_list = list(nodes.values()) if isinstance(nodes, dict) else nodes
        for node in node_list:
            if not isinstance(node, dict):
                continue
            metadata = node.get("metadata", {})
            preview = metadata.get("output_preview")
            output_type = metadata.get("output_type", "str")
            size_bytes = metadata.get("output_size_bytes")
            if preview is not None and size_bytes is not None:
                classification = classify_output(preview, output_type, size_bytes)
                output_classifications[classification["primary"]] += 1

    denom = total_flows or 1
    pass_through_pct = round(pass_through_count / denom * 100, 1)

    per_framework = {}
    for fw, transforms in sorted(fw_transforms.items(), key=lambda x: sum(x[1].values()), reverse=True):
        per_framework[fw] = {
            "total_flows": sum(transforms.values()),
            "transformation_distribution": dict(transforms.most_common()),
        }

    return {
        "total_content_flows": total_flows,
        "pass_through_count": pass_through_count,
        "pass_through_pct": pass_through_pct,
        "pass_through_note": (
            f"{pass_through_pct}% of observed content flows show pass-through "
            f"(zero validation) edges across open-source agent implementations."
        ),
        "per_framework": per_framework,
        "output_classifications": dict(output_classifications.most_common()),
    }


# ---------------------------------------------------------------------------
# Role semantics analysis
# ---------------------------------------------------------------------------

def _role_semantics(repos: list[dict]) -> dict:
    """Cluster repos by agent role names and analyze similarity."""
    role_repos: dict[str, list[str]] = defaultdict(list)
    role_topologies: dict[str, Counter] = defaultdict(Counter)
    role_event_counts: dict[str, list[int]] = defaultdict(list)

    for repo in repos:
        graph = repo.get("graph")
        if graph is None:
            continue

        nodes = graph.get("nodes", [])
        node_list = list(nodes.values()) if isinstance(nodes, dict) else nodes
        topo = graph.get("topology_type", "unknown")
        events_total = graph.get("summary", {}).get("total_events", 0)

        for node in node_list:
            if not isinstance(node, dict):
                continue
            role = (node.get("metadata") or {}).get("role")
            if not role:
                continue

            role_normalized = role.strip().lower()
            role_repos[role_normalized].append(repo.get("repo_url", repo["dir_name"]))
            role_topologies[role_normalized][topo] += 1
            role_event_counts[role_normalized].append(events_total)

    # Build role clusters
    clusters: dict[str, dict] = {}
    for role, repo_list in sorted(role_repos.items(), key=lambda x: len(x[1]), reverse=True):
        n = len(repo_list)
        entry: dict[str, Any] = {
            "occurrence_count": n,
            "repos": repo_list[:20],  # cap for readability
            "topology_distribution": dict(role_topologies[role].most_common()),
        }

        # Only compute per-agent distributions when 10+ samples exist
        if n >= MIN_SAMPLES_FOR_PER_AGENT_DIST:
            events = role_event_counts[role]
            entry["avg_events_in_repo"] = round(sum(events) / n, 1)
            entry["distribution_available"] = True
        else:
            entry["distribution_available"] = False
            entry["distribution_note"] = (
                f"Only {n} sample(s) for role '{role}'; per-agent distribution "
                f"analysis requires {MIN_SAMPLES_FOR_PER_AGENT_DIST}+ samples."
            )

        clusters[role] = entry

    return {
        "unique_roles": len(clusters),
        "total_role_assignments": sum(len(v) for v in role_repos.values()),
        "clusters": clusters,
    }


# ---------------------------------------------------------------------------
# Cross-reference topology with risk
# ---------------------------------------------------------------------------

def _cross_reference_topology_risk(repos: list[dict]) -> dict:
    """Compare risk profiles of complex vs simple topologies."""
    simple: list[dict] = []
    complex_topos: list[dict] = []

    for repo in repos:
        graph = repo.get("graph")
        if graph is None:
            continue

        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        node_list = list(nodes.values()) if isinstance(nodes, dict) else nodes
        edge_list = list(edges.values()) if isinstance(edges, dict) else edges

        agent_count = sum(
            1 for n in node_list
            if isinstance(n, dict) and n.get("type") in ("agent", "orchestrator")
        )
        has_branching = any(
            isinstance(e, dict) and e.get("type") in ("delegates_to", "orchestrates")
            for e in edge_list
        )

        risk = graph.get("risk_indicators", {})
        risk_report = repo.get("risk_report")
        risk_score = 0.0
        if risk_report and isinstance(risk_report, dict):
            es = risk_report.get("executive_summary", {})
            risk_score = es.get("overall_risk_score", 0.0)

        entry = {
            "agent_count": agent_count,
            "has_branching": has_branching,
            "topology_type": graph.get("topology_type", "unknown"),
            "risk_score": risk_score,
            "has_error_handling": risk.get("has_error_boundaries", False),
            "max_chain_depth": risk.get("max_chain_depth", 0),
            "pass_through": risk.get("unvalidated_pass_through", False),
        }

        if agent_count > 3 and has_branching:
            complex_topos.append(entry)
        else:
            simple.append(entry)

    def _summarize(group: list[dict], label: str) -> dict:
        n = len(group)
        if n == 0:
            return {"count": 0, "label": label}
        scores = [g["risk_score"] for g in group]
        error_handling = sum(1 for g in group if g["has_error_handling"])
        pass_through = sum(1 for g in group if g["pass_through"])
        return {
            "count": n,
            "label": label,
            "avg_risk_score": round(sum(scores) / n, 1) if scores else 0,
            "error_handling_pct": round(error_handling / n * 100, 1),
            "pass_through_pct": round(pass_through / n * 100, 1),
            "avg_chain_depth": round(
                sum(g["max_chain_depth"] for g in group) / n, 1
            ),
        }

    return {
        "simple_topologies": _summarize(simple, "simple (<=3 agents or no branching)"),
        "complex_topologies": _summarize(complex_topos, "complex (>3 agents with branching)"),
        "note": (
            "Directional comparison only. Sample sizes and single-run data "
            "limit the precision of these risk profile differences."
        ),
    }


# ---------------------------------------------------------------------------
# Key findings generator
# ---------------------------------------------------------------------------

def _generate_findings(
    repos: list[dict],
    topology_census: dict,
    framework_comparison: dict,
    resilience: dict,
    output_flow: dict,
    role_semantics: dict,
    cross_ref: dict,
) -> list[dict]:
    """Generate key findings from aggregated data with proper framing."""
    findings: list[dict] = []
    tier1_repos = [r for r in repos if r["tier"] == 1 and r.get("graph")]
    tier2_repos = [r for r in repos if r["tier"] != 1 and r.get("graph")]
    total_graphs = len(tier1_repos) + len(tier2_repos)

    # 1. Topology dominance finding
    topo_dist = topology_census.get("topology_distribution", {})
    if topo_dist:
        dominant_type = max(topo_dist, key=topo_dist.get)
        dominant_pct = topology_census["topology_prevalence_pct"].get(dominant_type, 0)
        findings.append({
            "finding": (
                f"The most prevalent topology among open-source agent "
                f"implementations is '{dominant_type}', observed in "
                f"{dominant_pct}% of analyzed repositories."
            ),
            "evidence": f"{topo_dist[dominant_type]} of {total_graphs} graphs classified as {dominant_type}.",
            "implication": (
                "Topology prevalence suggests directional patterns in how "
                "agent developers structure multi-agent systems."
            ),
            "tier1_evidence": f"Tier 1: {topology_census['tier1_distribution'].get(dominant_type, 0)} repos",
            "tier2_evidence": f"Tier 2: {topology_census['tier2_distribution'].get(dominant_type, 0)} repos",
        })

    # 2. Resilience finding
    continued_pct = resilience.get("continued_without_detection_pct", 0)
    findings.append({
        "finding": resilience.get("summary", ""),
        "evidence": (
            f"{resilience.get('continued_without_detection_count', 0)} of "
            f"{resilience.get('total_repos_analyzed', 0)} repositories "
            f"continued execution under model substitution."
        ),
        "implication": (
            "Open-source agent implementations may lack runtime checks for "
            "LLM response quality, creating a resilience gap."
        ),
        "tier1_evidence": f"Tier 1 repos included in resilience analysis: {len(tier1_repos)}",
        "tier2_evidence": f"Tier 2 repos included in resilience analysis: {len(tier2_repos)}",
    })

    # 3. Pass-through finding
    pt_pct = output_flow.get("pass_through_pct", 0)
    if output_flow.get("total_content_flows", 0) > 0:
        findings.append({
            "finding": (
                f"Approximately {pt_pct}% of content flows between agents show "
                f"pass-through behavior with no apparent validation or "
                f"transformation in open-source agent implementations."
            ),
            "evidence": (
                f"{output_flow['pass_through_count']} of "
                f"{output_flow['total_content_flows']} content flows classified "
                f"as pass-through."
            ),
            "implication": (
                "Pass-through edges may propagate errors or low-quality outputs "
                "without interception."
            ),
            "tier1_evidence": "See per-framework breakdown for tier separation.",
            "tier2_evidence": "Tier 2 traces reflect framework default behavior.",
        })

    # 4. Error handling finding
    fw_with_low_error_handling = [
        fw for fw, stats in framework_comparison.items()
        if stats.get("error_handling_pct", 0) < 30 and stats.get("repo_count", 0) >= 3
    ]
    if fw_with_low_error_handling:
        findings.append({
            "finding": (
                f"Error handling presence is directionally low in several "
                f"frameworks: {', '.join(fw_with_low_error_handling)}."
            ),
            "evidence": (
                "Based on has_error_boundaries indicator from runtime graphs."
            ),
            "implication": (
                "Agent repositories without error boundary signals may silently "
                "propagate failures through delegation chains."
            ),
            "tier1_evidence": "Framework comparison reflects Tier 1 and Tier 2 separately.",
            "tier2_evidence": "Tier 2 may undercount error handling due to limited execution depth.",
        })

    # 5. Complex vs simple topology risk
    simple_summary = cross_ref.get("simple_topologies", {})
    complex_summary = cross_ref.get("complex_topologies", {})
    if complex_summary.get("count", 0) > 0 and simple_summary.get("count", 0) > 0:
        findings.append({
            "finding": (
                f"Complex topologies (>3 agents, branching) show a directionally "
                f"different risk profile compared to simple sequential chains."
            ),
            "evidence": (
                f"Complex avg risk score: {complex_summary.get('avg_risk_score', 'N/A')}, "
                f"Simple avg risk score: {simple_summary.get('avg_risk_score', 'N/A')}."
            ),
            "implication": (
                "Topology complexity is a directional indicator of risk, though "
                "single-run data limits precise quantification."
            ),
            "tier1_evidence": "Cross-reference analysis includes both tiers; see tier breakdown in topology census.",
            "tier2_evidence": "Tier 2 traces may inflate simple topology counts.",
        })

    return findings


# ---------------------------------------------------------------------------
# Methodology block
# ---------------------------------------------------------------------------

def _build_methodology(repos: list[dict]) -> dict:
    """Build the methodology section with proper framing."""
    total = len(repos)
    tier1 = sum(1 for r in repos if r["tier"] == 1 and r.get("graph"))
    tier2 = sum(1 for r in repos if r["tier"] != 1 and r.get("graph"))
    successful = sum(1 for r in repos if r["status"] in SUCCESS_STATUSES)

    return {
        "scan_description": (
            f"Executed {total} open-source AI agent repositories under "
            f"controlled conditions. {successful} produced usable runtime traces."
        ),
        "degradation_test": (
            "All LLM calls routed through Mistral-7B-Instruct-v0.3 as a "
            "resilience test to observe agent behavior under degraded model "
            "conditions."
        ),
        "tier_explanation": (
            f"Tier 1 traces (N={tier1}) reflect authentic repo execution. "
            f"Tier 2 traces (N={tier2}) reflect framework default behavior "
            f"where original entry points could not be resolved."
        ),
        "limitations": [
            "Sample is open-source repositories, not production enterprise systems",
            "Single-run data limits per-agent distribution analysis",
            "Mistral-7B responses differ from GPT-4, affecting agent behavior",
            "Findings are directional, not precise statistical estimates",
        ],
    }


# ---------------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------------

def generate_ecosystem_report(results_dir: Path) -> dict:
    """Build the complete ecosystem report from per-repo results."""
    repos = _collect_repo_data(results_dir)

    if not repos:
        return {"error": "No repository data found in results directory."}

    # Load optional aggregate scan report for supplemental data
    scan_report_path = results_dir / "scan_report.json"
    scan_report = _load_json(scan_report_path)

    # Run analyses
    topology = _topology_census(repos)
    fw_comparison = _framework_comparison(repos)
    resilience = _resilience_findings(repos)
    output_flow = _output_flow_analysis(repos)
    roles = _role_semantics(repos)
    cross_ref = _cross_reference_topology_risk(repos)

    # Generate findings
    findings = _generate_findings(
        repos, topology, fw_comparison, resilience, output_flow, roles, cross_ref
    )

    total_with_graphs = sum(1 for r in repos if r.get("graph") is not None)

    report = {
        "title": (
            f"Stratum Ecosystem Analysis: Runtime Behavior of "
            f"{total_with_graphs} AI Agent Systems"
        ),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "methodology": _build_methodology(repos),
        "key_findings": findings,
        "topology_census": topology,
        "framework_comparison": fw_comparison,
        "resilience_analysis": resilience,
        "output_flow_analysis": output_flow,
        "role_semantics": roles,
        "topology_risk_crossref": cross_ref,
    }

    # Attach scan report summary if available
    if scan_report is not None:
        report["scan_report_summary"] = {
            "total_repos_attempted": scan_report.get("overview", {}).get("total_repos_attempted"),
            "success_rate_pct": scan_report.get("overview", {}).get("success_rate_percent"),
            "total_events": scan_report.get("overview", {}).get("total_events"),
        }

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate findings generator. Reads all risk reports and scan "
            "results, produces the final ecosystem report."
        ),
    )
    parser.add_argument(
        "results_dir",
        help="Path to the results directory containing per-repo subdirectories.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output file path (default: <results_dir>/ecosystem_report.json).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    report = generate_ecosystem_report(results_dir)

    output_path = Path(args.output) if args.output else results_dir / "ecosystem_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print summary to stderr
    n_findings = len(report.get("key_findings", []))
    topo = report.get("topology_census", {})
    fw = report.get("framework_comparison", {})

    print(f"\n{'=' * 60}", file=sys.stderr)
    print("ECOSYSTEM REPORT GENERATED", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    print(f"Title:         {report.get('title', 'N/A')}", file=sys.stderr)
    print(f"Key findings:  {n_findings}", file=sys.stderr)
    print(f"Topologies:    {topo.get('total_graphs_analyzed', 0)} graphs analyzed", file=sys.stderr)
    print(f"Frameworks:    {len(fw)} frameworks compared", file=sys.stderr)
    print(f"Roles:         {report.get('role_semantics', {}).get('unique_roles', 0)} unique roles", file=sys.stderr)
    print(f"\nReport saved to: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
