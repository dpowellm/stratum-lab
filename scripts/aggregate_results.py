#!/usr/bin/env python3
"""Post-scan aggregator -- produces scan_report.json from per-repo results.

Reads all results/<repo_hash>/ directories and produces a comprehensive
scan report including overview, status/tier/framework breakdowns, event
statistics, graph quality metrics, topology patterns, failure reasons,
and a list of successful repos.

Usage:
    python aggregate_results.py <output_dir>
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Statuses that count as "successful" (produced usable events)
SUCCESS_STATUSES = {"SUCCESS", "PARTIAL_SUCCESS", "TIER2_SUCCESS", "TIER2_PARTIAL"}


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, skipping malformed lines."""
    events: list[dict] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return events


def _load_json(path: Path) -> dict | None:
    """Load a JSON file, returning None on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _classify_failure(status: str, error_tail: str) -> str:
    """Classify a failure into a human-readable reason."""
    if status == "CLONE_FAILED":
        return "clone_failed"
    if status == "NO_ENTRY_POINT":
        return "no_entry_point"
    if status == "UNRESOLVABLE_IMPORT":
        return "missing_module"
    if status == "SERVER_BASED":
        return "server_based_repo"
    if status == "TIMEOUT_NO_EVENTS":
        return "timeout_no_events"
    if status == "NO_EVENTS":
        return "no_events_captured"
    if status == "RUNTIME_ERROR":
        low = error_tail.lower()
        if "api" in low or "connection" in low:
            return "api_connection_error"
        if "permission" in low:
            return "permission_error"
        if "syntax" in low:
            return "syntax_error"
        return "runtime_error"
    return status.lower()


def _grade_graph(graph: dict) -> str:
    """Grade a graph as RICH, BASIC, or EMPTY."""
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    n_nodes = len(nodes)
    n_edges = len(edges)
    if n_nodes >= 3 and n_edges >= 2:
        return "RICH"
    if n_nodes >= 1:
        return "BASIC"
    return "EMPTY"


def aggregate(output_dir: Path) -> dict:
    """Aggregate all per-repo results into a scan report."""
    status_counts: Counter[str] = Counter()
    tier_counts: Counter[int] = Counter()
    framework_counts: Counter[str] = Counter()
    error_reasons: Counter[str] = Counter()
    event_type_counts: Counter[str] = Counter()
    successful_repos: list[dict] = []
    total_events = 0
    total_repos = 0

    # Graph quality accumulators
    graph_grade_counts: Counter[str] = Counter()
    graph_node_counts: list[int] = []
    graph_edge_counts: list[int] = []

    # Topology accumulators
    node_type_counts: Counter[str] = Counter()
    edge_type_counts: Counter[str] = Counter()
    unique_nodes_per_repo: list[int] = []

    for repo_dir in sorted(output_dir.iterdir()):
        if not repo_dir.is_dir():
            continue

        status_file = repo_dir / "status.json"
        if not status_file.exists():
            continue

        total_repos += 1
        status_data = _load_json(status_file)
        if status_data is None:
            status_counts["PARSE_ERROR"] += 1
            continue

        status = status_data.get("status", "UNKNOWN")
        tier = status_data.get("tier", 1)
        event_count = status_data.get("event_count", 0)
        repo_url = status_data.get("repo", "")
        entry_point = status_data.get("entry_point", "")
        duration = status_data.get("duration_seconds", 0)

        status_counts[status] += 1
        tier_counts[tier] += 1

        if status in SUCCESS_STATUSES:
            # -- Parse events for framework and event type stats --
            events_file = repo_dir / "stratum_events.jsonl"
            events = _load_jsonl(events_file) if events_file.exists() else []
            repo_event_count = len(events) if events else event_count

            for event in events:
                et = event.get("event_type", "unknown")
                event_type_counts[et] += 1
                fw = event.get("framework", "")
                if fw and fw != "unknown":
                    framework_counts[fw] += 1

            total_events += repo_event_count

            successful_repos.append({
                "repo": repo_url,
                "status": status,
                "tier": tier,
                "event_count": repo_event_count,
                "entry_point": entry_point,
                "duration_seconds": duration,
            })

            # -- Graph quality and topology --
            graph_file = repo_dir / "graph.json"
            if graph_file.exists():
                graph = _load_json(graph_file)
                if graph is not None:
                    nodes = graph.get("nodes", [])
                    edges = graph.get("edges", [])
                    # graph_builder outputs lists, not dicts
                    node_list = list(nodes.values()) if isinstance(nodes, dict) else nodes
                    edge_list = list(edges.values()) if isinstance(edges, dict) else edges
                    n_nodes = len(node_list)
                    n_edges = len(edge_list)

                    graph_node_counts.append(n_nodes)
                    graph_edge_counts.append(n_edges)
                    graph_grade_counts[_grade_graph(graph)] += 1
                    unique_nodes_per_repo.append(n_nodes)

                    # Topology: node types
                    for nd in node_list:
                        if isinstance(nd, dict):
                            node_type_counts[nd.get("node_type", "unknown")] += 1
                    # Topology: edge types
                    for ed in edge_list:
                        if isinstance(ed, dict):
                            edge_type_counts[ed.get("edge_type", "unknown")] += 1
        else:
            error_tail = status_data.get("error_log_tail", "")
            reason = _classify_failure(status, error_tail)
            error_reasons[reason] += 1

    # -- Build report --
    num_success = len(successful_repos)
    success_rate = (num_success / total_repos * 100) if total_repos > 0 else 0.0
    repos_with_graphs = len(graph_node_counts)

    report = {
        "overview": {
            "total_repos_attempted": total_repos,
            "total_successful": num_success,
            "success_rate_percent": round(success_rate, 1),
            "scan_timestamp": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "total_events": total_events,
        },
        "status_breakdown": dict(status_counts.most_common()),
        "tier_breakdown": {
            f"tier_{k}": v for k, v in sorted(tier_counts.items())
        },
        "framework_breakdown": dict(framework_counts.most_common()),
        "event_statistics": {
            "total_events": total_events,
            "avg_events_per_success": (
                round(total_events / num_success, 1) if num_success > 0 else 0
            ),
            "event_type_distribution": dict(event_type_counts.most_common(20)),
        },
        "graph_quality": {
            "repos_with_graphs": repos_with_graphs,
            "avg_nodes_per_graph": (
                round(sum(graph_node_counts) / repos_with_graphs, 1)
                if repos_with_graphs > 0
                else 0
            ),
            "avg_edges_per_graph": (
                round(sum(graph_edge_counts) / repos_with_graphs, 1)
                if repos_with_graphs > 0
                else 0
            ),
            "grade_distribution": dict(graph_grade_counts.most_common()),
        },
        "topology_patterns": {
            "common_node_types": dict(node_type_counts.most_common()),
            "common_edge_types": dict(edge_type_counts.most_common()),
            "avg_unique_nodes_per_repo": (
                round(
                    sum(unique_nodes_per_repo) / len(unique_nodes_per_repo), 1
                )
                if unique_nodes_per_repo
                else 0
            ),
        },
        "common_failure_reasons": dict(error_reasons.most_common(10)),
        "successful_repos": sorted(
            successful_repos,
            key=lambda x: x["event_count"],
            reverse=True,
        ),
    }

    return report


def _print_summary(report: dict) -> None:
    """Print a human-readable summary to stderr."""
    ov = report["overview"]
    es = report["event_statistics"]
    gq = report["graph_quality"]

    print(f"\n{'=' * 60}", file=sys.stderr)
    print("SCAN REPORT", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    print(f"Total repos:     {ov['total_repos_attempted']}", file=sys.stderr)
    print(
        f"Successful:      {ov['total_successful']} "
        f"({ov['success_rate_percent']}%)",
        file=sys.stderr,
    )
    print(f"Total events:    {es['total_events']}", file=sys.stderr)
    print(f"Avg events/repo: {es['avg_events_per_success']}", file=sys.stderr)
    print(f"Graphs built:    {gq['repos_with_graphs']}", file=sys.stderr)
    print(file=sys.stderr)

    total = ov["total_repos_attempted"] or 1
    print("Status breakdown:", file=sys.stderr)
    for status, count in sorted(
        report["status_breakdown"].items(), key=lambda x: x[1], reverse=True
    ):
        pct = count / total * 100
        bar = "#" * min(int(pct / 2), 30)
        print(f"  {status:<25s} {count:>5d} ({pct:5.1f}%) {bar}", file=sys.stderr)

    if report["framework_breakdown"]:
        print("\nFramework breakdown:", file=sys.stderr)
        for fw, count in list(
            sorted(
                report["framework_breakdown"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )[:10]:
            print(f"  {fw:<20s} {count:>5d} events", file=sys.stderr)

    if gq["grade_distribution"]:
        print("\nGraph quality grades:", file=sys.stderr)
        for grade, count in sorted(
            gq["grade_distribution"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {grade:<10s} {count:>5d}", file=sys.stderr)

    if report["common_failure_reasons"]:
        print("\nTop failure reasons:", file=sys.stderr)
        for reason, count in sorted(
            report["common_failure_reasons"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            print(f"  {reason:<30s} {count:>5d}", file=sys.stderr)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: aggregate_results.py <output_dir>", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    if not output_dir.is_dir():
        print(f"Error: {output_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    report = aggregate(output_dir)

    # Write report
    report_path = output_dir / "scan_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary to stderr
    _print_summary(report)
    print(f"\nReport saved to: {report_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
