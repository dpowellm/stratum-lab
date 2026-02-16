#!/usr/bin/env python3
"""Post-scan aggregator -- reads ALL results/*/ directories and produces scan_report.json.

Separates Tier 1 and Tier 2 results across ALL statistics: execution results,
framework distributions, event counts, topology census, and successful repos.

Optionally reads behavioral_records/ directory to compute v6 quality metrics,
failure mode prevalence, and multi-run execution statistics.

Usage:
    python aggregate_results.py <results_dir> [-o scan_report.json] [--behavioral-records-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Statuses that produced usable events
# ---------------------------------------------------------------------------

SUCCESS_STATUSES = {"SUCCESS", "PARTIAL_SUCCESS", "TIER2_SUCCESS", "TIER2_PARTIAL"}

# Cost per hour: RunPod GPU + DigitalOcean droplet
COST_PER_HOUR = 0.38


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, skipping malformed lines."""
    events: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
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
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _read_tier(repo_dir: Path, status_data: dict) -> int:
    """Resolve tier from status.json or fallback to tier.txt."""
    tier = status_data.get("tier")
    if tier is not None:
        return int(tier)
    tier_file = repo_dir / "tier.txt"
    if tier_file.exists():
        try:
            return int(tier_file.read_text().strip())
        except (ValueError, OSError):
            pass
    return 1


# ---------------------------------------------------------------------------
# v6 behavioral record analysis
# ---------------------------------------------------------------------------

def _is_populated(value) -> bool:
    """Check if a section value is meaningfully populated (non-empty, non-null)."""
    if value is None:
        return False
    if isinstance(value, dict):
        return len(value) > 0
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        return len(value.strip()) > 0
    return bool(value)


def _compute_v6_quality(behavioral_records_dir: Path) -> dict:
    """Scan behavioral_records/ for v6 JSON files and compute quality metrics."""
    records_with_edge_validation = 0
    records_with_emergent_edges = 0
    records_with_error_propagation = 0
    records_with_failure_modes_manifested = 0
    records_with_monitoring_baselines = 0
    match_rates: list[float] = []
    total_records = 0

    for record_file in sorted(behavioral_records_dir.iterdir()):
        if not record_file.is_file() or record_file.suffix != ".json":
            continue
        record = _load_json(record_file)
        if record is None:
            continue
        total_records += 1

        # edge_validation: populated if dict has entries
        ev = record.get("edge_validation")
        if _is_populated(ev):
            records_with_edge_validation += 1

        # emergent_edges: populated if list is non-empty
        ee = record.get("emergent_edges")
        if _is_populated(ee):
            records_with_emergent_edges += 1

        # error_propagation: populated if list is non-empty
        ep = record.get("error_propagation")
        if _is_populated(ep):
            records_with_error_propagation += 1
            # Compute structural_prediction_match_rate for this record
            if isinstance(ep, list) and ep:
                matches = sum(
                    1 for trace in ep
                    if isinstance(trace, dict) and trace.get("structural_prediction_match")
                )
                match_rates.append(matches / len(ep))

        # failure_modes: count records where at least one mode has manifestation_observed=true
        fm = record.get("failure_modes")
        if isinstance(fm, list) and fm:
            has_manifested = any(
                isinstance(mode, dict) and mode.get("manifestation_observed") is True
                for mode in fm
            )
            if has_manifested:
                records_with_failure_modes_manifested += 1

        # monitoring_baselines: populated if list is non-empty
        mb = record.get("monitoring_baselines")
        if _is_populated(mb):
            records_with_monitoring_baselines += 1

    avg_match_rate = round(sum(match_rates) / len(match_rates), 2) if match_rates else 0.0

    return {
        "records_with_edge_validation": records_with_edge_validation,
        "records_with_emergent_edges": records_with_emergent_edges,
        "records_with_error_propagation": records_with_error_propagation,
        "records_with_failure_modes_manifested": records_with_failure_modes_manifested,
        "records_with_monitoring_baselines": records_with_monitoring_baselines,
        "avg_structural_prediction_match_rate": avg_match_rate,
    }


def _compute_failure_mode_prevalence(behavioral_records_dir: Path) -> dict:
    """Scan behavioral records for failure_modes with manifestation_observed=true.

    Returns dict mapping failure mode IDs to repos_affected count and manifestation_rate.
    """
    failure_mode_repos: Counter[str] = Counter()
    failure_mode_total: Counter[str] = Counter()
    total_records = 0

    for record_file in sorted(behavioral_records_dir.iterdir()):
        if not record_file.is_file() or record_file.suffix != ".json":
            continue
        record = _load_json(record_file)
        if record is None:
            continue
        total_records += 1

        fm = record.get("failure_modes")
        if not isinstance(fm, list):
            continue

        # Track which failure mode IDs are manifested in this record
        manifested_in_record: set[str] = set()
        seen_in_record: set[str] = set()
        for mode in fm:
            if not isinstance(mode, dict):
                continue
            mode_id = mode.get("failure_mode_id", mode.get("id", ""))
            if not mode_id:
                continue
            seen_in_record.add(mode_id)
            if mode.get("manifestation_observed") is True:
                manifested_in_record.add(mode_id)

        for mode_id in seen_in_record:
            failure_mode_total[mode_id] += 1
        for mode_id in manifested_in_record:
            failure_mode_repos[mode_id] += 1

    prevalence: dict[str, dict] = {}
    for mode_id in sorted(failure_mode_repos.keys()):
        repos_affected = failure_mode_repos[mode_id]
        total_seen = failure_mode_total[mode_id]
        manifestation_rate = round(repos_affected / total_seen, 2) if total_seen > 0 else 0.0
        prevalence[mode_id] = {
            "repos_affected": repos_affected,
            "manifestation_rate": manifestation_rate,
        }

    return prevalence


def _compute_multi_run_stats(behavioral_records_dir: Path) -> dict:
    """Compute multi-run execution statistics from behavioral records.

    Counts phase1 (run_number=1) and phase2 (run_number>1) info from
    execution_metadata and run_metadata in behavioral record files.
    """
    phase1_repos: set[str] = set()
    phase1_successful: set[str] = set()
    phase2_repos: set[str] = set()
    phase2_runs_total = 0
    behavioral_records_produced = 0

    for record_file in sorted(behavioral_records_dir.iterdir()):
        if not record_file.is_file() or record_file.suffix != ".json":
            continue
        record = _load_json(record_file)
        if record is None:
            continue

        behavioral_records_produced += 1
        repo_id = record.get("repo_full_name", record_file.stem)

        # Check execution_metadata for run_metadata entries
        exec_meta = record.get("execution_metadata", {})
        run_metadata = exec_meta.get("run_metadata", [])

        if isinstance(run_metadata, list) and run_metadata:
            for run in run_metadata:
                if not isinstance(run, dict):
                    continue
                run_number = run.get("run_number", 1)
                run_status = run.get("status", "")

                if run_number == 1:
                    phase1_repos.add(repo_id)
                    if run_status in SUCCESS_STATUSES:
                        phase1_successful.add(repo_id)
                elif run_number > 1:
                    phase2_repos.add(repo_id)
                    phase2_runs_total += 1
        else:
            # No run_metadata -- treat as a single phase1 run
            phase1_repos.add(repo_id)
            phase1_successful.add(repo_id)

    return {
        "phase1_repos_attempted": len(phase1_repos),
        "phase1_successful": len(phase1_successful),
        "phase2_repos_scanned": len(phase2_repos),
        "phase2_runs_total": phase2_runs_total,
        "behavioral_records_produced": behavioral_records_produced,
    }


# ---------------------------------------------------------------------------
# Trace grading
# ---------------------------------------------------------------------------

def _grade_trace(events: list[dict]) -> str:
    """Grade a trace as RICH, BASIC, or EMPTY.

    RICH  -- 2+ agent nodes AND output_hash on agent.task_end AND content flow
    BASIC -- has at least one llm.call_end event
    EMPTY -- zero events or lifecycle-only
    """
    if not events:
        return "EMPTY"

    agent_node_ids: set[str] = set()
    has_output_hash = False
    has_content_flow = False
    has_llm_call_end = False

    # Collect all agent.task_end node_ids that have output_hash, and all
    # agent.task_start events that have input_source (content flow indicator).
    task_end_nodes_with_hash: set[str] = set()

    for event in events:
        event_type = event.get("event_type", "")
        payload = event.get("payload") or {}
        source_node = event.get("source_node") or {}
        node_id = source_node.get("node_id", "")
        node_type = source_node.get("node_type", "")

        # Count agent nodes
        if node_type == "agent" and node_id:
            agent_node_ids.add(node_id)

        if event_type == "agent.task_end":
            if payload.get("output_hash"):
                has_output_hash = True
                if node_id:
                    task_end_nodes_with_hash.add(node_id)

        if event_type == "agent.task_start":
            if payload.get("input_source"):
                has_content_flow = True

        if event_type == "llm.call_end":
            has_llm_call_end = True

    if len(agent_node_ids) >= 2 and has_output_hash and has_content_flow:
        return "RICH"
    if has_llm_call_end:
        return "BASIC"

    # Check for lifecycle-only (only execution.start / execution.end)
    lifecycle_types = {"execution.start", "execution.end"}
    all_types = {e.get("event_type", "") for e in events}
    if all_types <= lifecycle_types:
        return "EMPTY"

    # Has some events but no llm.call_end -- still BASIC if non-trivial
    if len(events) >= 3:
        return "BASIC"

    return "EMPTY"


# ---------------------------------------------------------------------------
# Topology classification (from graph.json)
# ---------------------------------------------------------------------------

def _classify_topology(graph: dict) -> str | None:
    """Classify topology from graph.json structure.

    Returns one of: single_agent, sequential_2_agent, sequential_3_agent,
    hub_and_spoke, with_delegation, or None if unclassifiable.
    """
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if isinstance(nodes, dict):
        nodes = list(nodes.values())
    if isinstance(edges, dict):
        edges = list(edges.values())

    agent_nodes = [
        n for n in nodes
        if isinstance(n, dict) and n.get("type") in ("agent", "orchestrator")
    ]
    n_agents = len(agent_nodes)

    if n_agents == 0:
        return None

    has_delegation = any(
        isinstance(e, dict) and e.get("type") == "delegates_to"
        for e in edges
    )

    if has_delegation:
        return "with_delegation"

    if n_agents == 1:
        return "single_agent"

    # Compute max fan-out among agent nodes
    agent_ids = {n["id"] for n in agent_nodes if "id" in n}
    fan_out: dict[str, int] = defaultdict(int)
    for e in edges:
        if isinstance(e, dict):
            src = e.get("source", "")
            if src in agent_ids:
                fan_out[src] += 1

    max_fan = max(fan_out.values()) if fan_out else 0

    if max_fan > 2:
        return "hub_and_spoke"

    if n_agents == 2:
        return "sequential_2_agent"
    if n_agents == 3:
        return "sequential_3_agent"

    # 4+ agents in a sequential-ish pattern -- classify by count
    return f"sequential_{n_agents}_agent"


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def aggregate(results_dir: Path, behavioral_records_dir: Path | None = None) -> dict:
    """Aggregate all per-repo results into a scan_report.json structure.

    Args:
        results_dir: Directory containing per-repo result subdirectories.
        behavioral_records_dir: Optional path to behavioral_records/ directory.
            If provided, v6_quality, failure_mode_prevalence, and multi-run
            execution stats are computed and included in the report.
    """

    # ---- Execution result counters ----
    tier1_success = 0
    tier1_partial = 0
    tier2_success = 0
    clone_failed = 0
    no_entry_point = 0
    unresolvable_import = 0
    runtime_error = 0
    timeout_no_events = 0
    server_based = 0
    tier2_failed = 0  # Tier 2 repos that failed for any reason
    total_repos = 0

    # ---- Tier breakdown accumulators ----
    tier1_events_total = 0
    tier1_with_events = 0
    tier1_framework: Counter[str] = Counter()
    tier1_rich = 0
    tier1_basic = 0
    tier1_empty = 0

    tier2_events_total = 0
    tier2_with_events = 0
    tier2_framework: Counter[str] = Counter()

    # ---- Event statistics ----
    total_events = 0
    event_type_dist: Counter[str] = Counter()
    unique_agent_nodes: set[str] = set()
    repos_with_delegation = 0
    repos_with_tool_calls = 0
    repos_with_output_hashes = 0

    # ---- Topology census ----
    topology_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "tier1": 0, "tier2": 0})

    # ---- Successful repos list ----
    successful_repos: list[dict] = []

    # ---- Timing: track earliest and latest status.json mtime ----
    earliest_mtime: float | None = None
    latest_mtime: float | None = None

    # ---- Walk results directories ----
    for repo_dir in sorted(results_dir.iterdir()):
        if not repo_dir.is_dir():
            continue

        status_file = repo_dir / "status.json"
        if not status_file.exists():
            continue

        total_repos += 1
        status_data = _load_json(status_file)
        if status_data is None:
            continue

        # Track modification times for duration calculation
        try:
            mtime = status_file.stat().st_mtime
            if earliest_mtime is None or mtime < earliest_mtime:
                earliest_mtime = mtime
            if latest_mtime is None or mtime > latest_mtime:
                latest_mtime = mtime
        except OSError:
            pass

        status = status_data.get("status", "UNKNOWN")
        tier = _read_tier(repo_dir, status_data)
        repo_url = status_data.get("repo", "")

        # ---- Classify into execution_results buckets ----
        if status == "SUCCESS":
            tier1_success += 1
        elif status == "PARTIAL_SUCCESS":
            tier1_partial += 1
        elif status == "TIER2_SUCCESS" or status == "TIER2_PARTIAL":
            tier2_success += 1
        elif status == "CLONE_FAILED":
            clone_failed += 1
        elif status == "NO_ENTRY_POINT":
            no_entry_point += 1
        elif status == "UNRESOLVABLE_IMPORT":
            unresolvable_import += 1
        elif status == "RUNTIME_ERROR":
            runtime_error += 1
        elif status == "TIMEOUT_NO_EVENTS":
            timeout_no_events += 1
        elif status == "SERVER_BASED":
            server_based += 1
        else:
            # Any other failure status
            if tier == 2:
                tier2_failed += 1

        # For non-success tier 2 statuses that we already counted above,
        # also increment tier2_failed
        if tier == 2 and status not in SUCCESS_STATUSES and status not in (
            "CLONE_FAILED", "NO_ENTRY_POINT", "UNRESOLVABLE_IMPORT",
            "RUNTIME_ERROR", "TIMEOUT_NO_EVENTS", "SERVER_BASED",
        ):
            pass  # Already counted in the else branch above
        elif tier == 2 and status not in SUCCESS_STATUSES:
            tier2_failed += 1

        # ---- Process successful repos (those with events) ----
        if status not in SUCCESS_STATUSES:
            continue

        events_file = repo_dir / "stratum_events.jsonl"
        events = _load_jsonl(events_file) if events_file.exists() else []
        event_count = len(events) if events else status_data.get("event_count", 0)

        if event_count == 0:
            continue

        total_events += event_count

        # Framework detection
        repo_framework = ""
        repo_has_delegation = False
        repo_has_tool_calls = False
        repo_has_output_hash = False
        repo_agent_nodes: set[str] = set()

        for event in events:
            event_type = event.get("event_type", "")
            payload = event.get("payload") or {}
            source_node = event.get("source_node") or {}

            # Event type distribution
            if event_type:
                event_type_dist[event_type] += 1

            # Framework detection (first non-empty, non-unknown)
            fw = event.get("framework", "")
            if fw and fw != "unknown" and not repo_framework:
                repo_framework = fw

            # Agent node counting
            node_id = source_node.get("node_id", "")
            node_type = source_node.get("node_type", "")
            if node_type == "agent" and node_id:
                repo_agent_nodes.add(node_id)
                unique_agent_nodes.add(node_id)

            # Also check target_node for agent nodes
            target_node = event.get("target_node") or {}
            t_node_id = target_node.get("node_id", "")
            t_node_type = target_node.get("node_type", "")
            if t_node_type == "agent" and t_node_id:
                repo_agent_nodes.add(t_node_id)
                unique_agent_nodes.add(t_node_id)

            # Delegation detection
            if event_type == "delegation.initiated":
                repo_has_delegation = True

            # Tool call detection
            if event_type == "tool.invoked":
                repo_has_tool_calls = True

            # Output hash detection
            if payload.get("output_hash"):
                repo_has_output_hash = True

        if repo_has_delegation:
            repos_with_delegation += 1
        if repo_has_tool_calls:
            repos_with_tool_calls += 1
        if repo_has_output_hash:
            repos_with_output_hashes += 1

        # Grade the trace
        grade = _grade_trace(events)

        # ---- Tier-specific accumulation ----
        if tier == 1:
            tier1_with_events += 1
            tier1_events_total += event_count
            if repo_framework:
                tier1_framework[repo_framework] += 1
            if grade == "RICH":
                tier1_rich += 1
            elif grade == "BASIC":
                tier1_basic += 1
            else:
                tier1_empty += 1
        else:
            tier2_with_events += 1
            tier2_events_total += event_count
            if repo_framework:
                tier2_framework[repo_framework] += 1

        # ---- Topology from graph.json ----
        graph_file = repo_dir / "graph.json"
        topo_type = None
        if graph_file.exists():
            graph = _load_json(graph_file)
            if graph is not None:
                topo_type = _classify_topology(graph)

        if topo_type:
            topology_counts[topo_type]["count"] += 1
            if tier == 1:
                topology_counts[topo_type]["tier1"] += 1
            else:
                topology_counts[topo_type]["tier2"] += 1

        # ---- Build repo hash from directory name ----
        repo_hash = repo_dir.name

        successful_repos.append({
            "repo": repo_url,
            "hash": repo_hash,
            "framework": repo_framework,
            "tier": tier,
            "events": event_count,
            "agents": len(repo_agent_nodes),
            "grade": grade,
        })

    # ---- Compute total_with_events ----
    total_with_events = tier1_with_events + tier2_with_events

    # ---- Compute scan duration ----
    total_duration_hours = 0.0
    if earliest_mtime is not None and latest_mtime is not None:
        total_duration_hours = round((latest_mtime - earliest_mtime) / 3600.0, 1)

    # ---- Compute cost estimate ----
    cost_estimate = round(total_duration_hours * COST_PER_HOUR, 2)

    # ---- Build report ----
    report: dict = {
        "scan_metadata": {
            "total_repos": total_repos,
            "scan_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "vllm_model": "mistralai/Mistral-7B-Instruct-v0.3",
            "total_duration_hours": total_duration_hours,
            "compute_cost_estimate": f"${cost_estimate:.2f}",
        },
        "execution_results": {
            "tier1_success": tier1_success,
            "tier1_partial": tier1_partial,
            "tier2_success": tier2_success,
            "clone_failed": clone_failed,
            "no_entry_point": no_entry_point,
            "unresolvable_import": unresolvable_import,
            "runtime_error": runtime_error,
            "timeout_no_events": timeout_no_events,
            "server_based": server_based,
            "tier2_failed": tier2_failed,
            "total_with_events": total_with_events,
        },
        "tier_breakdown": {
            "tier1": {
                "total_with_events": tier1_with_events,
                "framework_distribution": dict(tier1_framework.most_common()),
                "avg_events_per_repo": (
                    round(tier1_events_total / tier1_with_events)
                    if tier1_with_events > 0 else 0
                ),
                "rich_traces": tier1_rich,
                "basic_traces": tier1_basic,
                "empty_traces": tier1_empty,
            },
            "tier2": {
                "total_with_events": tier2_with_events,
                "framework_distribution": dict(tier2_framework.most_common()),
                "avg_events_per_repo": (
                    round(tier2_events_total / tier2_with_events)
                    if tier2_with_events > 0 else 0
                ),
                "note": "Tier 2 traces reflect framework default behavior, not repo-specific architecture",
            },
        },
        "event_statistics": {
            "total_events": total_events,
            "event_type_distribution": dict(event_type_dist.most_common()),
            "unique_agent_nodes": len(unique_agent_nodes),
            "repos_with_delegation": repos_with_delegation,
            "repos_with_tool_calls": repos_with_tool_calls,
            "repos_with_output_hashes": repos_with_output_hashes,
        },
        "topology_census": {
            topo: {"count": v["count"], "tier1": v["tier1"], "tier2": v["tier2"]}
            for topo, v in sorted(
                topology_counts.items(), key=lambda x: x[1]["count"], reverse=True
            )
        },
        "successful_repos": sorted(
            successful_repos,
            key=lambda x: x["events"],
            reverse=True,
        ),
    }

    # ---- Behavioral records analysis (v6 quality, failure modes, multi-run stats) ----
    if behavioral_records_dir is not None and behavioral_records_dir.is_dir():
        report["v6_quality"] = _compute_v6_quality(behavioral_records_dir)
        report["failure_mode_prevalence"] = _compute_failure_mode_prevalence(
            behavioral_records_dir
        )
        multi_run_stats = _compute_multi_run_stats(behavioral_records_dir)
        report["execution_results"].update(multi_run_stats)

    return report


# ---------------------------------------------------------------------------
# Human-readable summary on stderr
# ---------------------------------------------------------------------------

def _print_summary(report: dict) -> None:
    """Print a human-readable summary to stderr."""
    meta = report["scan_metadata"]
    er = report["execution_results"]
    tb = report["tier_breakdown"]
    es = report["event_statistics"]
    tc = report["topology_census"]

    w = sys.stderr.write

    w(f"\n{'=' * 65}\n")
    w("SCAN REPORT\n")
    w(f"{'=' * 65}\n")
    w(f"Date:             {meta['scan_date']}\n")
    w(f"Total repos:      {meta['total_repos']}\n")
    w(f"Duration:         {meta['total_duration_hours']} hours\n")
    w(f"Cost estimate:    {meta['compute_cost_estimate']}\n")
    w(f"Model:            {meta['vllm_model']}\n")
    w("\n")

    # Execution results
    w("--- Execution Results ---\n")
    w(f"  Tier 1 success:       {er['tier1_success']:>5d}\n")
    w(f"  Tier 1 partial:       {er['tier1_partial']:>5d}\n")
    w(f"  Tier 2 success:       {er['tier2_success']:>5d}\n")
    w(f"  Clone failed:         {er['clone_failed']:>5d}\n")
    w(f"  No entry point:       {er['no_entry_point']:>5d}\n")
    w(f"  Unresolvable import:  {er['unresolvable_import']:>5d}\n")
    w(f"  Runtime error:        {er['runtime_error']:>5d}\n")
    w(f"  Timeout (no events):  {er['timeout_no_events']:>5d}\n")
    w(f"  Server-based:         {er['server_based']:>5d}\n")
    w(f"  Tier 2 failed:        {er['tier2_failed']:>5d}\n")
    w(f"  Total with events:    {er['total_with_events']:>5d}\n")
    if "phase1_repos_attempted" in er:
        w("  --- Multi-run Stats ---\n")
        w(f"  Phase 1 attempted:    {er['phase1_repos_attempted']:>5d}\n")
        w(f"  Phase 1 successful:   {er['phase1_successful']:>5d}\n")
        w(f"  Phase 2 repos:        {er['phase2_repos_scanned']:>5d}\n")
        w(f"  Phase 2 runs total:   {er['phase2_runs_total']:>5d}\n")
        w(f"  Behavioral records:   {er['behavioral_records_produced']:>5d}\n")
    w("\n")

    # Tier 1 breakdown
    t1 = tb["tier1"]
    w("--- Tier 1 ---\n")
    w(f"  Repos with events:  {t1['total_with_events']}\n")
    w(f"  Avg events/repo:    {t1['avg_events_per_repo']}\n")
    w(f"  RICH traces:        {t1['rich_traces']}\n")
    w(f"  BASIC traces:       {t1['basic_traces']}\n")
    w(f"  EMPTY traces:       {t1['empty_traces']}\n")
    if t1["framework_distribution"]:
        w("  Frameworks:\n")
        for fw, cnt in sorted(t1["framework_distribution"].items(), key=lambda x: x[1], reverse=True):
            w(f"    {fw:<20s} {cnt:>5d}\n")
    w("\n")

    # Tier 2 breakdown
    t2 = tb["tier2"]
    w("--- Tier 2 ---\n")
    w(f"  Repos with events:  {t2['total_with_events']}\n")
    w(f"  Avg events/repo:    {t2['avg_events_per_repo']}\n")
    if t2["framework_distribution"]:
        w("  Frameworks:\n")
        for fw, cnt in sorted(t2["framework_distribution"].items(), key=lambda x: x[1], reverse=True):
            w(f"    {fw:<20s} {cnt:>5d}\n")
    w("\n")

    # Event statistics
    w("--- Event Statistics ---\n")
    w(f"  Total events:          {es['total_events']}\n")
    w(f"  Unique agent nodes:    {es['unique_agent_nodes']}\n")
    w(f"  Repos w/ delegation:   {es['repos_with_delegation']}\n")
    w(f"  Repos w/ tool calls:   {es['repos_with_tool_calls']}\n")
    w(f"  Repos w/ output hash:  {es['repos_with_output_hashes']}\n")
    if es["event_type_distribution"]:
        w("  Event types (top 10):\n")
        for et, cnt in list(es["event_type_distribution"].items())[:10]:
            w(f"    {et:<30s} {cnt:>6d}\n")
    w("\n")

    # Topology census
    if tc:
        w("--- Topology Census ---\n")
        for topo, vals in tc.items():
            w(f"  {topo:<25s}  total={vals['count']:>4d}  T1={vals['tier1']:>4d}  T2={vals['tier2']:>4d}\n")
        w("\n")

    # v6 quality (only present if --behavioral-records-dir was provided)
    v6q = report.get("v6_quality")
    if v6q:
        w("--- v6 Quality ---\n")
        w(f"  Edge validation:         {v6q['records_with_edge_validation']:>5d}\n")
        w(f"  Emergent edges:          {v6q['records_with_emergent_edges']:>5d}\n")
        w(f"  Error propagation:       {v6q['records_with_error_propagation']:>5d}\n")
        w(f"  Failure modes manifested:{v6q['records_with_failure_modes_manifested']:>5d}\n")
        w(f"  Monitoring baselines:    {v6q['records_with_monitoring_baselines']:>5d}\n")
        w(f"  Avg prediction match:    {v6q['avg_structural_prediction_match_rate']:.2f}\n")
        w("\n")

    # Failure mode prevalence (only present if --behavioral-records-dir was provided)
    fmp = report.get("failure_mode_prevalence")
    if fmp:
        w("--- Failure Mode Prevalence ---\n")
        for mode_id, stats in fmp.items():
            w(f"  {mode_id:<20s}  affected={stats['repos_affected']:>4d}  rate={stats['manifestation_rate']:.2f}\n")
        w("\n")

    # Top successful repos
    top_n = 10
    repos = report["successful_repos"][:top_n]
    if repos:
        w(f"--- Top {min(top_n, len(repos))} Successful Repos (by event count) ---\n")
        for r in repos:
            w(f"  [{r['grade']}] T{r['tier']} {r['repo']:<45s} "
              f"events={r['events']:<5d} agents={r['agents']}\n")
        w("\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-scan aggregator. Reads results/*/ and produces scan_report.json.",
    )
    parser.add_argument(
        "results_dir",
        help="Directory containing per-repo result subdirectories",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output path for scan_report.json (default: <results_dir>/scan_report.json)",
    )
    parser.add_argument(
        "--behavioral-records-dir",
        default=None,
        help="Path to behavioral_records/ directory. If provided, compute v6 quality "
             "metrics, failure mode prevalence, and multi-run execution statistics.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    behavioral_records_dir = None
    if args.behavioral_records_dir:
        behavioral_records_dir = Path(args.behavioral_records_dir)
        if not behavioral_records_dir.is_dir():
            print(
                f"Warning: {behavioral_records_dir} is not a directory, "
                "skipping behavioral records analysis",
                file=sys.stderr,
            )
            behavioral_records_dir = None

    report = aggregate(results_dir, behavioral_records_dir=behavioral_records_dir)

    # Determine output path
    output_path = Path(args.output) if args.output else results_dir / "scan_report.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print human-readable summary to stderr
    _print_summary(report)
    print(f"Report saved to: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
