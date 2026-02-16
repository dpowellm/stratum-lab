#!/usr/bin/env python3
"""Validate per-repo scan output directories with enriched output.

Checks status.json, stratum_events.jsonl schema, topology, content flow,
model remapping, quality grading, v6 behavioral record compatibility,
and multi-run awareness.

Correct field paths:
    event["source_node"]["node_id"]     -- NOT event["source_node_id"]
    event["source_node"]["node_type"]   -- NOT event.get("node_type")
    event["payload"]["output_hash"]     -- NOT event["data"]["output_hash"]
    event["payload"]["model_requested"] -- model requested
    event["payload"]["model_actual"]    -- model actually used
    event["target_node"]["node_id"]     -- target node
    event["edge_type"]                  -- edge type

Usage: python validate_smoke.py <output_dir> [--verbose] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

VALID_STATUSES = {
    "SUCCESS", "PARTIAL_SUCCESS", "NO_EVENTS",
    "TIMEOUT_NO_EVENTS", "UNRESOLVABLE_IMPORT",
    "RUNTIME_ERROR", "CLONE_FAILED", "NO_ENTRY_POINT",
    "SERVER_BASED", "TIER2_SUCCESS", "TIER2_PARTIAL",
    "UNRUNNABLE",
}
EVENTS_EXPECTED = {"SUCCESS", "PARTIAL_SUCCESS", "TIER2_SUCCESS", "TIER2_PARTIAL"}
REQUIRED_EVENT_FIELDS = {
    "event_id", "timestamp_ns", "run_id", "repo_id", "framework", "event_type",
}
KNOWN_EVENT_TYPES = {
    "execution.start", "execution.end",
    "llm.call_start", "llm.call_end", "tool.call_failure",
    "agent.task_start", "agent.task_end",
    "tool.invoked", "tool.completed",
    "delegation.initiated", "delegation.completed",
    "state.access", "routing.decision",
    "edge.traversed", "data.read", "data.write",
    "message.received", "reply.generated",
    "speaker.selected",
    "error.occurred", "error.propagated",
    "guardrail.triggered", "guardrail.passed",
}
# Event types that are lifecycle-only (not substantive for grading)
LIFECYCLE_EVENT_TYPES = {
    "execution.start", "execution.end", "error.occurred", "error.propagated",
}


def _node_field(node: dict | None, field: str) -> str | None:
    """Extract a field from a source_node / target_node dict."""
    return node.get(field) if isinstance(node, dict) else None


def _short_node_id(node_id: str) -> str:
    """Return abbreviated node_id for display (last component or truncated)."""
    parts = node_id.split(":")
    if len(parts) >= 2:
        return parts[1]
    return node_id


def _grade(
    agent_node_ids: set[str],
    output_hashes: dict[str, str],
    content_flows: list[tuple[str, str]],
    event_types: set[str],
    llm_call_count: int,
    has_parent_or_input: bool = False,
) -> str:
    """Grade a repo's event quality.

    RICH:  2+ unique agent nodes AND output_hash on agent.task_end AND
           parent_node_id/input_source on agent.task_start AND at least one
           of: delegation.initiated, tool.invoked, state.access, routing.decision.
    BASIC: Has llm.call_end events with agent.task_start/end lifecycle.
    THIN:  Only llm events, no agent lifecycle.
    EMPTY: Zero events or only error/lifecycle events.
    """
    has_agent_nodes = len(agent_node_ids) >= 2
    has_output_hash = len(output_hashes) > 0
    rich_event_types = {"delegation.initiated", "tool.invoked",
                        "state.access", "routing.decision"}
    has_rich_event = bool(event_types & rich_event_types)
    has_agent_lifecycle = bool(
        {"agent.task_start", "agent.task_end"} & event_types
    )

    if (has_agent_nodes and has_output_hash and has_parent_or_input
            and has_rich_event):
        return "RICH"
    if llm_call_count > 0 and has_agent_lifecycle:
        return "BASIC"
    if llm_call_count > 0:
        return "THIN"
    # Only lifecycle/error events or zero events
    substantive = event_types - LIFECYCLE_EVENT_TYPES
    if not substantive:
        return "EMPTY"
    return "THIN"


# -- v6 event types that feed specific behavioral record sections -----------
_V6_EDGE_VALIDATION_TYPES = {
    "edge.traversed", "delegation.initiated", "delegation.completed",
    "tool.invoked", "tool.completed",
}
_V6_NODE_ACTIVATION_TYPES = {
    "agent.task_start", "agent.task_end",
    "llm.call_start", "llm.call_end",
    "tool.invoked", "tool.completed",
}
_V6_BEHAVIORAL_DIVERSITY_TYPES = {
    "agent.task_start", "agent.task_end",
    "llm.call_start", "llm.call_end",
    "delegation.initiated", "delegation.completed",
    "tool.invoked", "tool.completed",
    "state.access", "routing.decision",
}


def check_v6_compatibility(
    event_types: set[str],
) -> dict:
    """Check whether collected events are sufficient for v6 behavioral record.

    Returns a dict with:
      - compatible: "YES" | "NO" | "PARTIAL"
      - sections: dict mapping v6 section name -> bool (will be populated)
      - present_v6_types: sorted list of event types that feed v6 sections
      - missing_for_full: sorted list of event types needed for full v6
    """
    present_edge = event_types & _V6_EDGE_VALIDATION_TYPES
    present_node = event_types & _V6_NODE_ACTIVATION_TYPES
    present_diversity = event_types & _V6_BEHAVIORAL_DIVERSITY_TYPES

    sections = {
        "edge_validation": bool(present_edge),
        "node_activation": bool(present_node),
        "behavioral_diversity": len(present_diversity) >= 3,
    }

    all_v6_types = (
        _V6_EDGE_VALIDATION_TYPES
        | _V6_NODE_ACTIVATION_TYPES
        | _V6_BEHAVIORAL_DIVERSITY_TYPES
    )
    present_v6 = event_types & all_v6_types
    missing = all_v6_types - event_types

    populated = sum(sections.values())
    total_sections = len(sections)

    if populated == total_sections:
        compat = "YES"
    elif populated == 0:
        compat = "NO"
    else:
        compat = "PARTIAL"

    return {
        "compatible": compat,
        "sections": sections,
        "present_v6_types": sorted(present_v6),
        "missing_for_full": sorted(missing),
    }


def _collect_run_files(repo_dir: Path) -> dict:
    """Detect multi-run event files (events_run_*.jsonl).

    Returns a dict with:
      - run_files: list of filenames found
      - run_count: number of run files
      - enough_for_activation_rate: bool (need 3+)
      - per_run_event_counts: dict filename -> event count
    """
    run_files = sorted(repo_dir.glob("events_run_*.jsonl"))
    per_run: dict[str, int] = {}
    for rf in run_files:
        count = 0
        try:
            with open(rf) as f:
                for line in f:
                    if line.strip():
                        count += 1
        except OSError:
            count = -1  # indicate read error
        per_run[rf.name] = count

    return {
        "run_files": [rf.name for rf in run_files],
        "run_count": len(run_files),
        "enough_for_activation_rate": len(run_files) >= 3,
        "per_run_event_counts": per_run,
    }


def validate_repo(
    repo_dir: Path, verbose: bool = False,
) -> tuple[bool, list[str], dict]:
    """Validate a single repo output directory.

    Returns (passed, issues, meta).
    """
    issues: list[str] = []
    meta: dict = {"dir": repo_dir.name}

    # -- status.json --------------------------------------------------------
    status_file = repo_dir / "status.json"
    if not status_file.exists():
        return False, ["status.json missing"], meta
    try:
        with open(status_file) as f:
            status_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return False, [f"status.json invalid: {e}"], meta

    status = status_data.get("status", "")
    meta["status"] = status
    meta["tier"] = status_data.get("tier", 1)
    meta["repo"] = status_data.get("repo", "")

    for name in ("tier.txt", "entry_point.txt"):
        p = repo_dir / name
        if p.exists():
            meta[name.replace(".txt", "")] = p.read_text().strip()

    if status not in VALID_STATUSES:
        issues.append(f"unknown status: {status!r}")

    if status not in EVENTS_EXPECTED:
        meta["event_count"] = 0
        meta["grade"] = "EMPTY"
        return len(issues) == 0, issues, meta

    # -- stratum_events.jsonl -----------------------------------------------
    events_file = repo_dir / "stratum_events.jsonl"
    if not events_file.exists():
        issues.append(f"status={status} but stratum_events.jsonl missing")
        return False, issues, meta

    event_count = 0
    seen_ids: set[str] = set()
    event_types_seen: Counter[str] = Counter()
    node_ids: set[str] = set()
    node_names: set[str] = set()
    # node_id -> node_type for agent identification
    node_types: dict[str, str] = {}
    edge_count = 0

    # Content flow tracking
    # node_id -> output_hash (from agent.task_end payload)
    output_hashes: dict[str, str] = {}
    # (source_node_id, target_node_id) pairs where content flows
    content_flows: list[tuple[str, str]] = []
    # node_id -> input_source (from agent.task_start payload)
    input_sources: dict[str, str] = {}
    # node_id -> parent_node_id (from agent.task_start payload)
    parent_node_ids: dict[str, str] = {}

    # Model remapping tracking
    # (model_requested, model_actual) -> count
    model_remaps: Counter[tuple[str, str]] = Counter()
    llm_call_count = 0

    # Delegation tracking
    delegation_count = 0
    # Tool call tracking
    tool_call_count = 0

    # Output hash counts
    events_with_output_hash = 0

    # Ordered sequence of agent node_ids for topology display
    agent_sequence: list[str] = []
    event_sequence: list[str] = []  # ordered event types for topology

    try:
        with open(events_file) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    issues.append(f"malformed JSON at events line {line_num}")
                    continue

                event_count += 1
                missing = REQUIRED_EVENT_FIELDS - set(event.keys())
                if missing:
                    issues.append(f"event {line_num} missing fields: {missing}")

                et = event.get("event_type", "")
                if et:
                    event_types_seen[et] += 1
                    if et not in KNOWN_EVENT_TYPES:
                        pass  # warn later, don't fail

                eid = event.get("event_id", "")
                if eid:
                    if eid in seen_ids:
                        issues.append(
                            f"duplicate event_id line {line_num}: {eid}"
                        )
                    seen_ids.add(eid)

                # -- Topology: source_node / target_node dicts ---------------
                source_node = event.get("source_node")
                target_node = event.get("target_node")

                src_nid = _node_field(source_node, "node_id")
                src_ntype = _node_field(source_node, "node_type")
                src_name = _node_field(source_node, "node_name")
                tgt_nid = _node_field(target_node, "node_id")
                tgt_ntype = _node_field(target_node, "node_type")
                tgt_name = _node_field(target_node, "node_name")

                for nid, ntype, nm in [
                    (src_nid, src_ntype, src_name),
                    (tgt_nid, tgt_ntype, tgt_name),
                ]:
                    if nid:
                        node_ids.add(nid)
                        if ntype:
                            node_types[nid] = ntype
                    if nm:
                        node_names.add(nm)

                if event.get("edge_type"):
                    edge_count += 1

                # -- Payload extraction --------------------------------------
                payload = event.get("payload") or {}

                # Output hash tracking
                if payload.get("output_hash"):
                    events_with_output_hash += 1

                # agent.task_end -> output_hash
                if et == "agent.task_end" and src_nid:
                    oh = payload.get("output_hash")
                    if oh:
                        output_hashes[src_nid] = oh
                    if src_nid not in agent_sequence:
                        agent_sequence.append(src_nid)

                # agent.task_start -> input_source, parent_node_id
                if et == "agent.task_start" and src_nid:
                    isrc = payload.get("input_source")
                    if isrc:
                        input_sources[src_nid] = isrc
                    pnid = payload.get("parent_node_id")
                    if pnid:
                        parent_node_ids[src_nid] = pnid
                    if src_nid not in agent_sequence:
                        agent_sequence.append(src_nid)

                # LLM calls: model remapping
                if et == "llm.call_start":
                    llm_call_count += 1
                    mr = payload.get("model_requested", "")
                    ma = payload.get("model_actual", "")
                    if mr or ma:
                        model_remaps[(mr, ma)] += 1

                # Delegations
                if et in ("delegation.initiated", "delegation.completed"):
                    delegation_count += 1

                # Tool calls
                if et in ("tool.invoked", "tool.completed"):
                    tool_call_count += 1

                # Track event sequence for topology
                event_sequence.append(et)

    except OSError as e:
        issues.append(f"error reading events: {e}")

    if event_count == 0:
        issues.append(f"status={status} but events file is empty")

    reported = status_data.get("event_count", 0)
    if reported and reported != event_count:
        issues.append(
            f"event_count mismatch: status says {reported}, file has {event_count}"
        )

    # -- Content flow resolution --------------------------------------------
    # If input_source matches a node_id that produced an output_hash,
    # it's traceable content flow.
    for consumer_nid, isrc in input_sources.items():
        if isrc in output_hashes:
            content_flows.append((isrc, consumer_nid))

    # -- Identify agent nodes -----------------------------------------------
    agent_node_ids: set[str] = set()
    for nid in node_ids:
        ntype = node_types.get(nid, "")
        if ntype in ("agent", "orchestrator"):
            agent_node_ids.add(nid)
        elif nid in output_hashes or nid in input_sources:
            # Appeared in agent.task_start or agent.task_end
            agent_node_ids.add(nid)

    # -- Grading ------------------------------------------------------------
    has_parent_or_input = bool(input_sources or parent_node_ids)
    grade = _grade(
        agent_node_ids, output_hashes, content_flows,
        set(event_types_seen), llm_call_count,
        has_parent_or_input=has_parent_or_input,
    )

    # -- v6 compatibility ---------------------------------------------------
    v6_compat = check_v6_compatibility(set(event_types_seen))

    # -- Multi-run awareness ------------------------------------------------
    multi_run = _collect_run_files(repo_dir)

    # -- Build topology string ----------------------------------------------
    topology_parts: list[str] = []
    for nid in agent_sequence:
        short = _short_node_id(nid)
        topology_parts.append(short)
        # Check if there was an LLM call associated (interleave LLM)
        if nid in output_hashes or nid in input_sources:
            topology_parts.append("LLM")
    # Deduplicate consecutive duplicates
    deduped: list[str] = []
    for p in topology_parts:
        if not deduped or deduped[-1] != p:
            deduped.append(p)
    topology_str = " -> ".join(deduped) if deduped else ""

    # -- Unknown event types ------------------------------------------------
    unknown = set(event_types_seen) - KNOWN_EVENT_TYPES
    if unknown:
        meta["unknown_event_types"] = sorted(unknown)

    # -- Populate meta ------------------------------------------------------
    meta["event_count"] = event_count
    meta["event_types"] = dict(event_types_seen)
    meta["node_ids"] = sorted(node_ids)
    meta["node_names"] = sorted(node_names)
    meta["node_types"] = node_types
    meta["agent_node_ids"] = sorted(agent_node_ids)
    meta["edge_count"] = edge_count
    meta["grade"] = grade
    meta["llm_call_count"] = llm_call_count
    meta["model_remaps"] = {
        f"{mr} -> {ma}": cnt for (mr, ma), cnt in model_remaps.most_common()
    }
    meta["output_hashes"] = output_hashes
    meta["events_with_output_hash"] = events_with_output_hash
    meta["content_flows"] = [
        {"from": src, "to": tgt} for src, tgt in content_flows
    ]
    meta["delegation_count"] = delegation_count
    meta["tool_call_count"] = tool_call_count
    meta["topology"] = topology_str
    meta["parent_node_ids"] = parent_node_ids
    meta["v6_compatibility"] = v6_compat
    meta["multi_run"] = multi_run

    return len(issues) == 0, issues, meta


def _print_repo_detail(meta: dict) -> None:
    """Print enriched per-repo detail block."""
    tier = meta.get("tier", 1)
    event_count = meta.get("event_count", 0)
    event_types = meta.get("event_types", {})
    agent_nids = meta.get("agent_node_ids", [])
    node_types = meta.get("node_types", {})
    llm_call_count = meta.get("llm_call_count", 0)
    model_remaps = meta.get("model_remaps", {})
    events_with_hash = meta.get("events_with_output_hash", 0)
    content_flows = meta.get("content_flows", [])
    delegation_count = meta.get("delegation_count", 0)
    tool_call_count = meta.get("tool_call_count", 0)
    topology = meta.get("topology", "")
    grade = meta.get("grade", "EMPTY")

    print("=" * 60)
    print(f"Validating: results/{meta['dir']} (Tier: {tier})")
    print(f"Total events: {event_count}")

    if event_types:
        print("Event types:")
        for et, cnt in sorted(event_types.items(), key=lambda x: x[0]):
            print(f"  {et}: {cnt}")

    print()
    print(f"Agent nodes ({len(agent_nids)}):")
    for nid in agent_nids:
        ntype = node_types.get(nid, "agent")
        print(f"  {nid} ({ntype})")

    print()
    print(f"LLM calls: {llm_call_count}")
    if model_remaps:
        for pair, cnt in model_remaps.items():
            print(f"  model_requested: {pair.split(' -> ')[0]} "
                  f"-> model_actual: {pair.split(' -> ')[1]}")

    print()
    hash_label = f"{events_with_hash}/{event_count} events have output_hash"
    print(f"Output hashes: {hash_label}")
    if content_flows:
        for flow in content_flows:
            src_short = _short_node_id(flow["from"])
            tgt_short = _short_node_id(flow["to"])
            output_hashes = meta.get("output_hashes", {})
            src_hash = output_hashes.get(flow["from"], "")
            hash_abbr = src_hash[:7] if src_hash else "?"
            print(f"Content flow: {src_short} output (hash {hash_abbr}) "
                  f"-> {tgt_short} input")
    else:
        print("Content flow: none detected")

    print()
    print(f"Delegations: {delegation_count}")
    print(f"Tool calls: {tool_call_count}")

    print()
    if topology:
        print(f"Topology: {topology}")
    else:
        print("Topology: (none)")

    print()
    print(f"Grade: {grade} ", end="")
    if grade == "RICH":
        print("(2+ agents, output_hash, parent/input, delegation/tool/state/routing)")
    elif grade == "BASIC":
        print("(LLM calls with agent lifecycle)")
    elif grade == "THIN":
        print("(LLM events only, no agent lifecycle)")
    else:
        print("(zero events or only error/lifecycle events)")

    # -- v6 compatibility detail -------------------------------------------
    v6 = meta.get("v6_compatibility", {})
    if v6:
        compat = v6.get("compatible", "NO")
        print()
        print(f"v6 compatibility: {compat}")
        sections = v6.get("sections", {})
        for sec_name, populated in sections.items():
            status_label = "will populate" if populated else "EMPTY"
            print(f"  {sec_name}: {status_label}")
        present = v6.get("present_v6_types", [])
        if present:
            print(f"  v6 event types present: {', '.join(present)}")

    # -- Multi-run detail --------------------------------------------------
    mr = meta.get("multi_run", {})
    if mr and mr.get("run_count", 0) > 0:
        print()
        run_count = mr["run_count"]
        enough = mr["enough_for_activation_rate"]
        print(f"Runs available: {run_count} "
              f"({'enough for full v6' if enough else 'need 3+ for activation rate'})")
        for fname, cnt in mr.get("per_run_event_counts", {}).items():
            print(f"  {fname}: {cnt} events")

    # Print issues if any
    issues = meta.get("issues", [])
    if issues:
        print("Issues:")
        for issue in issues:
            print(f"  - {issue}")

    print("=" * 60)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate per-repo scan output directories.",
    )
    ap.add_argument(
        "output_dir",
        help="Base directory containing per-repo subdirectories",
    )
    ap.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-repo details",
    )
    ap.add_argument(
        "--json", action="store_true",
        help="Output results as JSON",
    )
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        print(f"Error: {output_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    total = passed_count = failed_count = 0
    status_counts: Counter[str] = Counter()
    tier_counts: Counter[int] = Counter()
    grade_counts: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    all_results: list[dict] = []

    for repo_dir in sorted(output_dir.iterdir()):
        if not repo_dir.is_dir():
            continue
        total += 1
        ok, issues, meta = validate_repo(repo_dir, verbose=args.verbose)
        meta["passed"] = ok
        meta["issues"] = issues
        all_results.append(meta)

        st = meta.get("status", "UNKNOWN")
        status_counts[st] += 1
        tier_counts[meta.get("tier", 1)] += 1
        if "grade" in meta:
            grade_counts[meta["grade"]] += 1

        if ok:
            passed_count += 1
        else:
            failed_count += 1
            for issue in issues:
                failure_reasons[issue.split(":")[0].split(" ")[0]] += 1

        # Per-repo enriched output in verbose mode
        if args.verbose:
            _print_repo_detail(meta)
            print()

    # -- Aggregate v6 readiness for JSON/summary ----------------------------
    v6_summary: dict = {"YES": 0, "PARTIAL": 0, "NO": 0}
    total_run_files = 0
    repos_with_3plus_runs = 0
    for r in all_results:
        v6 = r.get("v6_compatibility", {})
        if v6:
            v6_summary[v6.get("compatible", "NO")] = (
                v6_summary.get(v6.get("compatible", "NO"), 0) + 1
            )
        mr = r.get("multi_run", {})
        if mr:
            total_run_files += mr.get("run_count", 0)
            if mr.get("enough_for_activation_rate"):
                repos_with_3plus_runs += 1

    # -- JSON output --------------------------------------------------------
    if args.json:
        json.dump(
            {
                "total": total,
                "passed": passed_count,
                "failed": failed_count,
                "status_counts": dict(status_counts),
                "tier_counts": {
                    str(k): v for k, v in sorted(tier_counts.items())
                },
                "grade_counts": dict(grade_counts),
                "v6_readiness": {
                    "compatibility": v6_summary,
                    "total_run_files": total_run_files,
                    "repos_with_3plus_runs": repos_with_3plus_runs,
                },
                "results": all_results,
            },
            sys.stdout,
            indent=2,
            default=str,
        )
        return

    # -- Human-readable summary ---------------------------------------------
    print()
    print("=" * 60)
    if total:
        pct = 100 * passed_count / total
        print(f"Validation: {passed_count}/{total} passed ({pct:.0f}%)")
    else:
        print("No repos found")
    if failed_count:
        print(f"  Failed: {failed_count}")
    print()

    print("Status distribution:")
    for st, cnt in status_counts.most_common():
        bar = "#" * min(cnt, 40)
        print(f"  {st:<25s} {cnt:>4d}  {bar}")

    print()
    print("Tier distribution:")
    for tier, cnt in sorted(tier_counts.items()):
        print(f"  Tier {tier}: {cnt}")

    if grade_counts:
        print()
        print("Quality grade distribution:")
        for g, cnt in grade_counts.most_common():
            bar = "#" * min(cnt, 40)
            print(f"  {g:<10s} {cnt:>4d}  {bar}")

    # -- v6 readiness summary ----------------------------------------------
    v6_counts: Counter[str] = Counter()
    total_runs = 0
    repos_with_enough_runs = 0
    v6_event_types_present: Counter[str] = Counter()
    for r in all_results:
        v6 = r.get("v6_compatibility", {})
        if v6:
            v6_counts[v6.get("compatible", "NO")] += 1
            for et in v6.get("present_v6_types", []):
                v6_event_types_present[et] += 1
        mr = r.get("multi_run", {})
        if mr:
            total_runs += mr.get("run_count", 0)
            if mr.get("enough_for_activation_rate"):
                repos_with_enough_runs += 1

    if v6_counts:
        print()
        print("v6 readiness:")
        for compat_level in ("YES", "PARTIAL", "NO"):
            cnt = v6_counts.get(compat_level, 0)
            if cnt:
                print(f"  {compat_level:<10s} {cnt:>4d}")
        print(f"  Total multi-run files: {total_runs}")
        print(f"  Repos with 3+ runs (full v6): {repos_with_enough_runs}")
        if v6_event_types_present:
            print("  v6-feeding event types across repos:")
            for et, cnt in v6_event_types_present.most_common():
                print(f"    {et}: {cnt} repos")

    # Topology summary for successful repos
    topo = [r for r in all_results if r.get("grade") and r.get("passed")]
    if topo:
        print()
        print("Topology summary (successful repos):")
        for r in topo:
            nids = r.get("node_ids", [])
            names = r.get("node_names", [])
            ec = r.get("event_count", 0)
            edges = r.get("edge_count", 0)
            print(
                f"  {r['dir']}  events={ec}   "
                f"nodes={len(nids)}   edges={edges}   names={names}"
            )

    if failure_reasons:
        print()
        print("Common validation failures:")
        for reason, cnt in failure_reasons.most_common(10):
            print(f"  {reason:<30s} {cnt:>4d}")

    sys.exit(1 if failed_count else 0)


if __name__ == "__main__":
    main()
