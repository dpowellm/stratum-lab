#!/usr/bin/env python3
"""Validate per-repo scan output directories with enriched output.

Checks status.json, stratum_events.jsonl schema, topology, content flow,
model remapping, and quality grading.

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
) -> str:
    """Grade a repo's event quality.

    RICH:  2+ unique agent nodes AND output_hash on agent.task_end AND
           traceable content flow (agent B's input_source matches agent A
           who has output_hash).
    BASIC: Has llm.call_end events but missing node attribution or content flow.
    EMPTY: Zero events or only error/lifecycle events.
    """
    has_agent_nodes = len(agent_node_ids) >= 2
    has_output_hash = len(output_hashes) > 0
    has_content_flow = len(content_flows) > 0

    if has_agent_nodes and has_output_hash and has_content_flow:
        return "RICH"
    if llm_call_count > 0:
        return "BASIC"
    # Only lifecycle/error events or zero events
    substantive = event_types - LIFECYCLE_EVENT_TYPES
    if not substantive:
        return "EMPTY"
    return "BASIC"


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

                # agent.task_start -> input_source
                if et == "agent.task_start" and src_nid:
                    isrc = payload.get("input_source")
                    if isrc:
                        input_sources[src_nid] = isrc
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
    grade = _grade(
        agent_node_ids, output_hashes, content_flows,
        set(event_types_seen), llm_call_count,
    )

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
        print("(has nodes + edges + content flow)")
    elif grade == "BASIC":
        print("(has llm.call_end but missing node attribution or content flow)")
    else:
        print("(zero events or only error/lifecycle events)")

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
