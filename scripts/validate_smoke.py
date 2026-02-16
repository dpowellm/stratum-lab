#!/usr/bin/env python3
"""Validate per-repo scan output directories.

Checks status.json, stratum_events.jsonl schema, topology, and quality.
Correct field paths: event["source_node"]["node_id"], event["payload"][...].

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


def _node_field(node: dict | None, field: str) -> str | None:
    """Extract a field from a source_node / target_node dict."""
    return node.get(field) if isinstance(node, dict) else None


def _grade(event_count: int, node_ids: set, event_types: set) -> str:
    if event_count >= 10 and len(node_ids) >= 3 and len(event_types) >= 2:
        return "RICH"
    if event_count >= 3 and len(node_ids) >= 1:
        return "BASIC"
    return "EMPTY"


def validate_repo(repo_dir: Path) -> tuple[bool, list[str], dict]:
    """Validate a single repo output directory. Returns (passed, issues, meta)."""
    issues: list[str] = []
    meta: dict = {"dir": repo_dir.name}

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
        return len(issues) == 0, issues, meta

    events_file = repo_dir / "stratum_events.jsonl"
    if not events_file.exists():
        issues.append(f"status={status} but stratum_events.jsonl missing")
        return False, issues, meta

    event_count = 0
    seen_ids: set[str] = set()
    event_types_seen: Counter[str] = Counter()
    node_ids: set[str] = set()
    node_names: set[str] = set()
    edge_count = 0

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

                eid = event.get("event_id", "")
                if eid:
                    if eid in seen_ids:
                        issues.append(f"duplicate event_id line {line_num}: {eid}")
                    seen_ids.add(eid)

                # Topology -- node_id / node_name live inside source_node / target_node dicts
                for node_key in ("source_node", "target_node"):
                    nid = _node_field(event.get(node_key), "node_id")
                    if nid:
                        node_ids.add(nid)
                    nm = _node_field(event.get(node_key), "node_name")
                    if nm:
                        node_names.add(nm)

                if event.get("edge_type"):
                    edge_count += 1
    except OSError as e:
        issues.append(f"error reading events: {e}")

    if event_count == 0:
        issues.append(f"status={status} but events file is empty")

    reported = status_data.get("event_count", 0)
    if reported and reported != event_count:
        issues.append(f"event_count mismatch: status says {reported}, file has {event_count}")

    meta["event_count"] = event_count
    meta["event_types"] = dict(event_types_seen)
    meta["node_ids"] = sorted(node_ids)
    meta["node_names"] = sorted(node_names)
    meta["edge_count"] = edge_count
    meta["grade"] = _grade(event_count, node_ids, set(event_types_seen))
    unknown = set(event_types_seen) - KNOWN_EVENT_TYPES
    if unknown:
        meta["unknown_event_types"] = sorted(unknown)

    return len(issues) == 0, issues, meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate per-repo scan output directories.")
    ap.add_argument("output_dir", help="Base directory containing per-repo subdirectories")
    ap.add_argument("--verbose", "-v", action="store_true", help="Print per-repo details")
    ap.add_argument("--json", action="store_true", help="Output results as JSON")
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
        ok, issues, meta = validate_repo(repo_dir)
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
            if args.verbose:
                tier_tag = f" [T{meta.get('tier',1)}]" if meta.get("tier",1) != 1 else ""
                grade_tag = f" [{meta['grade']}]" if "grade" in meta else ""
                print(f"  PASS{tier_tag}{grade_tag}  {repo_dir.name}  ({st})")
        else:
            failed_count += 1
            if args.verbose:
                print(f"  FAIL  {repo_dir.name}  ({st})")
                for issue in issues:
                    print(f"        - {issue}")
            for issue in issues:
                failure_reasons[issue.split(":")[0].split(" ")[0]] += 1

    if args.json:
        json.dump({
            "total": total, "passed": passed_count, "failed": failed_count,
            "status_counts": dict(status_counts),
            "tier_counts": {str(k): v for k, v in sorted(tier_counts.items())},
            "grade_counts": dict(grade_counts),
            "results": all_results,
        }, sys.stdout, indent=2)
        return

    # Human-readable summary
    print(f"\n{'=' * 60}")
    if total:
        print(f"Validation: {passed_count}/{total} passed ({100*passed_count/total:.0f}%)")
    else:
        print("No repos found")
    if failed_count:
        print(f"  Failed: {failed_count}")
    print()

    print("Status distribution:")
    for st, cnt in status_counts.most_common():
        print(f"  {st:<25s} {cnt:>4d}  {'#' * min(cnt, 40)}")

    print("\nTier distribution:")
    for tier, cnt in sorted(tier_counts.items()):
        print(f"  Tier {tier}: {cnt}")

    if grade_counts:
        print("\nQuality grade distribution:")
        for g, cnt in grade_counts.most_common():
            print(f"  {g:<10s} {cnt:>4d}  {'#' * min(cnt, 40)}")

    topo = [r for r in all_results if r.get("grade")]
    if topo:
        print("\nTopology summary (successful repos):")
        for r in topo:
            nids, names = r.get("node_ids", []), r.get("node_names", [])
            print(f"  {r['dir']:<40s}  events={r.get('event_count',0):<5d} "
                  f"nodes={len(nids):<3d} edges={r.get('edge_count',0):<3d} "
                  f"names={names}")

    if failure_reasons:
        print("\nCommon validation failures:")
        for reason, cnt in failure_reasons.most_common(10):
            print(f"  {reason:<30s} {cnt:>4d}")

    sys.exit(1 if failed_count else 0)


if __name__ == "__main__":
    main()
