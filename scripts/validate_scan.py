#!/usr/bin/env python3
"""validate_scan.py — Post-scan data quality validation for stratum-lab mass scans.

Walks every repo directory in the results directory, validates metadata and
events, and produces a data quality report with a 0-100 score.

Usage:
    python3 scripts/validate_scan.py ~/scan_output/results
    python3 scripts/validate_scan.py ~/scan_output/results --output ~/scan_output/validation_report.json
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


def classify_behavioral(events_path: str) -> str:
    """Classify behavioral status from an events JSONL file."""
    has_agent = False
    has_llm = False
    try:
        with open(events_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                etype = evt.get("event_type", "")
                if etype.startswith("agent."):
                    has_agent = True
                if etype.startswith("llm."):
                    has_llm = True
    except (OSError, IOError):
        return "NO_BEHAVIORAL_DATA"

    if has_agent and has_llm:
        return "FULL_BEHAVIORAL"
    elif has_llm:
        return "LLM_ONLY"
    return "NO_BEHAVIORAL_DATA"


def validate_events_file(events_path: str) -> dict:
    """Validate a single events JSONL file and return stats."""
    stats = {
        "event_count": 0,
        "corrupt_lines": 0,
        "double_prefix": 0,
        "llm_calls": 0,
        "agent_starts": 0,
        "has_system_prompt": 0,
        "has_user_message": 0,
        "has_output": 0,
        "llm_start_count": 0,
        "llm_end_count": 0,
    }
    try:
        with open(events_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                stats["event_count"] += 1
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    stats["corrupt_lines"] += 1
                    continue

                if "openai/openai/" in line:
                    stats["double_prefix"] += 1

                etype = evt.get("event_type", "")
                if etype == "llm.call_start":
                    stats["llm_calls"] += 1
                    stats["llm_start_count"] += 1
                    payload = evt.get("payload", {})
                    if payload.get("system_prompt_preview"):
                        stats["has_system_prompt"] += 1
                    if payload.get("last_user_message_preview"):
                        stats["has_user_message"] += 1
                elif etype == "llm.call_end":
                    stats["llm_end_count"] += 1
                    payload = evt.get("payload", {})
                    if payload.get("output_preview"):
                        stats["has_output"] += 1
                if etype.startswith("agent."):
                    stats["agent_starts"] += 1
    except (OSError, IOError):
        pass
    return stats


def validate_scan(results_dir: str) -> dict:
    """Walk the results directory and produce a full validation report."""
    results_path = Path(results_dir)
    if not results_path.is_dir():
        print(f"ERROR: {results_dir} is not a directory", file=sys.stderr)
        sys.exit(2)

    repos_total = 0
    repos_with_events = 0
    repos_with_metadata = 0
    status_dist = Counter()
    tier_dist = Counter()
    framework_dist = Counter()
    behavioral_dist = Counter()
    total_events = 0
    total_llm_calls = 0
    total_agent_starts = 0
    double_prefix_count = 0
    corrupt_event_lines = 0
    scan_ids = set()
    io_has_system_prompt = 0
    io_has_user_message = 0
    io_has_output = 0
    io_llm_start_count = 0
    io_llm_end_count = 0
    recovered_repos = 0
    issues = []

    for entry in sorted(results_path.iterdir()):
        if not entry.is_dir():
            continue
        repos_total += 1
        repo_hash = entry.name

        # Check metadata
        meta_path = entry / "run_metadata_1.json"
        if meta_path.is_file():
            repos_with_metadata += 1
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                status = meta.get("status", "UNKNOWN")
                status_dist[status] += 1
                tier = meta.get("tier_detail", meta.get("tier", "unknown"))
                tier_dist[str(tier)] += 1
                fw = meta.get("framework_detected", "unknown")
                framework_dist[fw] += 1
                sid = meta.get("scan_id", "")
                if sid:
                    scan_ids.add(sid)
                else:
                    issues.append(f"{repo_hash}: missing scan_id in metadata")
                if meta.get("recovered") or status == "RECOVERED_FROM_TIMEOUT":
                    recovered_repos += 1
            except (json.JSONDecodeError, OSError) as e:
                issues.append(f"{repo_hash}: corrupt metadata: {e}")
        else:
            issues.append(f"{repo_hash}: missing run_metadata_1.json")

        # Check events
        events_files = sorted(entry.glob("events_run_*.jsonl"))
        repo_has_events = False
        for ef in events_files:
            if ef.stat().st_size > 0:
                repo_has_events = True
                estats = validate_events_file(str(ef))
                total_events += estats["event_count"]
                total_llm_calls += estats["llm_calls"]
                total_agent_starts += estats["agent_starts"]
                double_prefix_count += estats["double_prefix"]
                corrupt_event_lines += estats["corrupt_lines"]
                io_has_system_prompt += estats["has_system_prompt"]
                io_has_user_message += estats["has_user_message"]
                io_has_output += estats["has_output"]
                io_llm_start_count += estats["llm_start_count"]
                io_llm_end_count += estats["llm_end_count"]

        if repo_has_events:
            repos_with_events += 1
            beh = classify_behavioral(str(events_files[0]))
            behavioral_dist[beh] += 1
        else:
            behavioral_dist["NO_BEHAVIORAL_DATA"] += 1

    # Compute data quality score
    if repos_total > 0:
        score_events = (repos_with_events / repos_total) * 30
        score_meta = (repos_with_metadata / repos_total) * 20
        behavioral_repos = behavioral_dist.get("FULL_BEHAVIORAL", 0) + behavioral_dist.get("LLM_ONLY", 0)
        score_behavioral = (behavioral_repos / repos_total) * 30
    else:
        score_events = 0
        score_meta = 0
        score_behavioral = 0

    score_prefix = 10 if double_prefix_count == 0 else 0
    score_scan_id = 10 if len(scan_ids) == 1 else 0
    data_quality_score = round(score_events + score_meta + score_behavioral + score_prefix + score_scan_id, 1)

    report = {
        "repos_total": repos_total,
        "repos_with_events": repos_with_events,
        "repos_with_metadata": repos_with_metadata,
        "status_distribution": dict(status_dist),
        "tier_distribution": dict(tier_dist),
        "framework_distribution": dict(framework_dist),
        "behavioral_distribution": dict(behavioral_dist),
        "total_events": total_events,
        "total_llm_calls": total_llm_calls,
        "total_agent_starts": total_agent_starts,
        "double_prefix_count": double_prefix_count,
        "corrupt_event_lines": corrupt_event_lines,
        "scan_ids": sorted(scan_ids),
        "io_capture": {
            "has_system_prompt": io_has_system_prompt,
            "has_user_message": io_has_user_message,
            "has_output": io_has_output,
            "llm_start_count": io_llm_start_count,
            "llm_end_count": io_llm_end_count,
        },
        "recovered_repos": recovered_repos,
        "issues": issues,
        "data_quality_score": data_quality_score,
    }
    return report


def print_summary(report: dict) -> None:
    """Print a human-readable summary table."""
    print("=" * 60)
    print("STRATUM SCAN VALIDATION REPORT")
    print("=" * 60)
    print()
    print(f"  Repos total:          {report['repos_total']}")
    print(f"  Repos with events:    {report['repos_with_events']}")
    print(f"  Repos with metadata:  {report['repos_with_metadata']}")
    print(f"  Recovered repos:      {report['recovered_repos']}")
    print()
    print("  Status distribution:")
    for k, v in sorted(report["status_distribution"].items()):
        print(f"    {k:30s} {v}")
    print()
    print("  Tier distribution:")
    for k, v in sorted(report["tier_distribution"].items()):
        print(f"    Tier {k:25s} {v}")
    print()
    print("  Framework distribution:")
    for k, v in sorted(report["framework_distribution"].items(), key=lambda x: -x[1]):
        print(f"    {k:30s} {v}")
    print()
    print("  Behavioral classification:")
    for k, v in sorted(report["behavioral_distribution"].items()):
        print(f"    {k:30s} {v}")
    print()
    print(f"  Total events:         {report['total_events']}")
    print(f"  Total LLM calls:      {report['total_llm_calls']}")
    print(f"  Total agent starts:   {report['total_agent_starts']}")
    print(f"  Double-prefix bugs:   {report['double_prefix_count']}")
    print(f"  Corrupt event lines:  {report['corrupt_event_lines']}")
    print(f"  Scan IDs found:       {report['scan_ids']}")
    print()
    io = report["io_capture"]
    print("  I/O capture:")
    print(f"    system_prompt:      {io['has_system_prompt']} / {io['llm_start_count']}")
    print(f"    user_message:       {io['has_user_message']} / {io['llm_start_count']}")
    print(f"    output:             {io['has_output']} / {io['llm_end_count']}")
    print()
    if report["issues"]:
        print(f"  Issues ({len(report['issues'])}):")
        for issue in report["issues"][:20]:
            print(f"    - {issue}")
        if len(report["issues"]) > 20:
            print(f"    ... and {len(report['issues']) - 20} more")
        print()
    print(f"  DATA QUALITY SCORE: {report['data_quality_score']} / 100")
    print()
    if report["data_quality_score"] >= 50:
        print("  RESULT: PASS")
    else:
        print("  RESULT: FAIL (score < 50)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Validate stratum-lab mass scan results")
    parser.add_argument("results_dir", help="Path to the results directory (e.g., ~/scan_output/results)")
    parser.add_argument("--output", "-o", help="Write JSON report to this file")
    args = parser.parse_args()

    report = validate_scan(args.results_dir)
    print_summary(report)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nJSON report written to: {args.output}")

    sys.exit(0 if report["data_quality_score"] >= 50 else 1)


if __name__ == "__main__":
    main()
