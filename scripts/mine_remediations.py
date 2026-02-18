#!/usr/bin/env python3
"""Cross-repo remediation mining.

After ALL repos complete, reads all behavioral records, partitions repos by
finding presence x manifestation, computes defensive pattern differentials,
and produces topology-conditional remediation evidence.

Usage:
    python mine_remediations.py \
        --scan-dir <scan_output_dir> \
        --output <remediation_evidence.json> \
        [--min-n 5] [--p-threshold 0.05]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory for stratum_lab imports
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from stratum_lab.remediation import (
    partition_repos,
    compute_pattern_differential,
    compute_topology_conditional,
    compute_cross_pattern_interactions,
    compute_priority_score,
    generate_rationale,
)


def load_all_records(scan_dir: str) -> list[dict]:
    """Walk scan directory for behavioral_record.json files."""
    records: list[dict] = []

    # Check behavioral_records/ subdirectory
    br_dir = os.path.join(scan_dir, "behavioral_records")
    if os.path.isdir(br_dir):
        for f in sorted(os.listdir(br_dir)):
            if f.endswith(".json"):
                path = os.path.join(br_dir, f)
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        record = json.load(fh)
                    if isinstance(record, dict):
                        records.append(record)
                except (json.JSONDecodeError, OSError):
                    continue

    # Also check results/ subdirectories for behavioral_record.json
    results_dir = os.path.join(scan_dir, "results")
    if os.path.isdir(results_dir):
        for entry in sorted(os.listdir(results_dir)):
            br_path = os.path.join(results_dir, entry, "behavioral_record.json")
            if os.path.isfile(br_path):
                try:
                    with open(br_path, "r", encoding="utf-8") as fh:
                        record = json.load(fh)
                    if isinstance(record, dict):
                        records.append(record)
                except (json.JSONDecodeError, OSError):
                    continue

    return records


def write_json(path: str, data: dict) -> None:
    """Write dict as JSON to file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def mine_finding(
    finding_id: str, records: list, min_n: int, p_threshold: float,
) -> dict | None:
    """Run complete 4-step mining for a single finding."""
    # Step 1: Partition
    quadrants = partition_repos(finding_id, records)
    q1, q2 = quadrants["q1"], quadrants["q2"]
    q3, q4 = quadrants["q3"], quadrants["q4"]

    # Get finding severity from any record that has it
    severity = "medium"
    for rec in q1 + q2:
        for f in rec.get("failure_modes", rec.get("findings", [])):
            if isinstance(f, dict) and f.get("finding_id") == finding_id:
                severity = f.get("severity", "medium")
                break

    corpus_stats = {
        "total_repos_with_finding": len(q1) + len(q2),
        "manifestation_rate": round(len(q1) / max(1, len(q1) + len(q2)), 3),
        "q1_count": len(q1),
        "q2_count": len(q2),
        "q3_count": len(q3),
        "q4_count": len(q4),
    }

    # Step 2: Defensive pattern differential
    candidates = compute_pattern_differential(q1, q2, min_n, p_threshold)

    if not candidates:
        return {
            "finding_id": finding_id,
            "severity": severity,
            "corpus_statistics": corpus_stats,
            "remediation_candidates": [],
            "cross_pattern_interactions": [],
            "priority_ranked_remediations": [],
            "note": "insufficient data or no significant patterns found",
        }

    # Step 3: Topology-conditional enrichment
    candidates = compute_topology_conditional(finding_id, q1, q2, candidates)

    # Step 4: Cross-pattern interactions and priority scoring
    interactions = compute_cross_pattern_interactions(finding_id, records)

    priority_ranked: list[dict] = []
    for candidate in candidates:
        score = compute_priority_score(candidate, severity, interactions)
        priority_ranked.append({
            "rank": 0,
            "pattern": candidate["pattern"],
            "priority_score": score,
            "rationale": generate_rationale(candidate, interactions),
        })

    priority_ranked.sort(key=lambda x: x["priority_score"], reverse=True)
    for i, pr in enumerate(priority_ranked):
        pr["rank"] = i + 1

    return {
        "finding_id": finding_id,
        "severity": severity,
        "corpus_statistics": corpus_stats,
        "remediation_candidates": candidates,
        "cross_pattern_interactions": interactions,
        "priority_ranked_remediations": priority_ranked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-repo remediation mining.")
    parser.add_argument("--scan-dir", required=True, help="Path to top-level scan output directory.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--min-n", type=int, default=5, help="Minimum sample size per quadrant.")
    parser.add_argument("--p-threshold", type=float, default=0.05, help="P-value threshold.")
    args = parser.parse_args()

    try:
        records = load_all_records(args.scan_dir)

        if len(records) < 20:
            write_json(args.output, {
                "error": "insufficient sample size",
                "records_found": len(records),
                "minimum_required": 20,
                "findings": [],
            })
            return

        # Get all unique finding_ids across corpus
        all_finding_ids: set[str] = set()
        for rec in records:
            for finding in rec.get("failure_modes", rec.get("findings", [])):
                if isinstance(finding, dict) and finding.get("finding_id"):
                    all_finding_ids.add(finding["finding_id"])

        evidence: list[dict] = []
        for finding_id in sorted(all_finding_ids):
            result = mine_finding(finding_id, records, args.min_n, args.p_threshold)
            if result:
                evidence.append(result)

        write_json(args.output, {
            "mining_version": "1.0",
            "total_records": len(records),
            "findings_analyzed": len(all_finding_ids),
            "findings_with_evidence": sum(
                1 for e in evidence if e.get("remediation_candidates")
            ),
            "evidence": evidence,
        })

    except Exception as e:
        write_json(args.output, {
            "error": str(e),
            "findings": [],
        })


if __name__ == "__main__":
    main()
