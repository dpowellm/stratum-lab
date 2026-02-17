"""Validate behavioral trace export readiness.

Checks structural integrity of behavioral_traces.jsonl before
ingestion by stratum-graph.

Usage:
    python -m stratum_lab.validate_export --export-path behavioral_traces.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    valid: bool = True
    total_repos: int = 0
    success_count: int = 0
    partial_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    agents_per_repo: list[int] = field(default_factory=list)
    output_lengths: list[int] = field(default_factory=list)
    frameworks: dict[str, int] = field(default_factory=lambda: defaultdict(int))


REQUIRED_FIELDS = {"repo_id", "repo_hash", "framework", "scan_status", "runs"}


def validate_export(export_path: Path) -> ValidationResult:
    """Validate a behavioral_traces.jsonl file."""
    result = ValidationResult()
    path = Path(export_path)

    if not path.exists():
        result.valid = False
        result.errors.append(f"File does not exist: {path}")
        return result

    seen_repo_ids: set[str] = set()
    line_num = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line_num += 1
            line = line.strip()
            if not line:
                continue

            # Check valid JSON
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                result.valid = False
                result.errors.append(f"Line {line_num}: invalid JSON — {e}")
                continue

            if not isinstance(rec, dict):
                result.valid = False
                result.errors.append(f"Line {line_num}: not a JSON object")
                continue

            result.total_repos += 1

            # Required fields
            missing = REQUIRED_FIELDS - set(rec.keys())
            if missing:
                result.valid = False
                result.errors.append(
                    f"Line {line_num}: missing required fields: {missing}"
                )

            # repo_id format
            repo_id = rec.get("repo_id", "")
            if not repo_id.startswith("https://github.com/"):
                result.valid = False
                result.errors.append(
                    f"Line {line_num}: invalid repo_id (not a GitHub URL): {repo_id!r}"
                )

            # Duplicate check
            if repo_id in seen_repo_ids:
                result.valid = False
                result.errors.append(
                    f"Line {line_num}: duplicate repo_id: {repo_id}"
                )
            seen_repo_ids.add(repo_id)

            # Status
            status = rec.get("scan_status", "")
            if status == "SUCCESS":
                result.success_count += 1
            elif status == "PARTIAL_SUCCESS":
                result.partial_count += 1

            # Framework
            fw = rec.get("framework", "unknown")
            result.frameworks[fw] += 1

            # Runs
            runs = rec.get("runs", [])
            if not isinstance(runs, list) or not runs:
                result.warnings.append(
                    f"Line {line_num}: no runs for {repo_id}"
                )
                continue

            # Agents and output checks
            all_agent_names: set[str] = set()
            has_nonempty_output = False
            run_agent_count = 0

            for run in runs:
                agents = run.get("agents", [])
                run_agent_count += len(agents)
                for agent in agents:
                    aname = agent.get("agent_name", "")
                    all_agent_names.add(aname)
                    for task in agent.get("tasks", []):
                        out = task.get("output_text", "")
                        if out:
                            has_nonempty_output = True
                            result.output_lengths.append(len(out))

                # Delegation chain validation
                for dc in run.get("delegation_chains", []):
                    up = dc.get("upstream_agent", "")
                    down = dc.get("downstream_agent", "")
                    if up and up not in all_agent_names:
                        result.valid = False
                        result.errors.append(
                            f"Line {line_num}: delegation references "
                            f"non-existent upstream agent {up!r}"
                        )
                    if down and down not in all_agent_names:
                        result.valid = False
                        result.errors.append(
                            f"Line {line_num}: delegation references "
                            f"non-existent downstream agent {down!r}"
                        )

            result.agents_per_repo.append(run_agent_count)

            if not has_nonempty_output:
                result.warnings.append(
                    f"Line {line_num}: {repo_id} has no non-empty output_text"
                )

    return result


def print_result(result: ValidationResult) -> None:
    """Print validation result to stdout."""
    status = "PASS" if result.valid else "FAIL"
    print(f"\nValidation: {status}")
    print(f"  Total repos:    {result.total_repos}")
    print(f"  SUCCESS:        {result.success_count}")
    print(f"  PARTIAL:        {result.partial_count}")

    if result.agents_per_repo:
        avg = sum(result.agents_per_repo) / len(result.agents_per_repo)
        print(f"  Agents/repo:    mean={avg:.1f}, "
              f"min={min(result.agents_per_repo)}, "
              f"max={max(result.agents_per_repo)}")

    if result.output_lengths:
        avg = sum(result.output_lengths) / len(result.output_lengths)
        print(f"  Output length:  mean={avg:.0f} chars, "
              f"min={min(result.output_lengths)}, "
              f"max={max(result.output_lengths)}")

    if result.frameworks:
        print(f"  Frameworks:")
        for fw, count in sorted(result.frameworks.items(), key=lambda x: -x[1]):
            print(f"    {fw}: {count}")

    if result.errors:
        print(f"\n  Errors ({len(result.errors)}):")
        for err in result.errors[:20]:
            print(f"    - {err}")
        if len(result.errors) > 20:
            print(f"    ... and {len(result.errors) - 20} more")

    if result.warnings:
        print(f"\n  Warnings ({len(result.warnings)}):")
        for w in result.warnings[:10]:
            print(f"    - {w}")
        if len(result.warnings) > 10:
            print(f"    ... and {len(result.warnings) - 10} more")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        prog="stratum-lab-validate-export",
        description="Validate behavioral trace export",
    )
    parser.add_argument(
        "--export-path",
        required=True,
        help="Path to behavioral_traces.jsonl",
    )
    args = parser.parse_args()
    result = validate_export(Path(args.export_path))
    print_result(result)
    raise SystemExit(0 if result.valid else 1)


if __name__ == "__main__":
    main()
