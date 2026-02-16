#!/usr/bin/env python3
"""Pilot quality gate -- categorize pilot runs and check thresholds before full scan.

Reads each results/<hash>/ subdirectory and categorizes the run outcome.
Checks instrumentation and model failure rates against configurable thresholds.

Correct field paths:
    event["source_node"]["node_id"]   -- nested dict, NOT event["source_node_id"]
    event["event_type"]               -- dot-notation event type

Usage:
    python check_pilot_quality.py --results-dir results/ [--instr-threshold 0.20] [--model-threshold 0.15]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Category constants
# ---------------------------------------------------------------------------

CATEGORIES = [
    "SUCCESS",
    "PARTIAL_SUCCESS",
    "INSTRUMENTATION_FAILURE",
    "MODEL_FAILURE",
    "CRASH",
    "TIMEOUT",
    "DEPENDENCY_FAILURE",
    "ENTRY_POINT_FAILURE",
]

# Model-error patterns in stderr logs
MODEL_ERROR_PATTERNS = [
    "vllm", "model", "CUDA out of memory", "torch.cuda",
    "Connection refused", "503 Service Unavailable",
    "Internal Server Error", "model_not_found",
]


# ---------------------------------------------------------------------------
# Event counting
# ---------------------------------------------------------------------------

def _count_events(repo_dir: Path) -> int:
    """Count events from stratum_events.jsonl or events_run_*.jsonl."""
    events_file = repo_dir / "stratum_events.jsonl"
    if not events_file.exists():
        # Fall back to events_run_*.jsonl pattern
        candidates = sorted(repo_dir.glob("events_run_*.jsonl"))
        if candidates:
            events_file = candidates[0]
        else:
            return 0

    count = 0
    try:
        with open(events_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                    count += 1
                except json.JSONDecodeError:
                    pass
    except OSError:
        return 0
    return count


def _has_agent_interaction(repo_dir: Path) -> bool:
    """Check if events contain agent interaction (agent.task_start/end or delegation)."""
    events_file = repo_dir / "stratum_events.jsonl"
    if not events_file.exists():
        candidates = sorted(repo_dir.glob("events_run_*.jsonl"))
        if candidates:
            events_file = candidates[0]
        else:
            return False

    try:
        with open(events_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                et = event.get("event_type", "")
                if et.startswith("agent.") or et.startswith("delegation."):
                    return True
                # Also check for source_node with agent type
                source_node = event.get("source_node") or {}
                if source_node.get("node_type") == "agent":
                    return True
    except OSError:
        pass
    return False


def _stderr_has_model_errors(repo_dir: Path) -> bool:
    """Check stderr logs for vLLM / model errors."""
    for log_name in ("stderr.log", "tier2_stderr.log"):
        log_file = repo_dir / log_name
        if not log_file.exists():
            continue
        try:
            text = log_file.read_text(encoding="utf-8", errors="replace").lower()
            for pattern in MODEL_ERROR_PATTERNS:
                if pattern.lower() in text:
                    return True
        except OSError:
            pass
    return False


# ---------------------------------------------------------------------------
# Categorization
# ---------------------------------------------------------------------------

def categorize_run(repo_dir: Path) -> str:
    """Categorize a single run into one of the pilot categories."""
    # Read status from status.json (preferred) or status.txt
    status = ""
    status_json = repo_dir / "status.json"
    status_txt = repo_dir / "status.txt"

    if status_json.exists():
        try:
            with open(status_json, encoding="utf-8") as f:
                data = json.load(f)
            status = data.get("status", "")
        except (json.JSONDecodeError, OSError):
            pass

    if not status and status_txt.exists():
        try:
            status = status_txt.read_text(encoding="utf-8").strip()
        except OSError:
            pass

    # Read exit code
    exit_code = None
    exit_code_file = repo_dir / "exit_code.txt"
    if exit_code_file.exists():
        try:
            exit_code = int(exit_code_file.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            pass

    event_count = _count_events(repo_dir)

    # Entry point failure
    if status in ("NO_ENTRY_POINT", "SERVER_BASED"):
        return "ENTRY_POINT_FAILURE"

    # Dependency / import failure
    if status in ("UNRESOLVABLE_IMPORT",):
        return "DEPENDENCY_FAILURE"

    # Timeout
    if status in ("TIMEOUT_NO_EVENTS",) or exit_code == 124:
        if event_count > 0:
            return "PARTIAL_SUCCESS"
        return "TIMEOUT"

    # Check for model errors in stderr
    if event_count == 0 and _stderr_has_model_errors(repo_dir):
        return "MODEL_FAILURE"

    # Success statuses with events
    if status in ("SUCCESS", "TIER2_SUCCESS"):
        if event_count >= 10 and _has_agent_interaction(repo_dir):
            return "SUCCESS"
        if event_count > 0:
            return "PARTIAL_SUCCESS"
        # Status says success but no events -- instrumentation failure
        return "INSTRUMENTATION_FAILURE"

    if status in ("PARTIAL_SUCCESS", "TIER2_PARTIAL"):
        if event_count >= 10 and _has_agent_interaction(repo_dir):
            return "SUCCESS"
        if event_count > 0:
            return "PARTIAL_SUCCESS"
        return "INSTRUMENTATION_FAILURE"

    # Ran but zero events -- patcher didn't fire
    if status in ("NO_EVENTS",):
        return "INSTRUMENTATION_FAILURE"

    # Runtime error / crash
    if status in ("RUNTIME_ERROR", "MAX_RETRIES_EXCEEDED", "TIER2_FAILED"):
        if event_count > 0:
            return "PARTIAL_SUCCESS"
        return "CRASH"

    # Clone failure treated as dependency issue
    if status == "CLONE_FAILED":
        return "DEPENDENCY_FAILURE"

    # Fallback: use exit code
    if exit_code is not None and exit_code != 0:
        if event_count > 0:
            return "PARTIAL_SUCCESS"
        return "CRASH"

    # Unknown status with no events
    if event_count == 0:
        return "INSTRUMENTATION_FAILURE"

    # Has some events but didn't match above
    if event_count >= 10 and _has_agent_interaction(repo_dir):
        return "SUCCESS"
    return "PARTIAL_SUCCESS"


def categorize_all_runs(results_dir: Path) -> Counter:
    """Categorize every run subdirectory and return category counts."""
    categories: Counter = Counter()
    for repo_dir in sorted(results_dir.iterdir()):
        if not repo_dir.is_dir():
            continue
        # Skip directories without any status or event files
        has_status = (repo_dir / "status.txt").exists() or (repo_dir / "status.json").exists()
        has_events = (repo_dir / "stratum_events.jsonl").exists() or list(repo_dir.glob("events_run_*.jsonl"))
        if not has_status and not has_events:
            continue
        category = categorize_run(repo_dir)
        categories[category] += 1
    return categories


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_pilot_report(
    categories: Counter, total: int, passed: bool, reasons: list[str],
) -> None:
    """Print a formatted pilot quality report."""
    print()
    print("=" * 60)
    print("PILOT QUALITY REPORT")
    print("=" * 60)
    print(f"Total runs evaluated: {total}")
    print()

    # Table header
    print(f"  {'Category':<30s} {'Count':>6s} {'Pct':>7s}")
    print(f"  {'-' * 30} {'-' * 6} {'-' * 7}")

    for cat in CATEGORIES:
        count = categories.get(cat, 0)
        pct = (count / total * 100) if total > 0 else 0.0
        print(f"  {cat:<30s} {count:>6d} {pct:>6.1f}%")

    print()
    print("-" * 60)

    if passed:
        print("Verdict: PASS")
    else:
        print("Verdict: FAIL")
        for reason in reasons:
            print(f"  - {reason}")

    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

def check_pilot_quality(
    results_dir: Path,
    instr_threshold: float = 0.20,
    model_threshold: float = 0.15,
) -> bool:
    """Check if pilot run meets quality thresholds.

    Categorize each run into:
    - SUCCESS: >=10 events with agent interaction
    - PARTIAL_SUCCESS: ran, thin data (<10 events)
    - INSTRUMENTATION_FAILURE: patcher didn't fire
    - MODEL_FAILURE: vLLM returned errors
    - CRASH: Python exception
    - TIMEOUT: hit timeout
    - DEPENDENCY_FAILURE: pip install failed
    - ENTRY_POINT_FAILURE: no entry point found
    """
    categories = categorize_all_runs(results_dir)
    total = sum(categories.values())

    if total == 0:
        print("No pilot runs found.", file=sys.stderr)
        return False

    instr_rate = categories.get("INSTRUMENTATION_FAILURE", 0) / max(total, 1)
    model_rate = categories.get("MODEL_FAILURE", 0) / max(total, 1)

    passed = True
    reasons: list[str] = []

    if instr_rate > instr_threshold:
        passed = False
        reasons.append(
            f"Instrumentation failure rate {instr_rate:.0%} exceeds {instr_threshold:.0%}"
        )

    if model_rate > model_threshold:
        passed = False
        reasons.append(
            f"Model failure rate {model_rate:.0%} exceeds {model_threshold:.0%}"
        )

    print_pilot_report(categories, total, passed, reasons)
    return passed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pilot quality gate -- check failure rates before full scan.",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing per-repo result subdirectories",
    )
    parser.add_argument(
        "--instr-threshold",
        type=float,
        default=0.20,
        help="Max acceptable instrumentation failure rate (default: 0.20)",
    )
    parser.add_argument(
        "--model-threshold",
        type=float,
        default=0.15,
        help="Max acceptable model failure rate (default: 0.15)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    passed = check_pilot_quality(
        results_dir,
        instr_threshold=args.instr_threshold,
        model_threshold=args.model_threshold,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
