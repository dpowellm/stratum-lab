"""CLI entry point for the LLM-as-judge pipeline.

Usage:
    python -m stratum_lab.judge --results-dir results/full_scan/results --output-dir results/full_scan
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from stratum_lab.judge.config import (
    ANTHROPIC_API_KEY,
    BATCH_SIZE,
    COST_ABORT_THRESHOLD_USD,
    VALID_STATUSES,
    CostTracker,
)
from stratum_lab.judge.event_loader import load_execution_context, ExecutionContext
from stratum_lab.judge.runner import (
    build_batch_requests,
    submit_batch,
    poll_batch,
    parse_judge_result,
    build_retry_request,
)
from stratum_lab.judge.aggregator import (
    write_judge_results,
    build_summary,
    write_summary,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_runs(
    results_dir: Path,
    filter_status: set[str] | None = None,
    max_repos: int | None = None,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Discover successful runs under results_dir.

    Each run is a directory containing run_metadata_N.json and events_run_N.jsonl.
    Returns list of dicts with keys: repo_dir, metadata_path, events_path,
    status, repo_url, run_number.
    """
    if filter_status is None:
        filter_status = VALID_STATUSES

    runs: list[dict[str, Any]] = []
    if not results_dir.is_dir():
        logger.warning("Results directory does not exist: %s", results_dir)
        return runs

    for repo_dir in sorted(results_dir.iterdir()):
        if not repo_dir.is_dir():
            continue

        # Find run metadata files
        for meta_path in sorted(repo_dir.glob("run_metadata_*.json")):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            status = meta.get("status", meta.get("execution_status", ""))
            if status not in filter_status:
                continue

            # Extract run number from filename
            stem = meta_path.stem  # run_metadata_1
            run_num_str = stem.rsplit("_", 1)[-1]
            try:
                run_number = int(run_num_str)
            except ValueError:
                run_number = 1

            events_path = repo_dir / f"events_run_{run_number}.jsonl"
            if not events_path.exists():
                continue

            # Resume: skip if judge results already exist
            judge_path = repo_dir / f"judge_results_{run_number}.jsonl"
            if resume and judge_path.exists():
                continue

            runs.append({
                "repo_dir": repo_dir,
                "metadata_path": meta_path,
                "events_path": events_path,
                "status": status,
                "repo_url": meta.get("repo_url", meta.get("repo_id", "")),
                "run_number": run_number,
                "judge_output_path": judge_path,
            })

        if max_repos and len(runs) >= max_repos:
            break

    return runs[:max_repos] if max_repos else runs


def load_contexts(runs: list[dict[str, Any]]) -> list[ExecutionContext]:
    """Load ExecutionContext for each discovered run."""
    contexts: list[ExecutionContext] = []
    for run in runs:
        ctx = load_execution_context(
            run["events_path"],
            status=run["status"],
            repo_url=run["repo_url"],
        )
        # Attach run metadata for later routing
        ctx._run_info = run  # type: ignore[attr-defined]
        contexts.append(ctx)
    return contexts


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full judge pipeline."""
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    api_key = args.api_key or ANTHROPIC_API_KEY
    if api_key == "sk-ant-REPLACE-WITH-YOUR-KEY" and not args.dry_run:
        print("ERROR: No API key provided. Use --api-key or set ANTHROPIC_API_KEY.")
        sys.exit(1)

    filter_status = None
    if args.filter_status:
        filter_status = set(args.filter_status.split(","))

    # Step 1: Discover runs
    print(f"Discovering runs in {results_dir}...")
    runs = discover_runs(
        results_dir,
        filter_status=filter_status,
        max_repos=args.max_repos,
        resume=args.resume,
    )
    print(f"  Found {len(runs)} runs to evaluate")

    if not runs:
        print("No runs found. Exiting.")
        return

    # Step 2: Load execution contexts
    print("Loading execution contexts...")
    contexts = load_contexts(runs)
    total_agents = sum(len(c.agents) for c in contexts)
    print(f"  {total_agents} agent executions across {len(contexts)} repos")

    # Step 3: Build batch requests
    print("Building judge requests...")
    requests = build_batch_requests(contexts)
    print(f"  {len(requests)} judge calls to submit")

    # Step 4: Cost estimate
    cost_tracker = CostTracker()
    estimated_cost = cost_tracker.estimate_cost(len(requests))
    print(f"  Estimated cost: ${estimated_cost:.2f}")

    if estimated_cost > COST_ABORT_THRESHOLD_USD:
        print(f"ERROR: Estimated cost ${estimated_cost:.2f} exceeds "
              f"threshold ${COST_ABORT_THRESHOLD_USD:.2f}. Aborting.")
        sys.exit(1)

    # Dry run: stop here
    if args.dry_run:
        print("\n--- DRY RUN ---")
        print(f"Repos: {len(contexts)}")
        print(f"Agents: {total_agents}")
        print(f"Judge calls: {len(requests)}")
        print(f"Estimated cost: ${estimated_cost:.2f}")
        # Show per-criterion breakdown
        crit_counts: dict[str, int] = {}
        for req in requests:
            crit = req["custom_id"].rsplit("|", 1)[-1]
            crit_counts[crit] = crit_counts.get(crit, 0) + 1
        for crit, count in sorted(crit_counts.items()):
            print(f"  {crit}: {count}")
        return

    # Step 5: Submit batch
    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Split into batches if needed
    all_results: list[dict[str, Any]] = []
    for batch_start in range(0, len(requests), BATCH_SIZE):
        batch_reqs = requests[batch_start:batch_start + BATCH_SIZE]
        print(f"\nSubmitting batch {batch_start // BATCH_SIZE + 1} "
              f"({len(batch_reqs)} requests)...")

        batch_id = submit_batch(batch_reqs, client)
        print(f"  Batch ID: {batch_id}")

        # Step 6: Poll for completion
        print("Polling for batch completion...")
        raw_results = poll_batch(batch_id, client)

        # Parse results
        retry_needed: list[tuple[dict, dict]] = []  # (original_req, error_result)
        for raw in raw_results:
            parsed = parse_judge_result(raw, cost_tracker)
            if parsed and parsed.get("judge_error") == "malformed_json":
                # Find original request for retry
                for req in batch_reqs:
                    if req["custom_id"] == raw.custom_id:
                        retry_needed.append((req, parsed))
                        break
            elif parsed:
                all_results.append(parsed)

        # Retry malformed JSON (one attempt)
        if retry_needed:
            print(f"  Retrying {len(retry_needed)} malformed responses...")
            retry_reqs = [build_retry_request(r[0]) for r in retry_needed]
            retry_batch_id = submit_batch(retry_reqs, client)
            retry_raw = poll_batch(retry_batch_id, client)
            for raw in retry_raw:
                parsed = parse_judge_result(raw, cost_tracker)
                if parsed:
                    all_results.append(parsed)

    print(f"\nTotal results: {len(all_results)}")
    print(f"Total cost: ${cost_tracker.cost_usd:.2f}")

    # Step 7: Write per-repo results
    print("Writing per-repo judge results...")
    # Group results by repo_url and route to repo dirs
    repo_results: dict[str, list[dict]] = {}
    for rec in all_results:
        repo_results.setdefault(rec.get("repo_url", ""), []).append(rec)

    # Write using run info from contexts
    for ctx in contexts:
        run_info = getattr(ctx, "_run_info", {})
        judge_path = run_info.get("judge_output_path")
        if not judge_path:
            continue
        repo_recs = repo_results.get(ctx.repo_url, [])
        if repo_recs:
            # Enrich with framework
            for rec in repo_recs:
                rec["framework"] = ctx.framework
            write_judge_results(repo_recs, judge_path, run_info.get("run_number", 1))

    # Step 8: Build and write summary
    print("Building cross-repo summary...")
    summary = build_summary(all_results, cost_tracker)
    summary_path = output_dir / "judge_summary.json"
    write_summary(summary, summary_path)

    print(f"\nDone! Summary at {summary_path}")
    for headline in summary.get("headline_findings", []):
        print(f"  - {headline}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stratum-lab-judge",
        description="LLM-as-judge post-processing for stratum-lab behavioral scans",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Path to scan results directory (e.g. results/full_scan/results)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/full_scan",
        help="Where to write judge_summary.json (default: results/full_scan)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--filter-status",
        default=None,
        help="Comma-separated statuses to evaluate (default: SUCCESS,PARTIAL_SUCCESS)",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=None,
        help="Cap on number of repos to evaluate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print estimated call count and cost without submitting",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Max requests per batch (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip repos that already have judge_results_N.jsonl",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
