"""Adaptive run scheduling.

Assigns 3-7 runs per repo based on structural complexity
and the statistical needs of the preconditions it covers.
Budget: ~4500 total runs across 1000 repos (avg 4.5/repo).
"""

from __future__ import annotations

from typing import Any, Dict, List


# Rare preconditions that need every sample they can get
RARE_PRECONDITIONS = {
    "circular_delegation",
    "capability_overlap_no_priority",
    "data_store_no_schema_enforcement",
    "classification_without_validation",
}


def compute_run_count(repo: Dict[str, Any]) -> int:
    """Determine how many runs this repo needs.

    Base: 3 runs (minimum for any statistical claim)
    +1 if multi-agent (>=3 agents) — more variance expected
    +1 if has rare preconditions — need more samples for those CIs
    +1 if has classification nodes — semantic determinism needs samples
    +1 if high complexity — emergent behavior needs observation
    Cap: 7 runs (diminishing returns beyond this)
    """
    runs = 3  # Base

    agent_count = repo.get("agent_count", 1)
    preconditions = repo.get("confirmed_preconditions", [])
    complexity = repo.get("complexity", "low")

    if agent_count >= 3:
        runs += 1

    if set(preconditions) & RARE_PRECONDITIONS:
        runs += 1

    if any("classification" in p or "semantic" in p for p in preconditions):
        runs += 1

    if complexity == "high":
        runs += 1

    return min(runs, 7)


def plan_adaptive_runs(repo: Dict[str, Any], run_count: int) -> Dict[str, Any]:
    """Plan the run schedule for a repo.

    Strategy:
    - First ceil(run_count/2) runs: diverse inputs (different tasks/queries)
    - Remaining runs: repeat input 1 (for determinism measurement)
    """
    diverse_count = (run_count + 1) // 2
    repeat_count = run_count - diverse_count

    return {
        "total_runs": run_count,
        "diverse_inputs": diverse_count,
        "repeat_inputs": repeat_count,
        "repeat_input_index": 0,
        "rationale": _explain_schedule(repo, run_count),
    }


def _explain_schedule(repo: Dict[str, Any], runs: int) -> str:
    reasons: List[str] = []
    if repo.get("agent_count", 1) >= 3:
        reasons.append("multi-agent variance")
    if repo.get("complexity") == "high":
        reasons.append("high complexity")
    preconditions = repo.get("confirmed_preconditions", [])
    if any("semantic" in p or "classification" in p for p in preconditions):
        reasons.append("semantic determinism measurement")
    return f"{runs} runs" + (f" ({', '.join(reasons)})" if reasons else "")


def compute_budget(repos: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute total run budget for a set of repos."""
    run_counts = [compute_run_count(r) for r in repos]
    return {
        "total_repos": len(repos),
        "total_runs": sum(run_counts),
        "avg_runs_per_repo": sum(run_counts) / max(len(run_counts), 1),
        "by_run_count": {
            i: run_counts.count(i)
            for i in range(3, 8)
            if run_counts.count(i) > 0
        },
        "estimated_compute_hours": sum(run_counts) * 5 / 60,  # ~5 min per run
    }
