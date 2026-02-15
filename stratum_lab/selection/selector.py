"""Selection algorithm for Phase 1 repo selection.

Given a list of scored repo records, applies greedy selection with archetype
and framework diversity constraints to produce the final selection set.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from stratum_lab.config import (
    DEFAULT_MAX_PER_ARCHETYPE,
    DEFAULT_MIN_PER_ARCHETYPE,
    DEFAULT_MIN_RUNNABILITY,
    DEFAULT_SELECTION_TARGET,
    SUPPORTED_FRAMEWORKS,
)
from stratum_lab.selection.scorer import ARCHETYPES, score_repo


# ---------------------------------------------------------------------------
# Ecosystem framework proportions (approximate from GitHub survey)
# ---------------------------------------------------------------------------
# These are rough target proportions for the ecosystem; the selector uses
# them as soft guidance, not hard constraints.
ECOSYSTEM_FRAMEWORK_PROPORTIONS: dict[str, float] = {
    "crewai": 0.30,
    "langgraph": 0.25,
    "autogen": 0.15,
    "langchain": 0.20,
    "custom": 0.10,
}

MIN_FRAMEWORKS_REPRESENTED = 3


def select_repos(
    scored_repos: list[dict[str, Any]],
    *,
    target: int = DEFAULT_SELECTION_TARGET,
    min_runnability: float = DEFAULT_MIN_RUNNABILITY,
    max_per_archetype: int = DEFAULT_MAX_PER_ARCHETYPE,
    min_per_archetype: int = DEFAULT_MIN_PER_ARCHETYPE,
    control_fraction: float = 0.15,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the constrained greedy selection algorithm.

    Parameters
    ----------
    scored_repos:
        List of repo dicts as returned by ``scorer.score_repo``.
    target:
        Target number of repos to select (1500-2000).
    min_runnability:
        Minimum runnability score to be eligible.
    max_per_archetype:
        Hard cap on repos per archetype.
    min_per_archetype:
        Soft minimum for archetypes with enough eligible members.
    control_fraction:
        Fraction of target to reserve for "clean" repos with 0-1
        taxonomy preconditions (the control group for relative risk).

    Returns
    -------
    (selected, summary)
        selected is the list of selected repo dicts.
        summary is the aggregate summary dict.
    """
    # ------------------------------------------------------------------
    # Step 0: Split control group
    # ------------------------------------------------------------------
    control_count = int(target * control_fraction)
    treatment_target = target - control_count

    # Separate repos by precondition count
    def _precondition_count(r: dict[str, Any]) -> int:
        return len(r.get("taxonomy_preconditions", []))

    # ------------------------------------------------------------------
    # Step 1: Filter by runnability
    # ------------------------------------------------------------------
    eligible = [r for r in scored_repos if r["runnability_score"] >= min_runnability]

    # Select control group: repos with <=1 precondition, prioritizing 0
    clean_repos = [r for r in eligible if _precondition_count(r) <= 1]
    clean_repos.sort(key=lambda r: (_precondition_count(r), -r["runnability_score"]))
    control_group = clean_repos[:control_count]
    control_ids = {r["repo_id"] for r in control_group}

    # Tag control group
    for r in control_group:
        r["group"] = "control"

    # Remaining eligible repos for treatment selection
    eligible = [r for r in eligible if r["repo_id"] not in control_ids]
    target = treatment_target

    # ------------------------------------------------------------------
    # Step 2: Sort by total score descending
    # ------------------------------------------------------------------
    eligible.sort(key=lambda r: r["selection_score"], reverse=True)

    # ------------------------------------------------------------------
    # Step 3: Determine archetype member counts (for minimum enforcement)
    # ------------------------------------------------------------------
    archetype_eligible_counts: Counter[int] = Counter()
    for r in eligible:
        archetype_eligible_counts[r["archetype_id"]] += 1

    # Archetypes that have at least ``min_per_archetype`` eligible members
    # qualify for the minimum guarantee.
    archetypes_with_minimum = {
        a_id
        for a_id, count in archetype_eligible_counts.items()
        if count >= min_per_archetype
    }

    # ------------------------------------------------------------------
    # Step 4: Greedy selection with constraints
    # ------------------------------------------------------------------
    selected: list[dict[str, Any]] = []
    archetype_selected: Counter[int] = Counter()
    framework_selected: Counter[str] = Counter()

    # 4a — First pass: guarantee minimums for qualifying archetypes
    #       Process eligible repos in score order; for each qualifying
    #       archetype, accept up to ``min_per_archetype`` repos.
    remaining_after_minimums: list[dict[str, Any]] = []
    for repo in eligible:
        a_id = repo["archetype_id"]
        if (
            a_id in archetypes_with_minimum
            and archetype_selected[a_id] < min_per_archetype
        ):
            selected.append(repo)
            archetype_selected[a_id] += 1
            framework_selected[repo["framework"]] += 1
        else:
            remaining_after_minimums.append(repo)

    # 4b — Second pass: fill up to target from remaining eligible repos
    for repo in remaining_after_minimums:
        if len(selected) >= target:
            break

        a_id = repo["archetype_id"]
        if archetype_selected[a_id] >= max_per_archetype:
            continue

        selected.append(repo)
        archetype_selected[a_id] += 1
        framework_selected[repo["framework"]] += 1

    # ------------------------------------------------------------------
    # Step 5: Framework diversity check
    # ------------------------------------------------------------------
    #   If fewer than MIN_FRAMEWORKS_REPRESENTED are in the selection,
    #   swap in high-scoring repos from missing frameworks.
    frameworks_present = {fw for fw, c in framework_selected.items() if c > 0}
    if len(frameworks_present) < MIN_FRAMEWORKS_REPRESENTED:
        missing_frameworks = set(SUPPORTED_FRAMEWORKS) - frameworks_present
        # Index remaining eligible (not yet selected) by framework
        selected_ids = {r["repo_id"] for r in selected}
        by_missing_fw: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for repo in eligible:
            if repo["repo_id"] not in selected_ids and repo["framework"] in missing_frameworks:
                by_missing_fw[repo["framework"]].append(repo)

        for fw in missing_frameworks:
            if len(frameworks_present) >= MIN_FRAMEWORKS_REPRESENTED:
                break
            candidates = by_missing_fw.get(fw, [])
            if not candidates:
                continue

            # Replace the lowest-scored selected repo (from the most
            # over-represented framework) with the best candidate from
            # the missing framework.
            best_candidate = candidates[0]  # already sorted by score

            # Find the worst repo in the most common framework
            most_common_fw = framework_selected.most_common(1)[0][0]
            worst_in_common = None
            worst_idx = -1
            for idx in range(len(selected) - 1, -1, -1):
                if selected[idx]["framework"] == most_common_fw:
                    if worst_in_common is None or selected[idx]["selection_score"] < worst_in_common["selection_score"]:
                        worst_in_common = selected[idx]
                        worst_idx = idx
                        break  # last in list is already lowest-scored

            if worst_in_common is not None and worst_idx >= 0:
                # Swap
                archetype_selected[worst_in_common["archetype_id"]] -= 1
                framework_selected[worst_in_common["framework"]] -= 1
                selected[worst_idx] = best_candidate
                archetype_selected[best_candidate["archetype_id"]] += 1
                framework_selected[best_candidate["framework"]] += 1
                frameworks_present.add(fw)

    # ------------------------------------------------------------------
    # Step 6: Framework proportion soft-balancing
    # ------------------------------------------------------------------
    #   If a framework exceeds 2x its target proportion and the selection
    #   is above target, trim excess from that framework (remove the
    #   lowest-scored repos) and backfill from under-represented frameworks.
    if len(selected) > target:
        total_selected = len(selected)
        for fw, proportion in ECOSYSTEM_FRAMEWORK_PROPORTIONS.items():
            max_allowed = int(total_selected * proportion * 2)
            if framework_selected[fw] > max_allowed:
                # Identify the excess repos (lowest scores first)
                fw_repos = [(i, r) for i, r in enumerate(selected) if r["framework"] == fw]
                fw_repos.sort(key=lambda x: x[1]["selection_score"])
                excess = framework_selected[fw] - max_allowed
                indices_to_remove = [i for i, _ in fw_repos[:excess]]
                for idx in sorted(indices_to_remove, reverse=True):
                    removed = selected.pop(idx)
                    archetype_selected[removed["archetype_id"]] -= 1
                    framework_selected[removed["framework"]] -= 1

    # Final re-sort by score
    selected.sort(key=lambda r: r["selection_score"], reverse=True)

    # Tag treatment group
    for r in selected:
        if "group" not in r:
            r["group"] = "treatment"

    # Merge control + treatment
    selected = control_group + selected

    # ------------------------------------------------------------------
    # Build summary
    # ------------------------------------------------------------------
    summary = _build_summary(selected, [r for r in scored_repos if r["runnability_score"] >= min_runnability], scored_repos)
    summary["control_count"] = len(control_group)
    summary["treatment_count"] = len(selected) - len(control_group)
    return selected, summary


def score_and_select(
    raw_repos: list[dict[str, Any]],
    *,
    target: int = DEFAULT_SELECTION_TARGET,
    min_runnability: float = DEFAULT_MIN_RUNNABILITY,
    max_per_archetype: int = DEFAULT_MAX_PER_ARCHETYPE,
    min_per_archetype: int = DEFAULT_MIN_PER_ARCHETYPE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """End-to-end: score raw structural scan dicts, then select.

    This is the primary entry point when you have raw JSONs from stratum-cli.

    The archetype rarity dimension is computed iteratively: repos are first
    scored with zero archetype counts, sorted, and then re-scored as the
    greedy selector picks them.  For simplicity (and because the rarity
    bonus is a soft signal, not the dominant dimension), we perform a
    two-pass approach:

    1. Score all repos with archetype_counts = {} (maximum rarity bonus).
    2. Feed scored repos into ``select_repos`` which greedily selects.
    3. Re-score the selected repos with true archetype counts so the
       final ``selection_score`` and ``archetype_rarity`` are accurate.
    """
    # Pass 1: initial scoring (uniform rarity bonus)
    archetype_counts: Counter[int] = Counter()
    scored = [score_repo(repo, archetype_counts, target) for repo in raw_repos]

    # Pass 2: select
    selected, _ = select_repos(
        scored,
        target=target,
        min_runnability=min_runnability,
        max_per_archetype=max_per_archetype,
        min_per_archetype=min_per_archetype,
    )

    # Pass 3: re-score selected repos with actual archetype counts so that
    #          the archetype_rarity values reflect the true selection state.
    final_archetype_counts: Counter[int] = Counter()
    for r in selected:
        final_archetype_counts[r["archetype_id"]] += 1

    # Build a lookup from repo_id to original raw repo for re-scoring
    raw_by_id = {r.get("repo_id", ""): r for r in raw_repos}
    rescored: list[dict[str, Any]] = []
    for r in selected:
        raw = raw_by_id.get(r["repo_id"])
        if raw is not None:
            new_r = score_repo(raw, final_archetype_counts, target)
            # Preserve group tag from selection phase
            if "group" in r:
                new_r["group"] = r["group"]
            rescored.append(new_r)
        else:
            rescored.append(r)

    rescored.sort(key=lambda r: r["selection_score"], reverse=True)
    summary = _build_summary(rescored, scored, scored)
    # Add control/treatment counts
    control_repos = [r for r in rescored if r.get("group") == "control"]
    summary["control_count"] = len(control_repos)
    summary["treatment_count"] = len(rescored) - len(control_repos)
    return rescored, summary


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _build_summary(
    selected: list[dict[str, Any]],
    eligible: list[dict[str, Any]],
    total_scored: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the aggregate summary dict."""
    by_framework: Counter[str] = Counter()
    by_archetype: Counter[str] = Counter()
    structural_values: list[float] = []
    agent_counts: list[int] = []
    all_preconditions: set[str] = set()
    runnability_scores: list[float] = []

    for r in selected:
        by_framework[r["framework"]] += 1
        by_archetype[r["archetype_name"]] += 1
        structural_values.append(r["structural_value"])
        agent_counts.append(r["agent_count"])
        all_preconditions.update(r.get("taxonomy_preconditions", []))
        runnability_scores.append(r["runnability_score"])

    n = max(len(selected), 1)
    return {
        "total_scanned": len(total_scored),
        "total_eligible": len(eligible),
        "total_selected": len(selected),
        "by_framework": dict(by_framework.most_common()),
        "by_archetype": dict(by_archetype.most_common()),
        "avg_structural_value": round(sum(structural_values) / n, 2),
        "avg_runnability_score": round(sum(runnability_scores) / n, 2),
        "avg_agent_count": round(sum(agent_counts) / n, 2),
        "total_taxonomy_preconditions_covered": len(all_preconditions),
        "frameworks_represented": len(by_framework),
        "archetypes_represented": len(by_archetype),
    }
