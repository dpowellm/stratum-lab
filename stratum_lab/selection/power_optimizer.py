"""Statistical power-aware selection.

Instead of scoring repos independently and taking top-N,
this optimizer selects repos to maximize statistical power
across ALL claims the dataset needs to make.

Key insight: the marginal value of a repo depends on what
we've already selected. The 50th crewai sequential pipeline
adds almost nothing. The 8th circular_delegation repo might
cut a confidence interval in half.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple


# ===== DATASET COMPOSITION TARGETS =====

PRECONDITION_TARGETS: Dict[str, Dict[str, int]] = {
    # High-priority preconditions (core thesis): need n >= 80
    "unvalidated_semantic_chain": {"min_treatment": 80, "min_control": 30},
    "shared_state_no_arbitration": {"min_treatment": 60, "min_control": 30},
    "no_timeout_on_delegation": {"min_treatment": 60, "min_control": 30},
    "unbounded_delegation_depth": {"min_treatment": 50, "min_control": 30},
    "no_output_validation": {"min_treatment": 50, "min_control": 30},

    # Medium-priority: need n >= 40
    "no_error_propagation_strategy": {"min_treatment": 40, "min_control": 20},
    "single_point_of_failure": {"min_treatment": 40, "min_control": 20},
    "no_fallback_for_external": {"min_treatment": 40, "min_control": 20},
    "shared_tool_no_concurrency_control": {"min_treatment": 40, "min_control": 20},
    "trust_boundary_no_sanitization": {"min_treatment": 40, "min_control": 20},
    "classification_without_validation": {"min_treatment": 40, "min_control": 20},
    "unhandled_tool_failure": {"min_treatment": 40, "min_control": 20},

    # Lower-priority (rarer patterns): take what we can get
    "circular_delegation": {"min_treatment": 15, "min_control": 10},
    "capability_overlap_no_priority": {"min_treatment": 20, "min_control": 10},
    "no_rate_limiting": {"min_treatment": 20, "min_control": 10},
    "implicit_ordering_dependency": {"min_treatment": 30, "min_control": 15},
    "data_store_no_schema_enforcement": {"min_treatment": 15, "min_control": 10},
}

FRAMEWORK_TARGETS: Dict[str, Dict[str, int]] = {
    "crewai": {"min": 150, "ideal": 250},
    "autogen": {"min": 100, "ideal": 200},
    "langgraph": {"min": 100, "ideal": 200},
    "langchain": {"min": 100, "ideal": 200},
    "custom": {"min": 50, "ideal": 100},
}

ARCHETYPE_TARGETS: Dict[str, Dict[str, int]] = {
    "hierarchical_delegation": {"min": 50, "ideal": 100},
    "sequential_pipeline": {"min": 50, "ideal": 100},
    "hub_and_spoke_shared_state": {"min": 40, "ideal": 80},
    "supervisor_worker": {"min": 40, "ideal": 80},
    "reflection_loop": {"min": 30, "ideal": 60},
    "debate_consensus": {"min": 20, "ideal": 50},
    "parallel_fan_out": {"min": 20, "ideal": 50},
    "guardrail_gated_pipeline": {"min": 30, "ideal": 60},
    "human_in_the_loop": {"min": 15, "ideal": 40},
    "market_auction": {"min": 10, "ideal": 30},
    "blackboard_architecture": {"min": 10, "ideal": 30},
    "single_agent_tool_use": {"min": 30, "ideal": 60},
}

COMPLEXITY_TARGETS: Dict[str, Dict[str, int]] = {
    "low": {"min": 100, "ideal": 150},
    "medium": {"min": 300, "ideal": 400},
    "high": {"min": 300, "ideal": 400},
}

CONTROL_FRACTION = 0.15


def compute_marginal_value(
    repo: Dict[str, Any],
    current_selection: List[Dict[str, Any]],
    current_counts: Dict[str, Any],
) -> float:
    """Compute the marginal value of adding this repo to the current selection.

    Value = how much does this repo improve the dataset's ability to
    make statistically rigorous claims?
    """
    value = 0.0

    # --- Precondition coverage value ---
    preconditions = set(repo.get("confirmed_preconditions", []))
    precondition_count = len(preconditions)

    for pc in preconditions:
        if pc in PRECONDITION_TARGETS:
            target = PRECONDITION_TARGETS[pc]["min_treatment"]
            current = current_counts.get(f"pc_treatment_{pc}", 0)

            if current < target:
                gap_fraction = 1.0 - (current / target)
                value += 10.0 * gap_fraction
            else:
                value += 0.5

    # Control group value
    if precondition_count <= 1:
        control_count = current_counts.get("control_total", 0)
        control_target = int(len(current_selection) * CONTROL_FRACTION) + 15
        if control_count < control_target:
            gap = 1.0 - (control_count / max(control_target, 1))
            value += 15.0 * gap

    # --- Framework coverage value ---
    framework = repo.get("primary_framework", "custom")
    if framework in FRAMEWORK_TARGETS:
        target = FRAMEWORK_TARGETS[framework]["min"]
        current = current_counts.get(f"fw_{framework}", 0)
        if current < target:
            gap = 1.0 - (current / target)
            value += 8.0 * gap
        else:
            value += 0.3

    # --- Archetype coverage value ---
    archetype = repo.get("archetype", "")
    if archetype in ARCHETYPE_TARGETS:
        target = ARCHETYPE_TARGETS[archetype]["min"]
        current = current_counts.get(f"arch_{archetype}", 0)
        if current < target:
            gap = 1.0 - (current / target)
            value += 6.0 * gap
        else:
            value += 0.2

    # --- Complexity balance ---
    complexity = repo.get("complexity", "medium")
    if complexity in COMPLEXITY_TARGETS:
        target = COMPLEXITY_TARGETS[complexity]["min"]
        current = current_counts.get(f"cx_{complexity}", 0)
        if current < target:
            gap = 1.0 - (current / target)
            value += 4.0 * gap

    # --- Structural uniqueness bonus ---
    topo_hash = repo.get("topology_hash", "")
    if topo_hash and topo_hash not in current_counts.get("seen_topologies", set()):
        value += 3.0

    # --- Enterprise relevance bonus ---
    if repo.get("enterprise_signals", 0) >= 3:
        value += 2.0

    # --- Penalize extremely long install times ---
    if repo.get("probe_install_seconds", 0) > 60:
        value -= 2.0

    return value


def select_power_optimized(
    qualified_repos: List[Dict[str, Any]],
    target_count: int = 1000,
) -> List[Dict[str, Any]]:
    """Select repos to maximize statistical power across all claims.

    Uses a greedy algorithm that re-evaluates marginal value at each step.
    O(n * k) where n = pool size and k = target count.
    """
    selected: List[Dict[str, Any]] = []
    remaining = list(qualified_repos)
    counts = _init_counts()

    for i in range(target_count):
        if not remaining:
            break

        # Re-evaluate every 50 selections or in first 20
        if i % 50 == 0 or i < 20:
            for repo in remaining:
                repo["_marginal_value"] = compute_marginal_value(
                    repo, selected, counts,
                )
            remaining.sort(key=lambda r: r["_marginal_value"], reverse=True)

        best = remaining.pop(0)
        selected.append(best)
        _update_counts(counts, best)

        if (i + 1) % 100 == 0:
            _print_coverage_snapshot(counts, i + 1)

    return selected


def _init_counts() -> Dict[str, Any]:
    return {
        "control_total": 0,
        "treatment_total": 0,
        "seen_topologies": set(),
    }


def _update_counts(counts: Dict[str, Any], repo: Dict[str, Any]) -> None:
    preconditions = set(repo.get("confirmed_preconditions", []))

    if len(preconditions) <= 1:
        counts["control_total"] = counts.get("control_total", 0) + 1
    else:
        counts["treatment_total"] = counts.get("treatment_total", 0) + 1

    for pc in preconditions:
        counts[f"pc_treatment_{pc}"] = counts.get(f"pc_treatment_{pc}", 0) + 1

    for pc in PRECONDITION_TARGETS:
        if pc not in preconditions:
            counts[f"pc_control_{pc}"] = counts.get(f"pc_control_{pc}", 0) + 1

    fw = repo.get("primary_framework", "custom")
    counts[f"fw_{fw}"] = counts.get(f"fw_{fw}", 0) + 1

    arch = repo.get("archetype", "")
    counts[f"arch_{arch}"] = counts.get(f"arch_{arch}", 0) + 1

    cx = repo.get("complexity", "medium")
    counts[f"cx_{cx}"] = counts.get(f"cx_{cx}", 0) + 1

    topo = repo.get("topology_hash", "")
    if topo:
        counts["seen_topologies"].add(topo)


def _print_coverage_snapshot(counts: Dict[str, Any], n: int) -> None:
    """Print current coverage status."""
    print(f"\n  --- Selection progress: {n} repos ---")
    print(
        f"  Control: {counts.get('control_total', 0)}  "
        f"Treatment: {counts.get('treatment_total', 0)}"
    )

    under_target = []
    for pc, targets in PRECONDITION_TARGETS.items():
        current = counts.get(f"pc_treatment_{pc}", 0)
        if current < targets["min_treatment"]:
            under_target.append(f"{pc}({current}/{targets['min_treatment']})")

    if under_target:
        suffix = "..." if len(under_target) > 5 else ""
        print(f"  Under target: {', '.join(under_target[:5])}{suffix}")


def validate_composition(selected: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate that the selected dataset meets composition requirements."""
    counts = _init_counts()
    for repo in selected:
        _update_counts(counts, repo)

    report: Dict[str, Any] = {
        "total": len(selected),
        "control": counts.get("control_total", 0),
        "treatment": counts.get("treatment_total", 0),
        "control_fraction": counts.get("control_total", 0) / max(len(selected), 1),
        "precondition_coverage": {},
        "framework_coverage": {},
        "archetype_coverage": {},
        "unmet_targets": [],
    }

    for pc, targets in PRECONDITION_TARGETS.items():
        treatment = counts.get(f"pc_treatment_{pc}", 0)
        control = counts.get(f"pc_control_{pc}", 0)
        met = treatment >= targets["min_treatment"] and control >= targets["min_control"]

        if treatment >= 2:
            z = 1.96
            n = treatment
            p_hat = 0.5
            ci_width = (
                2 * z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
                / (1 + z**2 / n)
            )
        else:
            ci_width = 1.0

        report["precondition_coverage"][pc] = {
            "treatment": treatment,
            "control": control,
            "target_treatment": targets["min_treatment"],
            "target_control": targets["min_control"],
            "met": met,
            "expected_ci_width": round(ci_width, 3),
        }
        if not met:
            report["unmet_targets"].append(pc)

    for fw, targets in FRAMEWORK_TARGETS.items():
        count = counts.get(f"fw_{fw}", 0)
        report["framework_coverage"][fw] = {
            "count": count,
            "target": targets["min"],
            "met": count >= targets["min"],
        }

    for arch, targets in ARCHETYPE_TARGETS.items():
        count = counts.get(f"arch_{arch}", 0)
        report["archetype_coverage"][arch] = {
            "count": count,
            "target": targets["min"],
            "met": count >= targets["min"],
        }

    report["all_targets_met"] = len(report["unmet_targets"]) == 0

    return report
