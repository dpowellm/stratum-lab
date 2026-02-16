"""Graph discovery coverage optimizer.

Selects repos to maximize diversity across:
1. Framework strata (8 frameworks, balanced representation)
2. Graph complexity brackets (1 agent, 2-3, 4-7, 8+)
3. Finding configuration space (which STRAT findings fire)
4. Control configuration space (which controls are present/absent)
5. Archetype diversity (sequential, hub-spoke, hierarchical, etc.)
6. XCOMP coverage (repos where security × reliability overlap)

The unit of value is a NEW structural configuration we haven't
observed at runtime yet, not more samples of one we've already seen.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set


# ===== COVERAGE TARGETS =====

# Base targets are calibrated for BASE_TARGET=1000 selection from ~50k pool.
# validate_coverage() scales these proportionally when actual target differs.
BASE_TARGET = 1000

BASE_FRAMEWORK_TARGETS = {
    "crewai": 150, "langgraph": 150, "autogen": 100, "langchain": 100,
    "llamaindex": 60, "smolagents": 40, "agno": 30, "generic": 70,
}

BASE_COMPLEXITY_TARGETS = {
    "single": 80, "small": 200, "medium": 350, "large": 150,
}

FRAMEWORK_TARGETS = {
    "crewai":      {"min": 100, "ideal": 200},
    "langgraph":   {"min": 100, "ideal": 200},
    "autogen":     {"min": 80,  "ideal": 150},
    "langchain":   {"min": 80,  "ideal": 150},
    "llamaindex":  {"min": 40,  "ideal": 80},
    "smolagents":  {"min": 30,  "ideal": 60},
    "agno":        {"min": 20,  "ideal": 50},
    "generic":     {"min": 50,  "ideal": 100},
}

COMPLEXITY_TARGETS = {
    "single":  {"min": 80,  "agents": (1, 1)},
    "small":   {"min": 200, "agents": (2, 3)},
    "medium":  {"min": 350, "agents": (4, 7)},
    "large":   {"min": 150, "agents": (8, 999)},
}


def _scale_targets(base_targets: Dict[str, int], actual_target: int) -> Dict[str, int]:
    """Scale coverage targets proportionally to actual_target / BASE_TARGET."""
    scale = actual_target / BASE_TARGET
    return {k: max(1, round(v * scale)) for k, v in base_targets.items()}

FINDING_COVERAGE = {
    "STRAT-DC-001": 15,
    "STRAT-DC-002": 10,
    "STRAT-SI-001": 15,
    "STRAT-SI-004": 10,
    "STRAT-EA-001": 10,
    "STRAT-OC-002": 10,
    "STRAT-AB-001": 10,
    "STRAT-DC-003": 8,
    "STRAT-SI-002": 8,
    "STRAT-EA-002": 8,
    "STRAT-OC-001": 8,
    "STRAT-XCOMP-001": 5,
    "STRAT-XCOMP-006": 5,
}

# Rich targets with min/ideal for validation reporting (v6.3 B7)
FINDING_COVERAGE_TARGETS = {
    "STRAT-DC-001": {"min": 10, "ideal": 20, "label": "Unsupervised delegation chain"},
    "STRAT-DC-002": {"min": 8,  "ideal": 15, "label": "Missing delegation timeout"},
    "STRAT-DC-003": {"min": 5,  "ideal": 10, "label": "Circular delegation"},
    "STRAT-SI-001": {"min": 10, "ideal": 20, "label": "Error laundering"},
    "STRAT-SI-002": {"min": 5,  "ideal": 10, "label": "Implicit ordering dependency"},
    "STRAT-SI-004": {"min": 8,  "ideal": 15, "label": "Missing output validation"},
    "STRAT-EA-001": {"min": 8,  "ideal": 15, "label": "Unhandled tool failure"},
    "STRAT-EA-002": {"min": 5,  "ideal": 10, "label": "Single point of failure"},
    "STRAT-OC-001": {"min": 5,  "ideal": 10, "label": "Shared tool concurrency"},
    "STRAT-OC-002": {"min": 8,  "ideal": 15, "label": "Shared state contention"},
    "STRAT-AB-001": {"min": 8,  "ideal": 15, "label": "Missing rate limiting"},
    "STRAT-XCOMP-001": {"min": 3, "ideal": 8, "label": "Cross-domain: security x reliability"},
    "STRAT-XCOMP-006": {"min": 3, "ideal": 8, "label": "Cross-domain: availability x integrity"},
}

CONTROL_DIVERSITY = {
    "human_gate":          {"with": 30, "without": 30},
    "schema_validation":   {"with": 20, "without": 20},
    "error_boundary":      {"with": 20, "without": 20},
    "timeout":             {"with": 20, "without": 20},
    "observability_sink":  {"with": 15, "without": 15},
    "rate_limiter":        {"with": 10, "without": 10},
}


def compute_discovery_value(
    repo: Dict,
    current_counts: Dict,
    seen_topologies: Set[str],
) -> float:
    """Compute the marginal graph discovery value of adding this repo."""
    value = 0.0

    # --- Topology uniqueness (highest value) ---
    topo_hash = repo.get("topology_hash", "")
    if topo_hash and topo_hash not in seen_topologies:
        value += 15.0
    elif topo_hash:
        value += 1.0

    # --- Framework coverage ---
    fw = repo.get("framework", "generic")
    fw_key = f"fw_{fw}"
    fw_target = FRAMEWORK_TARGETS.get(fw, {"min": 20})
    current_fw = current_counts.get(fw_key, 0)
    if current_fw < fw_target["min"]:
        gap = 1.0 - (current_fw / fw_target["min"])
        value += 10.0 * gap

    # --- Complexity bracket ---
    agents = repo.get("agent_count", 1)
    for bracket, target in COMPLEXITY_TARGETS.items():
        lo, hi = target["agents"]
        if lo <= agents <= hi:
            cx_key = f"cx_{bracket}"
            current_cx = current_counts.get(cx_key, 0)
            if current_cx < target["min"]:
                gap = 1.0 - (current_cx / target["min"])
                value += 8.0 * gap
            break

    # --- Finding coverage ---
    findings = set(repo.get("finding_ids", []))
    for finding_id, target_n in FINDING_COVERAGE.items():
        if finding_id in findings:
            fk = f"finding_{finding_id}"
            current_f = current_counts.get(fk, 0)
            if current_f < target_n:
                gap = 1.0 - (current_f / target_n)
                multiplier = 12.0 if "XCOMP" in finding_id else 6.0
                value += multiplier * gap

    # --- Control configuration diversity ---
    present_controls = set(repo.get("present_control_types", []))
    for control_type, targets in CONTROL_DIVERSITY.items():
        if control_type in present_controls:
            ck = f"ctrl_with_{control_type}"
            current_c = current_counts.get(ck, 0)
            if current_c < targets["with"]:
                value += 4.0 * (1.0 - current_c / targets["with"])
        else:
            ck = f"ctrl_without_{control_type}"
            current_c = current_counts.get(ck, 0)
            if current_c < targets["without"]:
                value += 4.0 * (1.0 - current_c / targets["without"])

    # --- Runnability scaling ---
    runnability = repo.get("estimated_runnability", 0.5)
    value *= (0.5 + 0.5 * runnability)

    # --- XCOMP bonus ---
    if repo.get("xcomp_findings"):
        value += 8.0

    return value


def select_for_discovery(
    repos: List[Dict],
    target_count: int = 1000,
) -> List[Dict]:
    """Select repos to maximize graph discovery coverage."""
    selected = []
    remaining = list(repos)
    counts: Dict = {}
    seen_topos: Set[str] = set()

    for i in range(target_count):
        if not remaining:
            break

        if i % 50 == 0 or i < 20:
            for repo in remaining:
                repo["_discovery_value"] = compute_discovery_value(
                    repo, counts, seen_topos
                )
            remaining.sort(key=lambda r: r["_discovery_value"], reverse=True)

        best = remaining.pop(0)
        selected.append(best)
        _update_counts(counts, seen_topos, best)

    return selected


def _update_counts(counts: Dict, seen_topos: Set[str], repo: Dict) -> None:
    """Update coverage counts after selecting a repo."""
    fw = repo.get("framework", "generic")
    counts[f"fw_{fw}"] = counts.get(f"fw_{fw}", 0) + 1

    agents = repo.get("agent_count", 1)
    for bracket, target in COMPLEXITY_TARGETS.items():
        lo, hi = target["agents"]
        if lo <= agents <= hi:
            counts[f"cx_{bracket}"] = counts.get(f"cx_{bracket}", 0) + 1
            break

    for fid in repo.get("finding_ids", []):
        counts[f"finding_{fid}"] = counts.get(f"finding_{fid}", 0) + 1

    for ct in repo.get("present_control_types", []):
        counts[f"ctrl_with_{ct}"] = counts.get(f"ctrl_with_{ct}", 0) + 1

    topo = repo.get("topology_hash", "")
    if topo:
        seen_topos.add(topo)


def validate_coverage(selected: List[Dict], target_count: int | None = None) -> Dict:
    """Validate the selected set meets coverage requirements.

    Parameters
    ----------
    selected:
        The selected repos.
    target_count:
        If provided, framework and complexity targets are scaled
        proportionally to target_count / BASE_TARGET.
    """
    counts: Dict = {}
    seen_topos: Set[str] = set()
    for repo in selected:
        _update_counts(counts, seen_topos, repo)

    # Scale targets if target_count provided
    if target_count is not None:
        scaled_fw = _scale_targets(BASE_FRAMEWORK_TARGETS, target_count)
        scaled_cx = _scale_targets(BASE_COMPLEXITY_TARGETS, target_count)
    else:
        scaled_fw = BASE_FRAMEWORK_TARGETS
        scaled_cx = BASE_COMPLEXITY_TARGETS

    report: Dict[str, Any] = {
        "total": len(selected),
        "unique_topologies": len(seen_topos),
        "framework_coverage": {},
        "complexity_coverage": {},
        "finding_coverage": {},
        "control_coverage": {},
        "gaps": [],
    }

    for fw in FRAMEWORK_TARGETS:
        n = counts.get(f"fw_{fw}", 0)
        target_min = scaled_fw.get(fw, 1)
        met = n >= target_min
        report["framework_coverage"][fw] = {"count": n, "target": target_min, "met": met}
        if not met:
            report["gaps"].append(f"{fw}: {n}/{target_min}")

    for bracket in COMPLEXITY_TARGETS:
        n = counts.get(f"cx_{bracket}", 0)
        target_min = scaled_cx.get(bracket, 1)
        met = n >= target_min
        report["complexity_coverage"][bracket] = {"count": n, "target": target_min, "met": met}
        if not met:
            report["gaps"].append(f"complexity_{bracket}: {n}/{target_min}")

    for fid, target_n in FINDING_COVERAGE.items():
        n = counts.get(f"finding_{fid}", 0)
        rich = FINDING_COVERAGE_TARGETS.get(fid, {})
        min_target = rich.get("min", target_n)
        ideal_target = rich.get("ideal", target_n)
        met_min = n >= min_target
        met_ideal = n >= ideal_target
        report["finding_coverage"][fid] = {
            "count": n,
            "target": target_n,
            "min": min_target,
            "ideal": ideal_target,
            "met": met_min,
            "met_ideal": met_ideal,
            "label": rich.get("label", fid),
        }
        if not met_min:
            report["gaps"].append(f"{fid}: {n}/{min_target} (min)")

    report["all_targets_met"] = len(report["gaps"]) == 0
    return report
