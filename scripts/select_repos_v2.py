#!/usr/bin/env python3
"""Intelligent repo selection for stratum-lab behavioral scan v2.

Reads scan_results.jsonl (stratum-cli output for ~28k repos) and produces
a ranked list optimized for behavioral dataset value.

Usage:
    python select_repos_v2.py scan_results.jsonl [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# ── PHASE 1: HARD FILTERS ─────────────────────────────────────────────────────

SUPPORTED_FRAMEWORKS = {"CrewAI", "LangGraph", "AutoGen", "LangChain"}

# Finding category prefixes
FINDING_CATEGORIES = {"DC", "OC", "SI", "EA", "AB"}


def extract_finding_category(finding_id: str) -> str:
    """Extract category from a STRAT finding ID like STRAT-DC-001."""
    parts = finding_id.split("-")
    if len(parts) >= 2:
        return parts[1]  # DC, OC, SI, EA, AB
    return "UNKNOWN"


def structural_fingerprint(repo: dict) -> str:
    """Create a dedup fingerprint from structural features."""
    agent_count = repo.get("agent_count", 0)
    framework = repo.get("primary_framework", "")
    rules = sorted(repo.get("finding_rules", []))
    return f"{framework}:{agent_count}:{','.join(rules)}"


def apply_hard_filters(repos: list[dict]) -> list[dict]:
    """Phase 1: eliminate repos that can't produce useful data."""
    filtered = []
    for r in repos:
        agent_count = r.get("agent_count", 0)
        framework = r.get("primary_framework", "")
        tool_count = r.get("total_tool_count", 0)

        # Must be multi-agent
        if agent_count < 2:
            continue

        # Must have a working patcher
        if framework not in SUPPORTED_FRAMEWORKS:
            continue

        # Must have some complexity
        if tool_count < 1 and agent_count < 3:
            continue

        filtered.append(r)

    # Deduplicate by structural fingerprint — keep highest deployment_score
    by_fingerprint: dict[str, dict] = {}
    for r in filtered:
        fp = structural_fingerprint(r)
        existing = by_fingerprint.get(fp)
        if existing is None:
            by_fingerprint[fp] = r
        else:
            dep = r.get("deployment_signals", {})
            dep_score = dep.get("deployment_score", 0) if isinstance(dep, dict) else 0
            existing_dep = existing.get("deployment_signals", {})
            existing_score = existing_dep.get("deployment_score", 0) if isinstance(existing_dep, dict) else 0
            if dep_score > existing_score:
                by_fingerprint[fp] = r

    result = list(by_fingerprint.values())
    print(f"Phase 1: {len(repos)} -> {len(filtered)} after filters -> {len(result)} after dedup")
    return result


# ── PHASE 2: SCORING ────────────────────────────────────────────────────────

# Target framework distribution
TARGET_DISTRIBUTION = {
    "CrewAI": 0.50,
    "LangGraph": 0.25,
    "AutoGen": 0.15,
    "LangChain": 0.10,
}


def score_architectural(repo: dict) -> int:
    """Architectural value (0-30 points)."""
    score = 0
    agent_count = repo.get("agent_count", 0)

    # Agent count scoring
    if agent_count >= 5:
        score += 20
    elif agent_count >= 4:
        score += 15
    elif agent_count >= 3:
        score += 10
    elif agent_count >= 2:
        score += 5

    # Graph topology complexity
    topo = repo.get("graph_topology_metrics", {})
    if isinstance(topo, dict):
        if topo.get("diameter", 0) > 2:
            score += 5

    # Guardrails present
    if repo.get("has_any_guardrails", False):
        score += 5

    return min(score, 30)


def score_finding_diversity(repo: dict, finding_counts: Counter) -> int:
    """Finding diversity value (0-30 points). Repos with rare findings score higher."""
    score = 0
    findings = repo.get("finding_rules", [])
    if not findings:
        return 0

    # Rarity bonus: findings that appear in fewer selected repos are more valuable
    rarity_sum = 0.0
    for f in findings:
        count = finding_counts.get(f, 1)
        # Inverse frequency scoring
        rarity_sum += 1.0 / max(count, 1)

    # Normalize: max 20 points for rarity
    score += min(int(rarity_sum * 5), 20)

    # Category diversity bonus
    categories = {extract_finding_category(f) for f in findings}
    if len(categories) >= 4:
        score += 20
    elif len(categories) >= 3:
        score += 10

    return min(score, 30)


def score_runnability(repo: dict) -> int:
    """Runnability value (0-20 points)."""
    score = 0
    dep = repo.get("deployment_signals", {})
    if isinstance(dep, dict):
        dep_score = dep.get("deployment_score", 0)
        # Map 0-5 deployment score to 0-15 points
        score += min(int(dep_score * 3), 15)
        if dep.get("has_lockfile", False):
            score += 5
    return min(score, 20)


def score_framework_bonus(repo: dict, framework_counts: Counter, total_selected: int) -> int:
    """Framework diversity bonus (0-20 points). Boost underrepresented frameworks."""
    framework = repo.get("primary_framework", "")
    if framework not in TARGET_DISTRIBUTION:
        return 0

    target_ratio = TARGET_DISTRIBUTION[framework]
    current_count = framework_counts.get(framework, 0)
    current_ratio = current_count / max(total_selected, 1)

    # If underrepresented, give bonus proportional to the gap
    gap = target_ratio - current_ratio
    if gap > 0:
        return min(int(gap * 100), 20)
    return 0


def compute_scores(repos: list[dict]) -> list[dict]:
    """Phase 2: Score each repo and add composite_score."""
    # First pass: count findings across all repos for rarity scoring
    global_finding_counts: Counter = Counter()
    for r in repos:
        for f in r.get("finding_rules", []):
            global_finding_counts[f] += 1

    # Iterative scoring with framework rebalancing
    framework_counts: Counter = Counter()
    scored = []

    for r in repos:
        arch = score_architectural(r)
        finding_div = score_finding_diversity(r, global_finding_counts)
        runnability = score_runnability(r)
        fw_bonus = score_framework_bonus(r, framework_counts, len(scored))

        composite = arch + finding_div + runnability + fw_bonus

        r["composite_score"] = composite
        r["score_breakdown"] = {
            "architectural": arch,
            "finding_diversity": finding_div,
            "runnability": runnability,
            "framework_bonus": fw_bonus,
        }
        scored.append(r)

        # Track framework counts for dynamic rebalancing
        framework_counts[r.get("primary_framework", "")] += 1

    # Sort by composite score descending
    scored.sort(key=lambda x: x["composite_score"], reverse=True)

    # Second pass: recalculate framework bonuses with better ordering
    framework_counts.clear()
    for r in scored:
        fw_bonus = score_framework_bonus(r, framework_counts, len(framework_counts))
        old_composite = r["composite_score"]
        r["score_breakdown"]["framework_bonus"] = fw_bonus
        r["composite_score"] = (
            r["score_breakdown"]["architectural"]
            + r["score_breakdown"]["finding_diversity"]
            + r["score_breakdown"]["runnability"]
            + fw_bonus
        )
        framework_counts[r.get("primary_framework", "")] += 1

    # Re-sort after rebalancing
    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    return scored


# ── PHASE 3: COVERAGE GUARANTEES ──────────────────────────────────────────────

def ensure_coverage(repos: list[dict]) -> list[dict]:
    """Phase 3: Force-include repos to meet minimum coverage targets."""
    selected_set = set()
    selected = []

    # Start with top-scored repos
    for r in repos:
        key = r.get("repo_full_name", r.get("repo_url", ""))
        if key not in selected_set:
            selected_set.add(key)
            selected.append(r)

    # Coverage targets
    framework_min = 100  # per supported framework
    finding_category_min = 10  # per STRAT category
    guardrail_min = 20
    complex_min = 50  # agent_count >= 4

    # Check and fix framework coverage
    for fw in ["CrewAI", "LangGraph", "AutoGen"]:
        fw_repos = [r for r in selected if r.get("primary_framework") == fw]
        if len(fw_repos) < framework_min:
            # Find candidates not yet selected
            needed = framework_min - len(fw_repos)
            candidates = [
                r for r in repos
                if r.get("primary_framework") == fw
                and r.get("repo_full_name", r.get("repo_url", "")) not in selected_set
            ]
            candidates.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
            for c in candidates[:needed]:
                key = c.get("repo_full_name", c.get("repo_url", ""))
                selected_set.add(key)
                selected.append(c)

    # Check finding category coverage
    for cat in FINDING_CATEGORIES:
        cat_repos = [
            r for r in selected
            if any(extract_finding_category(f) == cat for f in r.get("finding_rules", []))
        ]
        if len(cat_repos) < finding_category_min:
            needed = finding_category_min - len(cat_repos)
            candidates = [
                r for r in repos
                if any(extract_finding_category(f) == cat for f in r.get("finding_rules", []))
                and r.get("repo_full_name", r.get("repo_url", "")) not in selected_set
            ]
            candidates.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
            for c in candidates[:needed]:
                key = c.get("repo_full_name", c.get("repo_url", ""))
                selected_set.add(key)
                selected.append(c)

    # Check guardrail coverage
    guard_repos = [r for r in selected if r.get("has_any_guardrails", False)]
    if len(guard_repos) < guardrail_min:
        needed = guardrail_min - len(guard_repos)
        candidates = [
            r for r in repos
            if r.get("has_any_guardrails", False)
            and r.get("repo_full_name", r.get("repo_url", "")) not in selected_set
        ]
        candidates.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
        for c in candidates[:needed]:
            key = c.get("repo_full_name", c.get("repo_url", ""))
            selected_set.add(key)
            selected.append(c)

    # Check complex topology coverage
    complex_repos = [r for r in selected if r.get("agent_count", 0) >= 4]
    if len(complex_repos) < complex_min:
        needed = complex_min - len(complex_repos)
        candidates = [
            r for r in repos
            if r.get("agent_count", 0) >= 4
            and r.get("repo_full_name", r.get("repo_url", "")) not in selected_set
        ]
        candidates.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
        for c in candidates[:needed]:
            key = c.get("repo_full_name", c.get("repo_url", ""))
            selected_set.add(key)
            selected.append(c)

    # Final sort by composite score
    selected.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
    return selected


# ── PHASE 4: OUTPUT ─────────────────────────────────────────────────────────

def format_output_record(repo: dict) -> dict:
    """Build the output JSONL record."""
    repo_name = repo.get("repo_full_name", "")
    repo_url = repo.get("repo_url", "")
    if not repo_url and repo_name:
        repo_url = f"https://github.com/{repo_name}"

    dep = repo.get("deployment_signals", {})
    dep_score = dep.get("deployment_score", 0) if isinstance(dep, dict) else 0

    return {
        "repo_url": repo_url,
        "repo_full_name": repo_name,
        "primary_framework": repo.get("primary_framework", ""),
        "agent_count": repo.get("agent_count", 0),
        "composite_score": repo.get("composite_score", 0),
        "score_breakdown": repo.get("score_breakdown", {}),
        "finding_rules": repo.get("finding_rules", []),
        "deployment_score": dep_score,
    }


def print_summary(selected: list[dict]) -> None:
    """Print selection summary statistics."""
    print(f"\n{'='*60}")
    print(f"SELECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total repos selected: {len(selected)}")

    # Framework distribution
    fw_counts: Counter = Counter()
    for r in selected:
        fw_counts[r.get("primary_framework", "Unknown")] += 1
    print(f"\nFramework distribution:")
    for fw, count in fw_counts.most_common():
        pct = count * 100 / len(selected) if selected else 0
        print(f"  {fw:15s}: {count:5d} ({pct:.1f}%)")

    # Finding category coverage
    cat_counts: Counter = Counter()
    for r in selected:
        for f in r.get("finding_rules", []):
            cat = extract_finding_category(f)
            cat_counts[cat] += 1
    print(f"\nFinding category coverage (repos with at least one):")
    for cat in sorted(FINDING_CATEGORIES):
        repos_with = sum(
            1 for r in selected
            if any(extract_finding_category(f) == cat for f in r.get("finding_rules", []))
        )
        print(f"  STRAT-{cat}: {repos_with} repos")

    # Agent count distribution
    agent_dist: Counter = Counter()
    for r in selected:
        ac = r.get("agent_count", 0)
        if ac >= 5:
            agent_dist["5+"] += 1
        else:
            agent_dist[str(ac)] += 1
    print(f"\nAgent count distribution:")
    for k in sorted(agent_dist.keys()):
        print(f"  {k} agents: {agent_dist[k]} repos")

    # Guardrails
    guard_count = sum(1 for r in selected if r.get("has_any_guardrails", False))
    print(f"\nRepos with guardrails: {guard_count}")

    # Top 10
    print(f"\nTop 10 repos by composite score:")
    for r in selected[:10]:
        name = r.get("repo_full_name", r.get("repo_url", "unknown"))
        score = r.get("composite_score", 0)
        fw = r.get("primary_framework", "?")
        ac = r.get("agent_count", 0)
        print(f"  {score:3d}  {fw:12s}  agents={ac}  {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Intelligent repo selection for stratum-lab v2")
    parser.add_argument("scan_results", help="Path to scan_results.jsonl")
    parser.add_argument("--output-dir", default=".", help="Output directory for JSONL files")
    args = parser.parse_args()

    scan_path = Path(args.scan_results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load scan results
    print(f"Loading {scan_path}...")
    repos: list[dict] = []
    with open(scan_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                repos.append(json.loads(line))
            except json.JSONDecodeError:
                if line_num <= 5:
                    print(f"Warning: skipping malformed line {line_num}")
    print(f"Loaded {len(repos)} repos")

    # Phase 1: Hard filters
    filtered = apply_hard_filters(repos)

    # Phase 2: Scoring
    print("Phase 2: Scoring...")
    scored = compute_scores(filtered)
    print(f"Phase 2: {len(scored)} repos scored")

    # Phase 3: Coverage guarantees
    print("Phase 3: Coverage guarantees...")
    selected = ensure_coverage(scored)
    print(f"Phase 3: {len(selected)} repos after coverage guarantees")

    # Phase 4: Output
    output_records = [format_output_record(r) for r in selected]

    # Write main output
    main_output = output_dir / "selected_repos_v2.jsonl"
    with open(main_output, "w", encoding="utf-8") as f:
        for rec in output_records:
            f.write(json.dumps(rec, default=str) + "\n")
    print(f"\nWrote {len(output_records)} repos to {main_output}")

    # Write split outputs for two droplets
    mid = len(output_records) // 2
    split_a = output_dir / "selected_repos_v2_A.jsonl"
    split_b = output_dir / "selected_repos_v2_B.jsonl"

    with open(split_a, "w", encoding="utf-8") as f:
        for rec in output_records[:mid]:
            f.write(json.dumps(rec, default=str) + "\n")

    with open(split_b, "w", encoding="utf-8") as f:
        for rec in output_records[mid:]:
            f.write(json.dumps(rec, default=str) + "\n")

    print(f"Wrote {mid} repos to {split_a}")
    print(f"Wrote {len(output_records) - mid} repos to {split_b}")

    # Print summary
    print_summary(selected)


if __name__ == "__main__":
    main()
