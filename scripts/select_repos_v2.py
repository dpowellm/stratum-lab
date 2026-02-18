#!/usr/bin/env python3
"""Intelligent repo selection for stratum-lab behavioral scan.

Selects repos that maximize value for the statistical reference model:
- Multi-agent systems (not single-agent chatbots)
- Framework diversity (CrewAI, LangGraph, AutoGen, LangChain)
- Finding coverage (need WITH and WITHOUT each finding for correlations)
- Architectural diversity (different topologies and complexity levels)
- Runnability (repos that will actually execute)
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── CONFIG ───────────────────────────────────────────────────────────────────
TARGET_TOTAL = 4000
SUPPORTED_FRAMEWORKS = {"CrewAI", "LangGraph", "AutoGen", "LangChain"}

# Target framework distribution (percentages of final selection)
FRAMEWORK_TARGETS = {
    "CrewAI": 0.45,      # Best patcher, richest data
    "LangChain": 0.25,   # Largest pool
    "LangGraph": 0.15,   # Important for product claims
    "AutoGen": 0.15,     # Important for product claims
}

# Minimum coverage guarantees
MIN_PER_FRAMEWORK = 80
MIN_PER_FINDING_PREFIX = 10
MIN_WITH_GUARDRAILS = 20
MIN_COMPLEX = 50  # agent_count >= 4


def get_agent_count(repo):
    """Get effective agent count, falling back to crew_size_distribution."""
    ac = repo.get("agent_count", 0)
    if ac < 2:
        csd = repo.get("crew_size_distribution", [])
        if csd:
            ac = max(csd)
    return ac


def get_framework(repo):
    """Get primary framework from frameworks list."""
    frameworks = repo.get("frameworks", [])
    if not frameworks:
        return "none"
    # Prefer supported frameworks
    for fw in frameworks:
        if fw in SUPPORTED_FRAMEWORKS:
            return fw
    return frameworks[0]


def get_finding_prefixes(repo):
    """Extract unique finding prefixes (e.g., STRATUM, CONTEXT, TELEMETRY)."""
    rules = repo.get("finding_rules", [])
    prefixes = set()
    for r in rules:
        # Split on hyphen or dash — prefix is everything before the last segment
        parts = r.rsplit("-", 1)
        if len(parts) >= 1:
            prefixes.add(parts[0])
    return prefixes


def structural_fingerprint(repo):
    """Dedup key: framework + agent_count + sorted findings."""
    fw = get_framework(repo)
    ac = get_agent_count(repo)
    rules = sorted(repo.get("finding_rules", []))
    return f"{fw}:{ac}:{','.join(rules)}"


# ── PHASE 1: LOAD AND FILTER ────────────────────────────────────────────────
def phase1_filter(input_path):
    """Hard filters: must be multi-agent with supported framework."""
    raw = []
    with open(input_path) as f:
        for line in f:
            try:
                raw.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(raw)} repos")

    # Filter
    passed = []
    filtered_reasons = Counter()

    for r in raw:
        ac = get_agent_count(r)
        fw = get_framework(r)
        tool_count = r.get("total_tool_count", 0)

        # Must be multi-agent
        if ac < 2:
            filtered_reasons["agent_count < 2"] += 1
            continue

        # Must have a supported framework
        if fw not in SUPPORTED_FRAMEWORKS:
            filtered_reasons[f"unsupported framework: {fw}"] += 1
            continue

        # Must have some complexity
        if tool_count < 1 and ac < 3:
            filtered_reasons["too simple (< 1 tool and < 3 agents)"] += 1
            continue

        passed.append(r)

    print(f"\nPhase 1 filters: {len(raw)} -> {len(passed)}")
    for reason, count in filtered_reasons.most_common():
        print(f"  Filtered {count}: {reason}")

    # Dedup by structural fingerprint (keep highest deployment_score)
    groups = defaultdict(list)
    for r in passed:
        fp = structural_fingerprint(r)
        groups[fp].append(r)

    deduped = []
    for fp, repos in groups.items():
        best = max(repos, key=lambda r: r.get("deployment_signals", {}).get("deployment_score", 0))
        deduped.append(best)

    print(f"After dedup: {len(passed)} -> {len(deduped)}")
    return deduped


# ── PHASE 2: SCORING ────────────────────────────────────────────────────────
def score_repo(repo, framework_counts, finding_counts):
    """Score a repo by value to the dataset."""
    ac = get_agent_count(repo)
    fw = get_framework(repo)
    score = 0
    breakdown = {}

    # a) Architectural value (0-30)
    arch = 0
    if ac >= 5:
        arch = 20
    elif ac >= 4:
        arch = 15
    elif ac >= 3:
        arch = 10
    else:
        arch = 5

    topo = repo.get("graph_topology_metrics", {})
    if topo.get("diameter", 0) > 2:
        arch += 5
    if repo.get("has_any_guardrails", False):
        arch += 5
    arch = min(arch, 30)
    breakdown["architectural"] = arch
    score += arch

    # b) Finding diversity value (0-30)
    find_score = 0
    rules = repo.get("finding_rules", [])
    prefixes = get_finding_prefixes(repo)

    # Rare findings are more valuable
    for rule in rules:
        count = finding_counts.get(rule, 0)
        if count < 50:
            find_score += 5
        elif count < 200:
            find_score += 3
        elif count < 500:
            find_score += 1

    if len(prefixes) >= 4:
        find_score += 10
    elif len(prefixes) >= 3:
        find_score += 5

    find_score = min(find_score, 30)
    breakdown["finding_diversity"] = find_score
    score += find_score

    # c) Runnability value (0-20)
    run_score = 0
    ds = repo.get("deployment_signals", {})
    dep_score = ds.get("deployment_score", 0)
    run_score += min(dep_score * 3, 15)
    if ds.get("has_lockfile", False):
        run_score += 5
    run_score = min(run_score, 20)
    breakdown["runnability"] = run_score
    score += run_score

    # d) Framework diversity bonus (0-20)
    fw_bonus = 0
    total_selected = sum(framework_counts.values()) or 1
    current_pct = framework_counts.get(fw, 0) / total_selected
    target_pct = FRAMEWORK_TARGETS.get(fw, 0.1)
    if current_pct < target_pct:
        # Underrepresented — boost
        gap = target_pct - current_pct
        fw_bonus = min(int(gap * 200), 20)
    breakdown["framework_bonus"] = fw_bonus
    score += fw_bonus

    return score, breakdown


def phase2_score(repos):
    """Score all repos, iteratively updating framework counts."""
    # Pre-compute finding frequencies
    finding_counts = Counter()
    for r in repos:
        for rule in r.get("finding_rules", []):
            finding_counts[rule] += 1

    framework_counts = Counter()

    # Score all repos with initial zero framework counts
    scored = []
    for r in repos:
        s, bd = score_repo(r, framework_counts, finding_counts)
        scored.append((s, bd, r))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Re-score top N iteratively (framework bonus adjusts as we select)
    selected = []
    framework_counts = Counter()

    for s, bd, r in scored:
        if len(selected) >= TARGET_TOTAL * 1.2:  # overshoot for coverage phase
            break
        # Re-score with current framework counts
        new_s, new_bd = score_repo(r, framework_counts, finding_counts)
        fw = get_framework(r)
        framework_counts[fw] += 1
        selected.append((new_s, new_bd, r))

    selected.sort(key=lambda x: x[0], reverse=True)
    print(f"\nPhase 2: {len(selected)} repos scored")
    return selected


# ── PHASE 3: COVERAGE GUARANTEES ────────────────────────────────────────────
def phase3_coverage(scored_repos, all_repos):
    """Ensure minimum coverage for frameworks, findings, complexity."""
    selected = [(s, bd, r) for s, bd, r in scored_repos[:TARGET_TOTAL]]
    selected_urls = {r.get("repo_full_name") for _, _, r in selected}

    def add_if_missing(repo, reason):
        name = repo.get("repo_full_name")
        if name not in selected_urls:
            selected.append((0, {"coverage_fill": reason}, repo))
            selected_urls.add(name)
            return True
        return False

    # Framework minimums
    for fw in SUPPORTED_FRAMEWORKS:
        fw_repos = [(s, bd, r) for s, bd, r in selected if get_framework(r) == fw]
        if len(fw_repos) < MIN_PER_FRAMEWORK:
            need = MIN_PER_FRAMEWORK - len(fw_repos)
            candidates = [r for r in all_repos if get_framework(r) == fw and r.get("repo_full_name") not in selected_urls]
            candidates.sort(key=lambda r: r.get("deployment_signals", {}).get("deployment_score", 0), reverse=True)
            added = 0
            for r in candidates[:need]:
                if add_if_missing(r, f"framework_fill:{fw}"):
                    added += 1
            if added > 0:
                print(f"  Added {added} repos for {fw} coverage (had {len(fw_repos)})")

    # Complex topology minimum
    complex_repos = [(s, bd, r) for s, bd, r in selected if get_agent_count(r) >= 4]
    if len(complex_repos) < MIN_COMPLEX:
        need = MIN_COMPLEX - len(complex_repos)
        candidates = [r for r in all_repos if get_agent_count(r) >= 4 and r.get("repo_full_name") not in selected_urls]
        candidates.sort(key=lambda r: get_agent_count(r), reverse=True)
        added = 0
        for r in candidates[:need]:
            if add_if_missing(r, "complex_fill"):
                added += 1
        if added > 0:
            print(f"  Added {added} repos for complex topology coverage (had {len(complex_repos)})")

    # Guardrail minimum
    guard_repos = [(s, bd, r) for s, bd, r in selected if r.get("has_any_guardrails", False)]
    if len(guard_repos) < MIN_WITH_GUARDRAILS:
        need = MIN_WITH_GUARDRAILS - len(guard_repos)
        candidates = [r for r in all_repos if r.get("has_any_guardrails", False) and r.get("repo_full_name") not in selected_urls]
        candidates.sort(key=lambda r: r.get("deployment_signals", {}).get("deployment_score", 0), reverse=True)
        added = 0
        for r in candidates[:need]:
            if add_if_missing(r, "guardrail_fill"):
                added += 1
        if added > 0:
            print(f"  Added {added} repos for guardrail coverage (had {len(guard_repos)})")

    # Trim to target if over
    selected.sort(key=lambda x: x[0], reverse=True)
    selected = selected[:TARGET_TOTAL]

    print(f"\nPhase 3: {len(selected)} repos after coverage guarantees")
    return selected


# ── PHASE 4: OUTPUT ─────────────────────────────────────────────────────────
def phase4_output(selected, output_dir):
    """Write output files and print summary."""
    output_dir = Path(output_dir)

    # Build output records
    records = []
    for score, breakdown, repo in selected:
        records.append({
            "repo_url": f"https://github.com/{repo.get('repo_full_name', '')}",
            "repo_full_name": repo.get("repo_full_name", ""),
            "primary_framework": get_framework(repo),
            "agent_count": get_agent_count(repo),
            "composite_score": score,
            "score_breakdown": breakdown,
            "finding_rules": repo.get("finding_rules", []),
            "deployment_score": repo.get("deployment_signals", {}).get("deployment_score", 0),
            "has_guardrails": repo.get("has_any_guardrails", False),
            "graph_topology": repo.get("graph_topology_metrics", {}),
        })

    # Write full list
    full_path = output_dir / "selected_repos_v2.jsonl"
    with open(full_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Split A/B
    mid = len(records) // 2
    a_path = output_dir / "selected_repos_v2_A.jsonl"
    b_path = output_dir / "selected_repos_v2_B.jsonl"
    with open(a_path, "w") as f:
        for r in records[:mid]:
            f.write(json.dumps(r) + "\n")
    with open(b_path, "w") as f:
        for r in records[mid:]:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(records)} repos to {full_path}")
    print(f"Wrote {mid} repos to {a_path}")
    print(f"Wrote {len(records) - mid} repos to {b_path}")

    # ── SUMMARY ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SELECTION SUMMARY")
    print("=" * 60)
    print(f"Total repos selected: {len(records)}")

    # Framework distribution
    fw_dist = Counter(r["primary_framework"] for r in records)
    print(f"\nFramework distribution:")
    for fw, count in fw_dist.most_common():
        pct = count / len(records) * 100
        target = FRAMEWORK_TARGETS.get(fw, 0) * 100
        print(f"  {fw:15s}: {count:5d} ({pct:.1f}%, target {target:.0f}%)")

    # Finding coverage
    finding_prefixes = Counter()
    all_findings = Counter()
    for r in records:
        prefixes = set()
        for rule in r["finding_rules"]:
            all_findings[rule] += 1
            parts = rule.rsplit("-", 1)
            if parts:
                prefixes.add(parts[0])
        for p in prefixes:
            finding_prefixes[p] += 1

    print(f"\nFinding prefix coverage:")
    for prefix, count in finding_prefixes.most_common():
        print(f"  {prefix:20s}: {count:5d} repos")

    print(f"\nTop 15 individual findings:")
    for finding, count in all_findings.most_common(15):
        print(f"  {finding:25s}: {count:5d} repos")

    # Agent count distribution
    ac_dist = Counter(r["agent_count"] for r in records)
    print(f"\nAgent count distribution:")
    for ac in sorted(ac_dist.keys()):
        print(f"  {ac} agents: {ac_dist[ac]:5d} repos")

    # Guardrails
    guard_count = sum(1 for r in records if r["has_guardrails"])
    print(f"\nRepos with guardrails: {guard_count}")

    # Top 10
    print(f"\nTop 10 repos by composite score:")
    for r in records[:10]:
        print(f"  {r['composite_score']:3d}  {r['primary_framework']:12s}  {r['agent_count']}a  {r['repo_full_name']}")


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python select_repos_v2.py <scan_results.jsonl> [output_dir]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else str(Path(input_path).parent)

    all_repos = phase1_filter(input_path)
    scored = phase2_score(all_repos)
    final = phase3_coverage(scored, all_repos)
    phase4_output(final, output_dir)
