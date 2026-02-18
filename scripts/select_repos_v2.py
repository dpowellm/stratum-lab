#!/usr/bin/env python3
"""Intelligent repo selection for stratum-lab behavioral scan.

Optimized for building a REFERENCE DISTRIBUTION that mirrors enterprise
deployment patterns. When Stratum scans a customer's agent topology, we run
statistical inference against this dataset to surface behavioral risk clusters.

Design rationale:
─────────────────
Agent count cap (50):  Repos claiming 200-800 agents are mislabeled — they're
  counting functions, configs, or class defs. Cap at 50; flag but don't filter.

Complexity tiers:  Enterprise deployments cluster at 3-15 agents. 2-agent repos
  are toy examples with non-transferable baselines. 30+ is rare in production.
  Scoring curve peaks in the enterprise sweet spot.

Framework strategy:
  LangGraph/AutoGen — scarce (~350-370 post-dedup) + disproportionately
    enterprise-relevant. Take nearly all quality repos.
  CrewAI — cleanest multi-agent signal (opinionated roles/tasks make behavioral
    attribution easier). Growing enterprise adoption. High allocation.
  LangChain — abundant but noisy. Many "multi-agent" repos are really single-
    agent-with-tools. Be highly selective: only take high-complexity, high-
    finding-diversity repos.

Finding diversity is the #1 scoring factor. More unique findings per repo =
  more behavioral signal per scan = better statistical power.

Runnability is real but bounded. Many repos will fail regardless (missing API
  keys, proprietary data, broken deps). Don't over-index — a repo with 5
  unique findings that might not run is more valuable than a toy 2-agent repo
  with a lockfile.

Telemetry-only penalty. Repos whose only findings are TELEMETRY-* contribute
  minimal behavioral signal — they just lack observability, which is the most
  common and least interesting finding.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── CONFIG ───────────────────────────────────────────────────────────────────
TARGET_TOTAL = 4000
SUPPORTED_FRAMEWORKS = {"CrewAI", "LangGraph", "AutoGen", "LangChain"}

# Cap agent count — anything above is flagged as suspect but kept with
# capped value so inflated counts don't distort scoring
AGENT_COUNT_CAP = 50

# Framework allocation targets (% of final selection)
# Driven by: scarcity × enterprise relevance × signal quality
FRAMEWORK_TARGETS = {
    "LangGraph": 0.09,   # ~360 — take nearly all (scarce, stateful, enterprise)
    "AutoGen":   0.075,  # ~300 — take nearly all (scarce, MSFT enterprise)
    "CrewAI":    0.35,   # ~1400 — clean signal, structured roles
    "LangChain": 0.485,  # ~1940 — selective: complexity + finding diversity
}

# Minimum coverage floors (safety net, not primary allocation mechanism)
MIN_PER_FRAMEWORK = 80
MIN_PER_FINDING_PREFIX = 30   # need enough WITH and WITHOUT for correlations
MIN_WITH_GUARDRAILS = 100     # enterprise maturity signal


# ── HELPERS ──────────────────────────────────────────────────────────────────

def get_agent_count(repo):
    """Get effective agent count, falling back to crew_size_distribution.
    Capped at AGENT_COUNT_CAP to prevent mislabeled repos from distorting."""
    ac = repo.get("agent_count", 0)
    if ac < 2:
        csd = repo.get("crew_size_distribution", [])
        if csd:
            ac = max(csd)
    return min(ac, AGENT_COUNT_CAP)


def get_raw_agent_count(repo):
    """Uncapped agent count for reporting/flagging."""
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
    for fw in frameworks:
        if fw in SUPPORTED_FRAMEWORKS:
            return fw
    return frameworks[0]


def get_finding_prefixes(repo):
    """Extract unique finding prefixes (e.g., STRATUM, CONTEXT, TELEMETRY)."""
    rules = repo.get("finding_rules", [])
    prefixes = set()
    for r in rules:
        parts = r.rsplit("-", 1)
        if len(parts) >= 1:
            prefixes.add(parts[0])
    return prefixes


def get_non_telemetry_findings(repo):
    """Get findings excluding TELEMETRY-* (the least interesting category)."""
    return [r for r in repo.get("finding_rules", []) if not r.startswith("TELEMETRY")]


def is_telemetry_only(repo):
    """True if the repo's only findings are TELEMETRY-*."""
    rules = repo.get("finding_rules", [])
    if not rules:
        return True  # no findings at all
    return all(r.startswith("TELEMETRY") for r in rules)


def structural_fingerprint(repo):
    """Dedup key: framework + capped agent_count + sorted findings."""
    fw = get_framework(repo)
    ac = get_agent_count(repo)
    rules = sorted(repo.get("finding_rules", []))
    return f"{fw}:{ac}:{','.join(rules)}"


# ── PHASE 1: LOAD AND FILTER ────────────────────────────────────────────────

def phase1_filter(input_path):
    """Hard filters: multi-agent, supported framework, minimum signal."""
    raw = []
    with open(input_path) as f:
        for line in f:
            try:
                raw.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(raw)} repos")

    passed = []
    filtered_reasons = Counter()
    capped_count = 0

    for r in raw:
        raw_ac = get_raw_agent_count(r)
        ac = get_agent_count(r)  # capped
        fw = get_framework(r)
        tool_count = r.get("total_tool_count", 0)

        if raw_ac > AGENT_COUNT_CAP:
            capped_count += 1

        # Must be multi-agent
        if raw_ac < 2:
            filtered_reasons["agent_count < 2"] += 1
            continue

        # Must have a supported framework
        if fw not in SUPPORTED_FRAMEWORKS:
            filtered_reasons[f"unsupported framework: {fw}"] += 1
            continue

        # Must have some complexity (2-agent repos need at least 1 tool)
        if tool_count < 1 and ac < 3:
            filtered_reasons["too simple (< 1 tool and < 3 agents)"] += 1
            continue

        passed.append(r)

    print(f"\nPhase 1 filters: {len(raw)} -> {len(passed)}")
    for reason, count in filtered_reasons.most_common():
        print(f"  Filtered {count}: {reason}")
    print(f"  Agent count capped at {AGENT_COUNT_CAP} for {capped_count} repos")

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
    """Score a repo by value to the enterprise reference dataset.

    Scoring budget (0-100):
      Finding diversity:    0-35  (most behavioral signal per scan)
      Architectural value:  0-30  (enterprise complexity sweet spot)
      Runnability:          0-15  (bounded — many will fail regardless)
      Framework scarcity:   0-20  (LangGraph/AutoGen bonus)
    """
    ac = get_agent_count(repo)
    fw = get_framework(repo)
    score = 0
    breakdown = {}

    # ── a) Finding diversity (0-35) — THE #1 SIGNAL ──────────────────────
    find_score = 0
    rules = repo.get("finding_rules", [])
    prefixes = get_finding_prefixes(repo)
    non_telemetry = get_non_telemetry_findings(repo)

    # Telemetry-only penalty: these repos add almost no behavioral signal
    if is_telemetry_only(repo):
        find_score = -10
    else:
        # Rare findings are more valuable (better statistical power)
        for rule in non_telemetry:
            count = finding_counts.get(rule, 0)
            if count < 50:
                find_score += 6
            elif count < 200:
                find_score += 3
            elif count < 500:
                find_score += 1

        # Breadth bonus — more prefixes = more dimensions of behavioral signal
        if len(prefixes) >= 5:
            find_score += 15
        elif len(prefixes) >= 4:
            find_score += 10
        elif len(prefixes) >= 3:
            find_score += 5

    find_score = max(min(find_score, 35), -10)  # allow negative for telemetry-only
    breakdown["finding_diversity"] = find_score
    score += find_score

    # ── b) Architectural value (0-30) — enterprise sweet spot ────────────
    arch = 0

    # Complexity tier scoring — peaks at 3-15, tapers at extremes
    if 3 <= ac <= 7:
        arch = 20       # core enterprise range — maximize
    elif 8 <= ac <= 15:
        arch = 18       # complex enterprise — high value
    elif 16 <= ac <= 30:
        arch = 12       # advanced orchestration — still useful
    elif 31 <= ac <= 50:
        arch = 6        # probably mislabeled, but keep some
    elif ac == 2:
        arch = 4        # minimal — only for baseline comparisons

    # Topology bonus
    topo = repo.get("graph_topology_metrics", {})
    if topo.get("diameter", 0) > 2:
        arch += 5       # non-trivial communication topology

    # Guardrails bonus — correlates with enterprise maturity
    if repo.get("has_any_guardrails", False):
        arch += 5

    arch = min(arch, 30)
    breakdown["architectural"] = arch
    score += arch

    # ── c) Runnability (0-15) — real but bounded ─────────────────────────
    # Many repos won't run (missing API keys, proprietary data, broken deps).
    # A lockfile or high deployment_score helps, but don't over-weight —
    # a repo with 5 unique findings that might not run is more valuable than
    # a toy 2-agent repo with perfect deps.
    run_score = 0
    ds = repo.get("deployment_signals", {})
    dep_score = ds.get("deployment_score", 0)
    run_score += min(dep_score * 2, 10)
    if ds.get("has_lockfile", False):
        run_score += 3
    if ds.get("has_dockerfile", False) or ds.get("has_docker_compose", False):
        run_score += 2
    run_score = min(run_score, 15)
    breakdown["runnability"] = run_score
    score += run_score

    # ── d) Framework scarcity bonus (0-20) ───────────────────────────────
    # LangGraph and AutoGen are scarce but enterprise-critical.
    # Dynamic: bonus shrinks as we accumulate repos from that framework.
    fw_bonus = 0
    total_selected = sum(framework_counts.values()) or 1
    current_pct = framework_counts.get(fw, 0) / total_selected
    target_pct = FRAMEWORK_TARGETS.get(fw, 0.1)
    if current_pct < target_pct:
        gap = target_pct - current_pct
        fw_bonus = min(int(gap * 200), 20)
    breakdown["framework_scarcity"] = fw_bonus
    score += fw_bonus

    return score, breakdown


def phase2_score(repos):
    """Score all repos, iteratively updating framework counts."""
    # Pre-compute finding frequencies across the full candidate pool
    finding_counts = Counter()
    for r in repos:
        for rule in r.get("finding_rules", []):
            finding_counts[rule] += 1

    framework_counts = Counter()

    # Initial score pass (no framework counts yet)
    scored = []
    for r in repos:
        s, bd = score_repo(r, framework_counts, finding_counts)
        scored.append((s, bd, r))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Re-score iteratively — framework bonus adjusts as we "select"
    selected = []
    framework_counts = Counter()

    for s, bd, r in scored:
        if len(selected) >= TARGET_TOTAL * 1.3:  # overshoot for coverage phase
            break
        new_s, new_bd = score_repo(r, framework_counts, finding_counts)
        fw = get_framework(r)
        framework_counts[fw] += 1
        selected.append((new_s, new_bd, r))

    selected.sort(key=lambda x: x[0], reverse=True)
    print(f"\nPhase 2: {len(selected)} repos scored")

    # Report telemetry-only stats
    telem_only = sum(1 for _, _, r in selected if is_telemetry_only(r))
    print(f"  Telemetry-only repos (penalized): {telem_only}")

    return selected


# ── PHASE 3: COVERAGE GUARANTEES ────────────────────────────────────────────

def phase3_coverage(scored_repos, all_repos):
    """Ensure minimum coverage for frameworks, findings, guardrails.

    Also enforces framework CAPS to prevent LangChain from flooding out
    scarce frameworks during the trim-to-target step.
    """
    selected = [(s, bd, r) for s, bd, r in scored_repos[:TARGET_TOTAL]]
    selected_names = {r.get("repo_full_name") for _, _, r in selected}

    def add_if_missing(repo, reason):
        name = repo.get("repo_full_name")
        if name not in selected_names:
            selected.append((0, {"coverage_fill": reason}, repo))
            selected_names.add(name)
            return True
        return False

    # ── Framework minimums ───────────────────────────────────────────────
    for fw in SUPPORTED_FRAMEWORKS:
        fw_repos = [x for x in selected if get_framework(x[2]) == fw]
        if len(fw_repos) < MIN_PER_FRAMEWORK:
            need = MIN_PER_FRAMEWORK - len(fw_repos)
            candidates = [
                r for r in all_repos
                if get_framework(r) == fw
                and r.get("repo_full_name") not in selected_names
            ]
            candidates.sort(
                key=lambda r: len(get_non_telemetry_findings(r)),
                reverse=True
            )
            added = 0
            for r in candidates[:need]:
                if add_if_missing(r, f"framework_floor:{fw}"):
                    added += 1
            if added > 0:
                print(f"  Added {added} repos for {fw} floor (had {len(fw_repos)})")

    # ── Finding prefix minimums — need WITH and WITHOUT for correlations ─
    prefix_counts = Counter()
    for _, _, r in selected:
        for p in get_finding_prefixes(r):
            prefix_counts[p] += 1

    all_prefixes = set()
    for r in all_repos:
        all_prefixes.update(get_finding_prefixes(r))

    for prefix in all_prefixes:
        if prefix_counts.get(prefix, 0) < MIN_PER_FINDING_PREFIX:
            need = MIN_PER_FINDING_PREFIX - prefix_counts.get(prefix, 0)
            candidates = [
                r for r in all_repos
                if prefix in get_finding_prefixes(r)
                and r.get("repo_full_name") not in selected_names
            ]
            candidates.sort(
                key=lambda r: len(get_non_telemetry_findings(r)),
                reverse=True
            )
            added = 0
            for r in candidates[:need]:
                if add_if_missing(r, f"finding_prefix_floor:{prefix}"):
                    added += 1
            if added > 0:
                print(f"  Added {added} repos for {prefix} coverage (had {prefix_counts.get(prefix, 0)})")

    # ── Guardrail minimum ────────────────────────────────────────────────
    guard_repos = [x for x in selected if x[2].get("has_any_guardrails", False)]
    if len(guard_repos) < MIN_WITH_GUARDRAILS:
        need = MIN_WITH_GUARDRAILS - len(guard_repos)
        candidates = [
            r for r in all_repos
            if r.get("has_any_guardrails", False)
            and r.get("repo_full_name") not in selected_names
        ]
        candidates.sort(
            key=lambda r: len(get_non_telemetry_findings(r)),
            reverse=True
        )
        added = 0
        for r in candidates[:need]:
            if add_if_missing(r, "guardrail_floor"):
                added += 1
        if added > 0:
            print(f"  Added {added} repos for guardrail coverage (had {len(guard_repos)})")

    # ── Framework caps — prevent LangChain from flooding during trim ─────
    # Sort by score, then trim from the bottom of overrepresented frameworks
    selected.sort(key=lambda x: x[0], reverse=True)

    if len(selected) > TARGET_TOTAL:
        # Calculate hard caps (with 10% slack above target)
        fw_caps = {
            fw: int(TARGET_TOTAL * pct * 1.1)
            for fw, pct in FRAMEWORK_TARGETS.items()
        }

        # Trim lowest-scoring repos from frameworks that exceed their cap
        keep = []
        fw_kept = Counter()
        for s, bd, r in selected:
            fw = get_framework(r)
            cap = fw_caps.get(fw, TARGET_TOTAL)
            if fw_kept[fw] < cap:
                keep.append((s, bd, r))
                fw_kept[fw] += 1
            # else: trimmed (over cap)

        selected = keep

    # Final trim to target
    selected.sort(key=lambda x: x[0], reverse=True)
    selected = selected[:TARGET_TOTAL]

    print(f"\nPhase 3: {len(selected)} repos after coverage + caps")
    return selected


# ── PHASE 4: OUTPUT ─────────────────────────────────────────────────────────

def phase4_output(selected, output_dir):
    """Write output files and print summary.

    A/B split is framework-interleaved so both machines get balanced work.
    """
    output_dir = Path(output_dir)

    # Build output records
    records = []
    for score, breakdown, repo in selected:
        raw_ac = get_raw_agent_count(repo)
        ac = get_agent_count(repo)
        records.append({
            "repo_url": f"https://github.com/{repo.get('repo_full_name', '')}",
            "repo_full_name": repo.get("repo_full_name", ""),
            "primary_framework": get_framework(repo),
            "agent_count": ac,
            "agent_count_raw": raw_ac,
            "agent_count_capped": raw_ac > AGENT_COUNT_CAP,
            "composite_score": score,
            "score_breakdown": breakdown,
            "finding_rules": repo.get("finding_rules", []),
            "finding_count_non_telemetry": len(get_non_telemetry_findings(repo)),
            "deployment_score": repo.get("deployment_signals", {}).get("deployment_score", 0),
            "has_guardrails": repo.get("has_any_guardrails", False),
            "graph_topology": repo.get("graph_topology_metrics", {}),
        })

    # Write full list
    full_path = output_dir / "selected_repos_v2.jsonl"
    with open(full_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # A/B split: interleave by framework so both machines get balanced work
    # Round-robin within each framework, alternating A/B
    by_fw = defaultdict(list)
    for r in records:
        by_fw[r["primary_framework"]].append(r)

    a_records, b_records = [], []
    for fw in sorted(by_fw.keys()):
        fw_list = by_fw[fw]
        for i, r in enumerate(fw_list):
            if i % 2 == 0:
                a_records.append(r)
            else:
                b_records.append(r)

    a_path = output_dir / "selected_repos_v2_A.jsonl"
    b_path = output_dir / "selected_repos_v2_B.jsonl"
    with open(a_path, "w") as f:
        for r in a_records:
            f.write(json.dumps(r) + "\n")
    with open(b_path, "w") as f:
        for r in b_records:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(records)} repos to {full_path}")
    print(f"Wrote {len(a_records)} repos to {a_path}")
    print(f"Wrote {len(b_records)} repos to {b_path}")

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
        print(f"  {fw:15s}: {count:5d} ({pct:.1f}%, target ~{target:.0f}%)")

    # Finding prefix coverage
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

    # Agent count distribution (bucketed for readability)
    ac_dist = Counter(r["agent_count"] for r in records)
    capped_in_output = sum(1 for r in records if r["agent_count_capped"])
    print(f"\nAgent count distribution:")
    buckets = [
        ("2 agents", lambda ac: ac == 2),
        ("3-7 agents", lambda ac: 3 <= ac <= 7),
        ("8-15 agents", lambda ac: 8 <= ac <= 15),
        ("16-30 agents", lambda ac: 16 <= ac <= 30),
        ("31-50 (capped)", lambda ac: 31 <= ac <= 50),
    ]
    for label, pred in buckets:
        count = sum(c for ac, c in ac_dist.items() if pred(ac))
        pct = count / len(records) * 100
        print(f"  {label:20s}: {count:5d} ({pct:.1f}%)")
    print(f"  Repos with capped agent count: {capped_in_output}")

    # Telemetry-only
    telem_only = sum(1 for r in records if all(
        f.startswith("TELEMETRY") for f in r["finding_rules"]
    ))
    print(f"\nTelemetry-only repos: {telem_only}")

    # Guardrails
    guard_count = sum(1 for r in records if r["has_guardrails"])
    print(f"Repos with guardrails: {guard_count}")

    # Score distribution
    scores = [r["composite_score"] for r in records]
    scores.sort(reverse=True)
    print(f"\nScore distribution:")
    print(f"  Max: {scores[0]}  P90: {scores[len(scores)//10]}  "
          f"Median: {scores[len(scores)//2]}  P10: {scores[9*len(scores)//10]}  "
          f"Min: {scores[-1]}")

    # Top 10
    print(f"\nTop 10 repos by composite score:")
    for r in records[:10]:
        cap_flag = " [CAPPED]" if r["agent_count_capped"] else ""
        print(f"  {r['composite_score']:3d}  {r['primary_framework']:12s}  "
              f"{r['agent_count']}a  {r['repo_full_name']}{cap_flag}")

    # A/B balance check
    print(f"\nA/B split balance:")
    for label, recs in [("Machine A", a_records), ("Machine B", b_records)]:
        fw_c = Counter(r["primary_framework"] for r in recs)
        parts = [f"{fw}={c}" for fw, c in fw_c.most_common()]
        print(f"  {label}: {len(recs)} repos — {', '.join(parts)}")


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Select repos for behavioral scan")
    parser.add_argument("scan_results", help="Path to scan_results.jsonl")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same as input)")
    parser.add_argument("--target", type=int, default=TARGET_TOTAL,
                        help=f"Target number of repos (default: {TARGET_TOTAL})")
    args = parser.parse_args()

    if args.target != TARGET_TOTAL:
        TARGET_TOTAL = args.target
        print(f"Target overridden to {TARGET_TOTAL}")

    output_dir = args.output_dir or str(Path(args.scan_results).parent)

    all_repos = phase1_filter(args.scan_results)
    scored = phase2_score(all_repos)
    final = phase3_coverage(scored, all_repos)
    phase4_output(final, output_dir)
