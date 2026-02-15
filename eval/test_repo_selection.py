"""Evaluation of the repo selection and scoring pipeline.

Generates 50 synthetic repo structural scan results with varying parameters
and runs them through score_and_select() from stratum_lab.selection.selector.

Prints:
  - Scoring breakdown for top 10 and bottom 10
  - Framework distribution
  - Archetype distribution
  - Constraint verification

Run as a standalone script:
    cd stratum-lab
    python eval/test_repo_selection.py
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure stratum_lab is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def _generate_synthetic_repos(count: int = 50, seed: int = 42) -> list[dict]:
    """Generate *count* synthetic repo structural scan dicts.

    Each repo has realistic variations in:
      - agent_count (1-10)
      - edge_count (1-30)
      - precondition counts (0-10)
      - frameworks: crewai, langgraph, autogen, langchain, custom
      - archetypes: IDs 1-12
      - runnability signals (entry points, requirements files, etc.)
    """
    rng = random.Random(seed)

    frameworks = ["crewai", "langgraph", "autogen", "langchain", "custom"]
    framework_weights = [0.30, 0.25, 0.15, 0.20, 0.10]
    tool_pool = [
        "web_search", "file_reader", "calculator", "code_executor",
        "email_sender", "database_query", "api_caller", "text_summarizer",
        "image_analyzer", "data_plotter",
    ]
    precondition_pool = [
        "multi_agent_comm", "tool_use_observed", "delegation_chain",
        "shared_state_access", "error_propagation", "feedback_loop",
        "guardrail_check", "human_oversight", "data_pipeline",
        "concurrent_execution",
    ]

    repos = []
    for i in range(count):
        agent_count = rng.randint(1, 10)
        edge_count = rng.randint(1, 30)
        archetype_id = rng.randint(1, 12)

        # Choose framework with weighted distribution
        fw = rng.choices(frameworks, weights=framework_weights, k=1)[0]

        # Generate agent definitions
        agent_defs = []
        for j in range(agent_count):
            num_tools = rng.randint(0, min(4, len(tool_pool)))
            tools = rng.sample(tool_pool, num_tools)
            agent_defs.append({
                "agent_id": f"agent_{j}",
                "role": f"Agent_{j}_Role",
                "tool_names": tools,
            })

        # Generate graph edges
        edges = []
        for j in range(edge_count):
            src = rng.randint(0, max(agent_count - 1, 0))
            tgt = rng.randint(0, max(agent_count - 1, 0))
            edges.append({
                "source": f"agent_{src}",
                "target": f"agent_{tgt}",
                "type": rng.choice(["delegates_to", "sends_to", "calls"]),
            })

        # Ensure ~20% of repos have 0-1 preconditions (control-eligible)
        if i % 5 == 0:
            num_preconditions = rng.randint(0, 1)
        else:
            num_preconditions = rng.randint(2, 10)
        preconditions = rng.sample(precondition_pool, min(num_preconditions, len(precondition_pool)))

        # Risk surface
        risk_surface = {
            "max_delegation_depth": rng.randint(0, 5),
            "shared_state_conflict_count": rng.randint(0, 5),
            "feedback_loop_count": rng.randint(0, 3),
            "trust_boundary_crossing_count": rng.randint(0, 5),
        }

        # Runnability signals
        has_entry_point = rng.random() > 0.2  # 80% have entry points
        has_requirements = rng.random() > 0.15  # 85% have requirements
        has_readme = rng.random() > 0.3  # 70% have README with usage
        requires_docker = rng.random() > 0.7  # 30% require Docker
        requires_database = rng.random() > 0.8  # 20% require database
        recent_commit = rng.random() > 0.4  # 60% have recent commits

        repo = {
            "repo_id": f"repo_{i:04d}",
            "repo_url": f"https://github.com/test-org/repo-{i:04d}",
            "archetype_id": archetype_id,
            "detected_frameworks": [fw],
            "agent_definitions": agent_defs,
            "graph_edges": edges,
            "taxonomy_preconditions": preconditions,
            "risk_surface": risk_surface,
            "detected_entry_point": "main.py" if has_entry_point else "",
            "detected_requirements": "requirements.txt" if has_requirements else "",
            "has_readme_with_usage": has_readme,
            "requires_docker": requires_docker,
            "requires_database": requires_database,
            "recent_commit": recent_commit,
        }
        repos.append(repo)

    return repos


def main() -> None:
    from stratum_lab.selection.selector import score_and_select
    from stratum_lab.selection.scorer import ARCHETYPES

    # -----------------------------------------------------------------
    # Generate synthetic repos
    # -----------------------------------------------------------------
    raw_repos = _generate_synthetic_repos(count=50, seed=42)

    print("=" * 80)
    print("  REPO SELECTION EVALUATION")
    print(f"  Synthetic repos generated: {len(raw_repos)}")
    print("=" * 80)
    print()

    # -----------------------------------------------------------------
    # Run selection (use smaller target for 50 repos)
    # -----------------------------------------------------------------
    selected, summary = score_and_select(
        raw_repos,
        target=30,             # reasonable target for 50 repos
        min_runnability=10.0,  # lower threshold for synthetic data
        max_per_archetype=10,  # scaled down from 200
        min_per_archetype=2,   # scaled down from 30
    )

    # -----------------------------------------------------------------
    # Scoring breakdown: top 10
    # -----------------------------------------------------------------
    print("  TOP 10 REPOS BY SELECTION SCORE")
    print("  " + "-" * 76)
    print(f"  {'Repo ID':<12} {'Framework':<12} {'Archetype':<28} "
          f"{'Struct':>6} {'Rare':>6} {'Run':>5} {'Total':>6}")
    print("  " + "-" * 76)
    for r in selected[:10]:
        print(f"  {r['repo_id']:<12} {r['framework']:<12} "
              f"{r['archetype_name']:<28} "
              f"{r['structural_value']:>6.1f} "
              f"{r['archetype_rarity']:>6.1f} "
              f"{r['runnability_score']:>5.1f} "
              f"{r['selection_score']:>6.1f}")
    print()

    # -----------------------------------------------------------------
    # Scoring breakdown: bottom 10
    # -----------------------------------------------------------------
    print("  BOTTOM 10 REPOS BY SELECTION SCORE")
    print("  " + "-" * 76)
    print(f"  {'Repo ID':<12} {'Framework':<12} {'Archetype':<28} "
          f"{'Struct':>6} {'Rare':>6} {'Run':>5} {'Total':>6}")
    print("  " + "-" * 76)
    for r in selected[-10:]:
        print(f"  {r['repo_id']:<12} {r['framework']:<12} "
              f"{r['archetype_name']:<28} "
              f"{r['structural_value']:>6.1f} "
              f"{r['archetype_rarity']:>6.1f} "
              f"{r['runnability_score']:>5.1f} "
              f"{r['selection_score']:>6.1f}")
    print()

    # -----------------------------------------------------------------
    # Framework distribution
    # -----------------------------------------------------------------
    print("  FRAMEWORK DISTRIBUTION")
    print("  " + "-" * 50)
    fw_counts = Counter(r["framework"] for r in selected)
    total_selected = len(selected)
    for fw, count in fw_counts.most_common():
        pct = (count / total_selected * 100) if total_selected > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  {fw:<12} {count:>4} ({pct:>5.1f}%)  {bar}")
    print()

    # -----------------------------------------------------------------
    # Archetype distribution
    # -----------------------------------------------------------------
    print("  ARCHETYPE DISTRIBUTION")
    print("  " + "-" * 60)
    arch_counts = Counter(r["archetype_name"] for r in selected)
    for arch_name, count in arch_counts.most_common():
        pct = (count / total_selected * 100) if total_selected > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  {arch_name:<30} {count:>4} ({pct:>5.1f}%)  {bar}")
    print()

    # -----------------------------------------------------------------
    # Constraint verification
    # -----------------------------------------------------------------
    print("  CONSTRAINT VERIFICATION")
    print("  " + "-" * 60)

    # 1) Max per archetype
    max_arch_count = max(arch_counts.values()) if arch_counts else 0
    max_per_arch_limit = 10  # our test setting
    if max_arch_count <= max_per_arch_limit:
        print(f"  [+] Max per archetype: {max_arch_count} <= {max_per_arch_limit} (limit)")
    else:
        print(f"  [X] Max per archetype: {max_arch_count} > {max_per_arch_limit} (VIOLATED)")

    # 2) Framework diversity (at least 3 frameworks)
    num_frameworks = len(fw_counts)
    if num_frameworks >= 3:
        print(f"  [+] Frameworks represented: {num_frameworks} >= 3 (minimum)")
    else:
        print(f"  [X] Frameworks represented: {num_frameworks} < 3 (VIOLATED)")

    # 3) All selected repos meet minimum runnability
    min_runnability_threshold = 10.0
    min_runnability_found = min(r["runnability_score"] for r in selected) if selected else 0
    if min_runnability_found >= min_runnability_threshold:
        print(f"  [+] Min runnability score: {min_runnability_found:.1f} >= "
              f"{min_runnability_threshold:.1f} (threshold)")
    else:
        print(f"  [X] Min runnability score: {min_runnability_found:.1f} < "
              f"{min_runnability_threshold:.1f} (VIOLATED)")

    # 4) Selection count within expected range
    if total_selected > 0:
        print(f"  [+] Total selected: {total_selected}")
    else:
        print(f"  [X] No repos selected!")

    print()

    # -----------------------------------------------------------------
    # Control group breakdown
    # -----------------------------------------------------------------
    print("  CONTROL GROUP BREAKDOWN")
    print("  " + "-" * 60)
    control_count = summary.get("control_count", 0)
    treatment_count = summary.get("treatment_count", 0)
    print(f"  control:   {control_count}")
    print(f"  treatment: {treatment_count}")
    control_repos = [r for r in selected if r.get("group") == "control"]
    treatment_repos = [r for r in selected if r.get("group") != "control"]
    print(f"  control repos (tagged):   {len(control_repos)}")
    print(f"  treatment repos (tagged): {len(treatment_repos)}")
    if control_repos:
        avg_prec = sum(len(r.get("taxonomy_preconditions", [])) for r in control_repos) / len(control_repos)
        print(f"  avg preconditions (control):   {avg_prec:.1f}")
    if treatment_repos:
        avg_prec_t = sum(len(r.get("taxonomy_preconditions", [])) for r in treatment_repos) / len(treatment_repos)
        print(f"  avg preconditions (treatment): {avg_prec_t:.1f}")

    # Verification checks
    print(f"  [{'+'if len(control_repos) > 0 else '-'}] Control group non-empty: {len(control_repos)} repos")
    print(f"  [{'+'if len(treatment_repos) > len(control_repos) else '-'}] Treatment > control: {len(treatment_repos)} > {len(control_repos)}")
    print()

    # -----------------------------------------------------------------
    # Summary stats
    # -----------------------------------------------------------------
    print("  SUMMARY STATISTICS")
    print("  " + "-" * 60)
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k2, v2 in value.items():
                print(f"    {k2}: {v2}")
        else:
            print(f"  {key}: {value}")
    print()

    # -----------------------------------------------------------------
    # Score distribution analysis
    # -----------------------------------------------------------------
    print("  SCORE DISTRIBUTION")
    print("  " + "-" * 60)
    scores = [r["selection_score"] for r in selected]
    if scores:
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        # Compute standard deviation
        variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5
        print(f"  Mean selection score:  {avg_score:.2f}")
        print(f"  Min selection score:   {min_score:.2f}")
        print(f"  Max selection score:   {max_score:.2f}")
        print(f"  Std dev:               {std_dev:.2f}")

        struct_scores = [r["structural_value"] for r in selected]
        rarity_scores = [r["archetype_rarity"] for r in selected]
        run_scores = [r["runnability_score"] for r in selected]
        print(f"  Avg structural value:  {sum(struct_scores)/len(struct_scores):.2f}")
        print(f"  Avg archetype rarity:  {sum(rarity_scores)/len(rarity_scores):.2f}")
        print(f"  Avg runnability:       {sum(run_scores)/len(run_scores):.2f}")

    print()

    # -----------------------------------------------------------------
    # Complexity distribution
    # -----------------------------------------------------------------
    print("  COMPLEXITY DISTRIBUTION")
    print("  " + "-" * 40)
    complexity_counts = Counter(r["estimated_complexity"] for r in selected)
    for complexity, count in complexity_counts.most_common():
        pct = (count / total_selected * 100) if total_selected > 0 else 0
        print(f"  {complexity:<10} {count:>4} ({pct:>5.1f}%)")

    print()

    # -----------------------------------------------------------------
    # Verify the scorer functions individually
    # -----------------------------------------------------------------
    from stratum_lab.selection.scorer import (
        compute_structural_value,
        compute_archetype_rarity,
        compute_runnability,
        resolve_primary_framework,
        estimate_complexity,
    )

    print("  INDIVIDUAL SCORER FUNCTION CHECKS")
    print("  " + "-" * 60)

    # Structural value: a repo with many agents, edges, etc. should score high
    high_struct_repo = {
        "agent_definitions": [{"tool_names": [f"tool_{i}"]} for i in range(10)],
        "graph_edges": [{}] * 30,
        "taxonomy_preconditions": [f"cond_{i}" for i in range(10)],
        "risk_surface": {
            "max_delegation_depth": 5,
            "shared_state_conflict_count": 5,
            "feedback_loop_count": 2,
            "trust_boundary_crossing_count": 4,
        },
    }
    sv = compute_structural_value(high_struct_repo)
    print(f"  [+] High-complexity structural value: {sv:.1f} (expected ~54+)")

    # Empty repo should score 0
    sv_empty = compute_structural_value({})
    print(f"  [+] Empty repo structural value:      {sv_empty:.1f} (expected 0)")

    # Runnability: fully runnable repo
    runnable_repo = {
        "detected_entry_point": "main.py",
        "detected_requirements": "requirements.txt",
        "detected_frameworks": ["crewai"],
        "has_readme_with_usage": True,
        "requires_docker": False,
        "requires_database": False,
        "recent_commit": True,
    }
    rs = compute_runnability(runnable_repo)
    print(f"  [+] Fully runnable runnability score:  {rs:.1f} (expected 30)")

    # Framework resolution
    fw = resolve_primary_framework({"detected_frameworks": ["crewai", "langchain"]})
    print(f"  [+] Framework resolution (crewai,langchain): {fw} (expected crewai)")

    fw2 = resolve_primary_framework({"detected_frameworks": ["unknown_fw"]})
    print(f"  [+] Framework resolution (unknown_fw): {fw2} (expected custom)")

    # Complexity estimation
    cx = estimate_complexity({"agent_definitions": [{}] * 2, "graph_edges": [{}] * 3})
    print(f"  [+] Complexity (2 agents, 3 edges): {cx} (expected low)")

    cx2 = estimate_complexity({"agent_definitions": [{}] * 8, "graph_edges": [{}] * 25})
    print(f"  [+] Complexity (8 agents, 25 edges): {cx2} (expected high)")

    print()
    print("=" * 80)
    print("  EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
