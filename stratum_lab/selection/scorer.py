"""Scoring functions for repo selection.

Computes three scoring dimensions from structural scan data:
  - Structural Value (0-40): graph complexity and observability interest
  - Archetype Rarity (0-30): diversity bonus for underrepresented archetypes
  - Runnability Likelihood (0-30): likelihood the repo can be executed
"""

from __future__ import annotations

from typing import Any

from stratum_lab.config import SUPPORTED_FRAMEWORKS

# ---------------------------------------------------------------------------
# Archetype catalogue
# ---------------------------------------------------------------------------
# Maps archetype_id -> archetype_name.  The structural scan JSON is expected
# to carry an ``archetype_id`` field assigned during the stratum-cli scan.
ARCHETYPES: dict[int, str] = {
    1: "single_agent_tool_use",
    2: "sequential_pipeline",
    3: "parallel_fan_out",
    4: "supervisor_worker",
    5: "debate_consensus",
    6: "reflection_loop",
    7: "hub_and_spoke_shared_state",
    8: "hierarchical_delegation",
    9: "market_auction",
    10: "blackboard_architecture",
    11: "human_in_the_loop",
    12: "guardrail_gated_pipeline",
}

# Standard frameworks that are straightforward to run
STANDARD_FRAMEWORKS = {"crewai", "langgraph", "autogen", "langchain"}


# ---------------------------------------------------------------------------
# Dimension 1 — Structural Value (0-40)
# ---------------------------------------------------------------------------

def compute_structural_value(repo: dict[str, Any]) -> float:
    """Return the structural value score (0-40) for a repo scan dict.

    The score rewards graph complexity, taxonomy coverage, delegation depth,
    shared-state conflicts, feedback loops, trust-boundary crossings, and
    capability diversity — all properties that make a repo interesting from
    an observability research perspective.
    """
    agent_count = len(repo.get("agent_definitions", []))
    total_edges = len(repo.get("graph_edges", []))
    taxonomy_preconditions_count = len(repo.get("taxonomy_preconditions", []))

    risk = repo.get("risk_surface", {})
    delegation_depth = risk.get("max_delegation_depth", 0)
    shared_state_conflicts = risk.get("shared_state_conflict_count", 0)
    feedback_loops_detected = risk.get("feedback_loop_count", 0)
    trust_boundary_crossings = risk.get("trust_boundary_crossing_count", 0)

    # Distinct capability types across all agents
    capability_types: set[str] = set()
    for agent in repo.get("agent_definitions", []):
        for tool in agent.get("tool_names", []):
            capability_types.add(tool)
    distinct_capability_types = len(capability_types)

    score = (
        min(agent_count, 10) * 2
        + min(total_edges, 30) * 0.5
        + min(taxonomy_preconditions_count, 10) * 1.5
        + min(delegation_depth, 5) * 2
        + min(shared_state_conflicts, 5) * 2
        + (1 if feedback_loops_detected > 0 else 0) * 3
        + (1 if trust_boundary_crossings > 2 else 0) * 2
        + min(distinct_capability_types, 8) * 0.5
    )
    return float(score)


# ---------------------------------------------------------------------------
# Dimension 2 — Archetype Rarity (0-30)
# ---------------------------------------------------------------------------

def compute_archetype_rarity(
    repo: dict[str, Any],
    archetype_counts: dict[int, int],
    selection_target: int,
) -> float:
    """Return the archetype rarity score (0-30).

    Repos belonging to an under-selected archetype receive a higher score,
    encouraging diversity across the final selection.

    Parameters
    ----------
    repo:
        Structural scan dict.  Must contain ``archetype_id``.
    archetype_counts:
        Running count of how many repos have already been selected for each
        archetype_id.
    selection_target:
        Total number of repos we aim to select (e.g. 1500).
    """
    archetype_id = repo.get("archetype_id", 0)
    num_archetypes = max(len(ARCHETYPES), 1)
    target_per_archetype = selection_target / num_archetypes
    current_count = archetype_counts.get(archetype_id, 0)

    if target_per_archetype <= 0:
        return 0.0

    ratio = current_count / target_per_archetype
    score = 30.0 * (1.0 - ratio)
    return max(score, 0.0)


# ---------------------------------------------------------------------------
# Dimension 3 — Runnability Likelihood (0-30)
# ---------------------------------------------------------------------------

def compute_runnability(repo: dict[str, Any]) -> float:
    """Return the runnability likelihood score (0-30).

    Rewards repos that have an entry point, requirements file, standard
    framework, usage docs, and minimal infrastructure dependencies.
    """
    has_entry_point = 1 if repo.get("detected_entry_point") else 0
    has_requirements_file = 1 if repo.get("detected_requirements") else 0

    frameworks = [f.lower() for f in repo.get("detected_frameworks", [])]
    uses_standard_framework = 1 if any(f in STANDARD_FRAMEWORKS for f in frameworks) else 0

    has_readme_with_usage = 1 if repo.get("has_readme_with_usage", False) else 0
    no_docker_required = 1 if not repo.get("requires_docker", False) else 0
    no_database_required = 1 if not repo.get("requires_database", False) else 0
    recent_commit = 1 if repo.get("recent_commit", False) else 0

    score = (
        10 * has_entry_point
        + 5 * has_requirements_file
        + 5 * uses_standard_framework
        + 3 * has_readme_with_usage
        + 3 * no_docker_required
        + 2 * no_database_required
        + 2 * recent_commit
    )
    return float(score)


# ---------------------------------------------------------------------------
# Primary framework resolution
# ---------------------------------------------------------------------------

def resolve_primary_framework(repo: dict[str, Any]) -> str:
    """Pick the primary framework label for a repo.

    Uses the first detected framework that is in SUPPORTED_FRAMEWORKS; falls
    back to ``"custom"`` if none match.
    """
    for fw in repo.get("detected_frameworks", []):
        if fw.lower() in SUPPORTED_FRAMEWORKS:
            return fw.lower()
    return "custom"


# ---------------------------------------------------------------------------
# Complexity estimation
# ---------------------------------------------------------------------------

def estimate_complexity(repo: dict[str, Any]) -> str:
    """Return ``"low"``, ``"medium"``, or ``"high"`` based on agent count and edges."""
    agent_count = len(repo.get("agent_definitions", []))
    edge_count = len(repo.get("graph_edges", []))
    if agent_count <= 2 and edge_count <= 5:
        return "low"
    if agent_count <= 6 and edge_count <= 20:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Convenience: score a single repo
# ---------------------------------------------------------------------------

def score_repo(
    repo: dict[str, Any],
    archetype_counts: dict[int, int],
    selection_target: int,
) -> dict[str, Any]:
    """Score a single repo and return the full selection record dict.

    The returned dict matches the ``RepoSelection`` schema plus a
    ``total_score`` convenience field.
    """
    structural_value = compute_structural_value(repo)
    archetype_rarity = compute_archetype_rarity(repo, archetype_counts, selection_target)
    runnability_score = compute_runnability(repo)
    total_score = structural_value + archetype_rarity + runnability_score

    archetype_id = repo.get("archetype_id", 0)
    archetype_name = ARCHETYPES.get(archetype_id, "unknown")
    framework = resolve_primary_framework(repo)

    return {
        "repo_id": repo.get("repo_id", ""),
        "repo_url": repo.get("repo_url", ""),
        "framework": framework,
        "selection_score": round(total_score, 2),
        "structural_value": round(structural_value, 2),
        "archetype_id": archetype_id,
        "archetype_name": archetype_name,
        "archetype_rarity": round(archetype_rarity, 2),
        "runnability_score": round(runnability_score, 2),
        "agent_count": len(repo.get("agent_definitions", [])),
        "taxonomy_preconditions": repo.get("taxonomy_preconditions", []),
        "detected_entry_point": repo.get("detected_entry_point", ""),
        "detected_requirements": repo.get("detected_requirements", ""),
        "estimated_complexity": estimate_complexity(repo),
    }
