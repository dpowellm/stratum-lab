"""Input schema for repo selection.

Takes reliability scanner output (per-repo JSON) as the selection source.
Each repo has: graph topology, finding list, control inventory,
framework detection, and structural metrics.
"""

from __future__ import annotations

from typing import Any

# Required fields from reliability scanner output
REQUIRED_FIELDS = {
    "repo_full_name",           # "owner/repo"
    "graph",                    # {nodes: [...], edges: [...]}
    "findings",                 # [{finding_id, severity, ...}, ...]
    "control_inventory",        # {present_controls: [...], absent_controls: [...]}
    "framework",                # "crewai" | "langgraph" | "autogen" | ...
    "agent_count",              # int
    "edge_count",               # int
}

# Fields used for selection scoring
SCORING_FIELDS = {
    "archetype",                # from structural classification
    "graph_complexity",         # low/medium/high
    "finding_ids",              # list of STRAT-XX-NNN finding IDs
    "control_configuration",    # summary of which controls are present/absent
    "xcomp_findings",           # STRAT-XCOMP findings (security × reliability overlap)
    "has_entry_point",          # bool — structural scanner detected runnable entry
    "has_requirements",         # bool
    "estimated_runnability",    # 0-1 from structural signals
}


def validate_selection_input(repo: dict) -> tuple[bool, list[str]]:
    """Validate a repo dict has the fields needed for selection."""
    missing = REQUIRED_FIELDS - set(repo.keys())
    return len(missing) == 0, list(missing)
