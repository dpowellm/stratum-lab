"""Dataset quality validation.

Run after all phases complete, before finalizing the knowledge base.
Reports whether the dataset meets statistical requirements for each claim.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List


def wilson_ci_width(n: int, p: float = 0.5) -> float:
    """Wilson score CI width at 95% confidence for proportion p with sample n."""
    if n <= 0:
        return 1.0
    z = 1.96
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return 2 * spread


def validate_dataset_quality(
    enriched_graphs: List[Dict[str, Any]],
    run_records: List[Dict[str, Any]],
    knowledge_base: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate dataset meets quality requirements.

    Returns
    -------
    Dict with per-claim quality assessment and overall verdict.
    """
    total_repos = len(enriched_graphs)
    total_runs = len(run_records)

    # Count repos per precondition
    precondition_counts: Dict[str, int] = {}
    for graph in enriched_graphs:
        confirmed = graph.get("confirmed_preconditions", [])
        if not confirmed:
            confirmed = graph.get("taxonomy_preconditions", [])
        for pc in confirmed:
            precondition_counts[pc] = precondition_counts.get(pc, 0) + 1

    # Count repos WITHOUT each precondition (control)
    control_counts: Dict[str, int] = {}
    for pc in precondition_counts:
        control_counts[pc] = total_repos - precondition_counts[pc]

    # Check semantic data availability
    repos_with_semantic = sum(
        1 for g in enriched_graphs
        if g.get("semantic_lineage", {}).get("total_handoffs", 0) > 0
    )

    # Count runs with usable behavioral data
    usable_runs = sum(
        1 for r in run_records
        if r.get("status") in ("SUCCESS", "PARTIAL_SUCCESS")
        and r.get("event_count", r.get("total_events", 0)) >= 5
    )

    # Per-claim quality
    claims: Dict[str, Dict[str, Any]] = {}

    # Claim 1: Semantic chain risk
    n_semantic_treatment = precondition_counts.get("unvalidated_semantic_chain", 0)
    n_semantic_control = control_counts.get("unvalidated_semantic_chain", 0)
    claims["semantic_chain_risk"] = {
        "treatment_n": n_semantic_treatment,
        "control_n": n_semantic_control,
        "ci_width_treatment": wilson_ci_width(n_semantic_treatment),
        "ci_width_control": wilson_ci_width(n_semantic_control),
        "semantic_data_available": repos_with_semantic,
        "sufficient": (
            n_semantic_treatment >= 60
            and n_semantic_control >= 20
            and repos_with_semantic >= 50
        ),
    }

    # Claim 2: Shared state races
    n_shared = precondition_counts.get("shared_state_no_arbitration", 0)
    claims["shared_state_risk"] = {
        "treatment_n": n_shared,
        "control_n": control_counts.get("shared_state_no_arbitration", 0),
        "ci_width": wilson_ci_width(n_shared),
        "sufficient": n_shared >= 40,
    }

    # Claim 3: Relative risk (needs BOTH treatment AND control)
    relative_risk_ready = 0
    for pc, n_treatment in precondition_counts.items():
        n_control = control_counts.get(pc, 0)
        if n_treatment >= 20 and n_control >= 10:
            relative_risk_ready += 1

    claims["relative_risk"] = {
        "preconditions_with_sufficient_data": relative_risk_ready,
        "total_preconditions": len(precondition_counts),
        "sufficient": relative_risk_ready >= 10,
    }

    # Overall
    all_sufficient = all(c.get("sufficient", False) for c in claims.values())

    quality_report: Dict[str, Any] = {
        "total_repos": total_repos,
        "total_runs": total_runs,
        "usable_runs": usable_runs,
        "usable_run_rate": usable_runs / max(total_runs, 1),
        "repos_with_semantic_data": repos_with_semantic,
        "unique_preconditions_observed": len(precondition_counts),
        "per_claim_quality": claims,
        "all_claims_supported": all_sufficient,
        "verdict": "DATASET READY" if all_sufficient else "DATASET NEEDS MORE DATA",
        "recommendations": _generate_recommendations(
            claims, precondition_counts, control_counts,
        ),
    }

    return quality_report


def _generate_recommendations(
    claims: Dict[str, Dict[str, Any]],
    pc_counts: Dict[str, int],
    ctrl_counts: Dict[str, int],
) -> List[str]:
    recs: List[str] = []

    for claim_name, claim_data in claims.items():
        if not claim_data.get("sufficient", True):
            if "treatment_n" in claim_data and claim_data["treatment_n"] < 40:
                recs.append(
                    f"{claim_name}: need {40 - claim_data['treatment_n']} more treatment repos"
                )
            if "control_n" in claim_data and claim_data["control_n"] < 15:
                recs.append(
                    f"{claim_name}: need {15 - claim_data['control_n']} more control repos"
                )

    return recs
