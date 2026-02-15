"""Cross-pattern interaction analysis.

Computes pairwise co-occurrence rates and tests whether
co-occurring preconditions have multiplicative risk effects.
"""

from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List


def compute_interaction_matrix(enriched_repos: List[Dict]) -> Dict:
    """Compute pairwise interaction effects between preconditions.

    For each pair (A, B):
    - P(fail | A only): failure rate with A but not B
    - P(fail | B only): failure rate with B but not A
    - P(fail | A AND B): failure rate with both
    - P(fail | neither): control failure rate
    - interaction_effect: ratio of observed(A+B) vs expected(A) * expected(B)
    """
    repo_preconditions: Dict[str, set] = {}
    repo_failed: Dict[str, bool] = {}

    for repo in enriched_repos:
        repo_id = repo.get("repo_id", "")
        # Get confirmed preconditions from taxonomy
        preconds = set(repo.get("taxonomy_preconditions", []))
        repo_preconditions[repo_id] = preconds

        # Determine if repo had runtime failure
        had_failure = False
        for nid, ndata in repo.get("nodes", {}).items():
            beh = ndata.get("behavioral", {})
            err = beh.get("error_behavior", {})
            if err.get("errors_occurred", 0) > 0:
                had_failure = True
                break
        repo_failed[repo_id] = had_failure

    all_preconditions: set = set()
    for precs in repo_preconditions.values():
        all_preconditions.update(precs)

    interactions = []
    for a, b in combinations(sorted(all_preconditions), 2):
        both = [r for r, p in repo_preconditions.items() if a in p and b in p]
        a_only = [r for r, p in repo_preconditions.items() if a in p and b not in p]
        b_only = [r for r, p in repo_preconditions.items() if b in p and a not in p]
        neither = [r for r, p in repo_preconditions.items() if a not in p and b not in p]

        def fail_rate(repo_ids: list) -> float:
            if not repo_ids:
                return 0.0
            return sum(1 for r in repo_ids if repo_failed.get(r, False)) / len(repo_ids)

        p_both = fail_rate(both)
        p_a_only = fail_rate(a_only)
        p_b_only = fail_rate(b_only)
        p_neither = fail_rate(neither)

        p_expected = max(p_a_only * p_b_only / max(p_neither, 0.01), 0.01)
        interaction_effect = p_both / p_expected if p_expected > 0 else 1.0

        interactions.append({
            "precondition_a": a,
            "precondition_b": b,
            "co_occurrence_count": len(both),
            "p_fail_both": round(p_both, 4),
            "p_fail_a_only": round(p_a_only, 4),
            "p_fail_b_only": round(p_b_only, 4),
            "p_fail_neither": round(p_neither, 4),
            "interaction_effect": round(interaction_effect, 2),
            "synergistic": interaction_effect > 1.5,
            "sample_sizes": {
                "both": len(both),
                "a_only": len(a_only),
                "b_only": len(b_only),
                "neither": len(neither),
            },
        })

    interactions.sort(key=lambda x: x["interaction_effect"], reverse=True)

    return {
        "interactions": interactions,
        "synergistic_pairs": [i for i in interactions if i["synergistic"]],
        "most_dangerous_combination": interactions[0] if interactions else None,
    }
