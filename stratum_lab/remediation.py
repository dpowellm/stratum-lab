"""Remediation mining algorithms and statistical tests for stratum-lab.

Contains partition logic, chi-squared and t-test pattern differentials,
topology-conditional remediation, cross-pattern interactions, and priority scoring.
"""
from __future__ import annotations

import math

# Try scipy; fall back to manual chi-squared
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Manual chi-squared fallback
# ---------------------------------------------------------------------------

def chi2_manual(table: list[list[int]]) -> tuple[float, float]:
    """Manual chi-squared calculation without scipy.

    Args:
        table: 2x2 contingency table [[a, b], [c, d]].

    Returns:
        (chi2_statistic, p_value_approximate).
    """
    a, b = table[0]
    c, d = table[1]
    n = a + b + c + d
    if n == 0:
        return 0.0, 1.0

    row1 = a + b
    row2 = c + d
    col1 = a + c
    col2 = b + d

    expected = [
        [row1 * col1 / n, row1 * col2 / n],
        [row2 * col1 / n, row2 * col2 / n],
    ]

    chi2 = 0.0
    for i in range(2):
        for j in range(2):
            if expected[i][j] > 0:
                chi2 += (table[i][j] - expected[i][j]) ** 2 / expected[i][j]

    # Approximate p-value using chi2 with 1 dof
    # Using survival function approximation
    if chi2 <= 0:
        return 0.0, 1.0
    try:
        p = math.exp(-chi2 / 2)
    except OverflowError:
        p = 0.0
    return chi2, p


def chi2_contingency(table: list[list[int]]) -> tuple[float, float]:
    """Chi-squared test with scipy fallback."""
    if HAS_SCIPY:
        try:
            chi2, p, _, _ = scipy_stats.chi2_contingency(table)
            return chi2, p
        except ValueError:
            return 0.0, 1.0
    return chi2_manual(table)


def ttest_ind(vals1: list[float], vals2: list[float]) -> tuple[float, float]:
    """Independent t-test with scipy fallback."""
    if HAS_SCIPY:
        try:
            t, p = scipy_stats.ttest_ind(vals1, vals2, equal_var=False)
            return float(t), float(p)
        except ValueError:
            return 0.0, 1.0
    # Manual t-test approximation
    n1, n2 = len(vals1), len(vals2)
    if n1 < 2 or n2 < 2:
        return 0.0, 1.0
    m1 = sum(vals1) / n1
    m2 = sum(vals2) / n2
    s1 = sum((x - m1) ** 2 for x in vals1) / (n1 - 1)
    s2 = sum((x - m2) ** 2 for x in vals2) / (n2 - 1)
    se = math.sqrt(s1 / n1 + s2 / n2) if (s1 / n1 + s2 / n2) > 0 else 1e-10
    t = (m2 - m1) / se
    # Rough p-value approximation
    try:
        p = math.exp(-abs(t))
    except OverflowError:
        p = 0.0
    return t, p


# ---------------------------------------------------------------------------
# Pattern descriptions
# ---------------------------------------------------------------------------

PATTERN_DESCRIPTIONS = {
    "timeout_iteration_guards": "Timeout or iteration limit guards near agent delegation calls",
    "exception_handling_topology": "Exception handling (try/except) with specific catches and logging",
    "output_validation": "Pydantic model, TypedDict, or JSON schema validation on agent outputs",
    "concurrency_controls": "Threading locks, semaphores, or queues for shared state",
    "rate_limiting_backoff": "Rate limiting or exponential backoff on API/LLM calls",
    "input_sanitization": "Input validation on data received from other agents",
    "prompt_constraints": "Prompt template constraints (role boundaries, output format requirements)",
    "timeout_at_boundary": "Timeout guards specifically at delegation boundaries",
    "effective_timeout": "Timeout values <= 120 seconds (effective range)",
    "error_handling_at_boundary": "Exception handling at delegation boundaries",
    "error_logging": "Error logging in exception handlers",
    "output_validation_at_boundary": "Output validation at agent output boundaries",
    "pydantic_validation": "Pydantic model validation on outputs",
    "prompt_role_boundary": "Prompt constraints defining explicit role boundaries",
    "uncertainty_instruction": "Prompt instructions for expressing uncertainty",
}

# Sub-feature mapping
SUB_FEATURE_MAP = {
    "timeout_at_boundary": ("timeout_iteration_guards", "near_boundary_count"),
    "effective_timeout": ("timeout_iteration_guards", "effective_count"),
    "error_handling_at_boundary": ("exception_handling_topology", "near_boundary_count"),
    "error_logging": ("exception_handling_topology", "has_logging_count"),
    "output_validation_at_boundary": ("output_validation", "at_boundary_count"),
    "pydantic_validation": ("output_validation", "pydantic_count"),
    "prompt_role_boundary": ("prompt_constraints", "role_boundary_count"),
    "uncertainty_instruction": ("prompt_constraints", "uncertainty_instruction_count"),
}


# ---------------------------------------------------------------------------
# Step 1: Finding-Conditioned Partitioning
# ---------------------------------------------------------------------------

def check_manifestation(finding_id: str, record: dict) -> bool:
    """Check if the risk identified by finding_id actually manifested."""
    sem = record.get("semantic_analysis", {}).get("aggregate_scores", {})

    manifestation_checks = {
        "STRAT-SD-001": sem.get("trust_elevation_rate", 0) > 0.5,
        "STRAT-HC-001": sem.get("mean_chain_fidelity", 1.0) < 0.3,
        "STRAT-CE-001": sem.get("fabrication_risk_rate", 0) > 0.3,
        "STRAT-SC-001": sem.get("mean_stability_score", 1.0) < 0.3,
        "STRAT-TV-001": any(
            n.get("error_count_across_runs", 0) > 0
            for n in record.get("semantic_analysis", {}).get("nodes", [])
            if isinstance(n, dict) and n.get("vulnerability", {}).get("vulnerability_score", 0) > 0.7
        ),
    }

    if finding_id in manifestation_checks:
        return manifestation_checks[finding_id]

    # For structural findings, check manifestation_observed
    for f in record.get("failure_modes", record.get("findings", [])):
        if isinstance(f, dict) and f.get("finding_id") == finding_id:
            return f.get("manifestation_observed", False)

    return False


def partition_repos(finding_id: str, records: list) -> dict:
    """Partition repos into Q1-Q4 quadrants for a given finding."""
    q1, q2, q3, q4 = [], [], [], []

    for rec in records:
        findings = rec.get("failure_modes", rec.get("findings", []))
        has_finding = any(
            isinstance(f, dict) and f.get("finding_id") == finding_id
            for f in findings
        )
        manifestation = check_manifestation(finding_id, rec)

        if has_finding and manifestation:
            q1.append(rec)
        elif has_finding and not manifestation:
            q2.append(rec)
        elif not has_finding and manifestation:
            q3.append(rec)
        else:
            q4.append(rec)

    return {"q1": q1, "q2": q2, "q3": q3, "q4": q4}


# ---------------------------------------------------------------------------
# Step 2: Defensive Pattern Differential
# ---------------------------------------------------------------------------

def has_pattern(record: dict, pattern_name: str) -> bool:
    """Check if a record has a given defensive pattern."""
    summary = record.get("defensive_patterns", {}).get("summary", {})

    if pattern_name in summary:
        return summary[pattern_name].get("count", 0) > 0

    if pattern_name in SUB_FEATURE_MAP:
        cat, key = SUB_FEATURE_MAP[pattern_name]
        return summary.get(cat, {}).get(key, 0) > 0

    return False


def compute_pattern_differential(
    q1_records: list, q2_records: list, min_n: int, p_threshold: float,
) -> list:
    """Compare defensive pattern prevalence between Q1 (manifested) and Q2 (survived)."""
    if len(q1_records) < min_n or len(q2_records) < min_n:
        return []

    pattern_categories = [
        "timeout_iteration_guards",
        "exception_handling_topology",
        "output_validation",
        "concurrency_controls",
        "rate_limiting_backoff",
        "input_sanitization",
        "prompt_constraints",
    ]

    sub_features = [
        ("timeout_iteration_guards", "near_boundary_count", "timeout_at_boundary"),
        ("timeout_iteration_guards", "effective_count", "effective_timeout"),
        ("exception_handling_topology", "near_boundary_count", "error_handling_at_boundary"),
        ("exception_handling_topology", "has_logging_count", "error_logging"),
        ("output_validation", "at_boundary_count", "output_validation_at_boundary"),
        ("output_validation", "pydantic_count", "pydantic_validation"),
        ("prompt_constraints", "role_boundary_count", "prompt_role_boundary"),
        ("prompt_constraints", "uncertainty_instruction_count", "uncertainty_instruction"),
    ]

    candidates: list[dict] = []

    # Binary feature tests
    for category in pattern_categories:
        q1_has = sum(
            1 for r in q1_records
            if r.get("defensive_patterns", {}).get("summary", {}).get(category, {}).get("count", 0) > 0
        )
        q1_total = len(q1_records)
        q2_has = sum(
            1 for r in q2_records
            if r.get("defensive_patterns", {}).get("summary", {}).get(category, {}).get("count", 0) > 0
        )
        q2_total = len(q2_records)

        q1_rate = q1_has / q1_total if q1_total > 0 else 0
        q2_rate = q2_has / q2_total if q2_total > 0 else 0

        if q1_has + q2_has == 0 or q1_rate == q2_rate:
            continue

        table = [
            [q1_has, q1_total - q1_has],
            [q2_has, q2_total - q2_has],
        ]
        chi2, p_value = chi2_contingency(table)

        if p_value < p_threshold and q2_rate > q1_rate:
            a, b_val = q2_has, q2_total - q2_has
            c, d = q1_has, q1_total - q1_has
            odds_ratio = (a * d) / max(1, b_val * c)

            candidates.append({
                "pattern": category,
                "description": PATTERN_DESCRIPTIONS.get(category, category),
                "q1_prevalence": round(q1_rate, 3),
                "q2_prevalence": round(q2_rate, 3),
                "odds_ratio": round(odds_ratio, 2),
                "p_value": round(p_value, 4),
                "confidence": "high" if p_value < 0.01 else "medium",
                "expected_manifestation_reduction": round(q2_rate - q1_rate, 3),
                "sample_sizes": {"q1": q1_total, "q2": q2_total},
            })

    # Sub-feature tests (continuous)
    for category, sub_key, feature_name in sub_features:
        q1_vals = [
            r.get("defensive_patterns", {}).get("summary", {}).get(category, {}).get(sub_key, 0)
            for r in q1_records
        ]
        q2_vals = [
            r.get("defensive_patterns", {}).get("summary", {}).get(category, {}).get(sub_key, 0)
            for r in q2_records
        ]

        if all(v == 0 for v in q1_vals + q2_vals):
            continue

        t_stat, p_value = ttest_ind(q1_vals, q2_vals)

        q1_mean = sum(q1_vals) / max(1, len(q1_vals))
        q2_mean = sum(q2_vals) / max(1, len(q2_vals))

        if p_value < p_threshold and q2_mean > q1_mean:
            candidates.append({
                "pattern": feature_name,
                "description": PATTERN_DESCRIPTIONS.get(feature_name, feature_name),
                "q1_mean": round(q1_mean, 3),
                "q2_mean": round(q2_mean, 3),
                "t_statistic": round(t_stat, 3),
                "p_value": round(p_value, 4),
                "confidence": "high" if p_value < 0.01 else "medium",
                "sample_sizes": {"q1": len(q1_vals), "q2": len(q2_vals)},
            })

    candidates.sort(key=lambda c: c.get("odds_ratio", c.get("t_statistic", 0)), reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Step 3: Topology-Conditional Remediation
# ---------------------------------------------------------------------------

def get_topology_class(record: dict) -> str:
    """Classify record's topology."""
    topo = record.get("topology_analysis", {})
    node_count = topo.get("node_count", 0)
    edge_count = topo.get("edge_count", 0)
    has_branching = topo.get("has_branching", False)

    if node_count <= 2:
        return "minimal"
    elif not has_branching and edge_count == node_count - 1:
        return "pipeline"
    elif any(n.get("fan_out", 0) > 2 for n in topo.get("nodes", []) if isinstance(n, dict)):
        return "hub_and_spoke"
    elif has_branching:
        return "branching"
    return "pipeline"


def compute_topology_conditional(
    finding_id: str, q1: list, q2: list, candidates: list,
) -> list:
    """Break down effectiveness by topology class for each candidate."""
    enriched = []
    for candidate in candidates:
        topology_breakdown: dict[str, dict] = {}

        for topo_class in ["pipeline", "hub_and_spoke", "branching", "minimal"]:
            q1_topo = [r for r in q1 if get_topology_class(r) == topo_class]
            q2_topo = [r for r in q2 if get_topology_class(r) == topo_class]

            if len(q1_topo) < 3 or len(q2_topo) < 3:
                continue

            pattern = candidate["pattern"]
            q1_rate = sum(1 for r in q1_topo if has_pattern(r, pattern)) / len(q1_topo)
            q2_rate = sum(1 for r in q2_topo if has_pattern(r, pattern)) / len(q2_topo)
            reduction = q2_rate - q1_rate

            try:
                table = [
                    [sum(1 for r in q1_topo if has_pattern(r, pattern)),
                     len(q1_topo) - sum(1 for r in q1_topo if has_pattern(r, pattern))],
                    [sum(1 for r in q2_topo if has_pattern(r, pattern)),
                     len(q2_topo) - sum(1 for r in q2_topo if has_pattern(r, pattern))],
                ]
                _, p = chi2_contingency(table)
            except (ValueError, ZeroDivisionError):
                p = 1.0

            topology_breakdown[topo_class] = {
                "reduction": round(reduction, 3),
                "n": len(q1_topo) + len(q2_topo),
                "p": round(p, 4),
                "q1_count": len(q1_topo),
                "q2_count": len(q2_topo),
            }

        enriched_candidate = {**candidate, "topology_conditional": topology_breakdown}
        enriched_candidate["implementation_patterns"] = get_implementation_patterns(
            candidate["pattern"]
        )
        enriched.append(enriched_candidate)

    return enriched


def get_implementation_patterns(pattern_name: str) -> list:
    """Return framework-specific implementation guidance."""
    patterns = {
        "output_validation": [
            {"framework": "crewai", "description": "Add output_pydantic parameter to Task definition", "complexity": "low"},
            {"framework": "langgraph", "description": "Add TypedDict state schema with validation on state updates", "complexity": "medium"},
            {"framework": "autogen", "description": "Add output validation in ConversableAgent.receive() override", "complexity": "medium"},
        ],
        "output_validation_at_boundary": [
            {"framework": "crewai", "description": "Add output_pydantic to Task at delegation boundary", "complexity": "low"},
            {"framework": "langgraph", "description": "Add validation node between agent nodes in graph", "complexity": "medium"},
        ],
        "timeout_iteration_guards": [
            {"framework": "crewai", "description": "Set max_rpm and timeout on Agent; set max_iterations on Crew", "complexity": "low"},
            {"framework": "langgraph", "description": "Add recursion_limit to graph.invoke() call", "complexity": "low"},
            {"framework": "autogen", "description": "Set max_consecutive_auto_reply on ConversableAgent", "complexity": "low"},
        ],
        "timeout_at_boundary": [
            {"framework": "crewai", "description": "Set timeout parameter on each Agent near delegation", "complexity": "low"},
            {"framework": "langgraph", "description": "Wrap node functions with timeout decorator", "complexity": "medium"},
        ],
        "exception_handling_topology": [
            {"framework": "all", "description": "Add try/except with specific exception types at delegation call sites", "complexity": "low"},
        ],
        "error_handling_at_boundary": [
            {"framework": "crewai", "description": "Wrap crew.kickoff() in try/except with fallback logic", "complexity": "low"},
            {"framework": "langgraph", "description": "Add error handling edges in graph definition", "complexity": "medium"},
        ],
        "prompt_constraints": [
            {"framework": "all", "description": "Add role boundary and output format constraints to system prompts", "complexity": "low"},
        ],
        "prompt_role_boundary": [
            {"framework": "all", "description": "Add 'You are ONLY responsible for X' to system prompt", "complexity": "low"},
        ],
        "uncertainty_instruction": [
            {"framework": "all", "description": "Add 'If unsure, explicitly state your confidence level' to system prompt", "complexity": "low"},
        ],
        "input_sanitization": [
            {"framework": "all", "description": "Validate and type-check inputs from other agents before processing", "complexity": "medium"},
        ],
        "pydantic_validation": [
            {"framework": "crewai", "description": "Define Pydantic BaseModel for expected output, use output_pydantic", "complexity": "low"},
            {"framework": "langgraph", "description": "Use Pydantic models for state schema validation", "complexity": "medium"},
        ],
    }
    return patterns.get(pattern_name, [{"framework": "all", "description": f"Implement {pattern_name}", "complexity": "medium"}])


# ---------------------------------------------------------------------------
# Step 4: Cross-Pattern Interactions & Priority Scoring
# ---------------------------------------------------------------------------

def compute_cross_pattern_interactions(finding_id: str, records: list) -> list:
    """Find findings that frequently co-occur with this one."""
    finding_repos: set[str] = set()
    for rec in records:
        findings = rec.get("failure_modes", rec.get("findings", []))
        if any(isinstance(f, dict) and f.get("finding_id") == finding_id for f in findings):
            finding_repos.add(rec.get("repo_full_name", ""))

    co_occurrences: dict[str, dict] = {}
    for rec in records:
        repo = rec.get("repo_full_name", "")
        if repo not in finding_repos:
            continue
        findings = rec.get("failure_modes", rec.get("findings", []))
        for f in findings:
            if not isinstance(f, dict):
                continue
            fid = f.get("finding_id", "")
            if fid and fid != finding_id:
                if fid not in co_occurrences:
                    co_occurrences[fid] = {"co_count": 0, "total": 0}
                co_occurrences[fid]["co_count"] += 1

    for rec in records:
        findings = rec.get("failure_modes", rec.get("findings", []))
        for f in findings:
            if not isinstance(f, dict):
                continue
            fid = f.get("finding_id", "")
            if fid in co_occurrences:
                co_occurrences[fid]["total"] += 1

    interactions = []
    for fid, counts in co_occurrences.items():
        co_rate = counts["co_count"] / max(1, len(finding_repos))
        if co_rate > 0.3:
            interactions.append({
                "interacting_finding": fid,
                "co_occurrence_rate": round(co_rate, 3),
                "co_occurrence_count": counts["co_count"],
            })

    interactions.sort(key=lambda x: x["co_occurrence_rate"], reverse=True)
    return interactions[:5]


def compute_priority_score(
    candidate: dict, finding_severity: str, interactions: list,
) -> float:
    """Compute remediation priority score."""
    severity_weights = {"critical": 4.0, "high": 3.0, "medium": 2.0, "low": 1.0}
    complexity_effort = {"low": 1.0, "medium": 2.0, "high": 3.0}

    expected_reduction = candidate.get("expected_manifestation_reduction", 0)
    if expected_reduction <= 0:
        expected_reduction = min(0.8, (candidate.get("odds_ratio", 1.0) - 1.0) / 10.0)

    severity = severity_weights.get(finding_severity, 2.0)

    interaction_multiplier = 1.0
    if interactions:
        interaction_multiplier += 0.2 * len(interactions)

    efforts = [
        complexity_effort.get(p.get("complexity", "medium"), 2.0)
        for p in candidate.get("implementation_patterns", [])
    ]
    effort = min(efforts) if efforts else 2.0

    return round(expected_reduction * severity * interaction_multiplier / effort, 2)


def generate_rationale(candidate: dict, interactions: list) -> str:
    """Generate human-readable rationale for remediation priority."""
    parts: list[str] = []

    or_val = candidate.get("odds_ratio")
    if or_val and or_val > 1:
        parts.append(f"Odds ratio {or_val}x: repos with this pattern are {or_val}x more likely to avoid manifestation")

    p = candidate.get("p_value")
    if p and p < 0.01:
        parts.append(f"High statistical confidence (p={p})")

    topo = candidate.get("topology_conditional", {})
    if topo:
        best_topo = max(topo.items(), key=lambda x: x[1].get("reduction", 0), default=(None, {}))
        if best_topo[0]:
            parts.append(f"Most effective for {best_topo[0]} topologies (reduction={best_topo[1].get('reduction', 0)})")

    if interactions:
        co_findings = [i["interacting_finding"] for i in interactions[:2]]
        parts.append(f"Also addresses co-occurring findings: {', '.join(co_findings)}")

    return ". ".join(parts)
