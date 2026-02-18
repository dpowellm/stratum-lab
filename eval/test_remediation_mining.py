"""Tests for cross-repo remediation mining.

Tests with synthetic corpus data to verify partitioning, pattern differential,
topology-conditional analysis, priority scoring, and rationale generation.
"""
from __future__ import annotations

import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PARENT_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PARENT_DIR)

from stratum_lab.remediation import (
    partition_repos,
    compute_pattern_differential,
    compute_topology_conditional,
    compute_cross_pattern_interactions,
    compute_priority_score,
    generate_rationale,
    has_pattern,
    check_manifestation,
    chi2_manual,
    chi2_contingency,
    ttest_ind,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_record(
    repo_name: str = "test-repo",
    has_finding: bool = False,
    finding_id: str = "STRAT-SD-001",
    manifested: bool = False,
    severity: str = "medium",
    has_output_validation: bool = False,
    has_timeout: bool = False,
    has_error_handling: bool = False,
    has_prompt_constraints: bool = False,
    topology_class: str = "pipeline",
    ov_count: int = 0,
    ov_at_boundary: int = 0,
    ov_pydantic: int = 0,
    timeout_count: int = 0,
    timeout_near_boundary: int = 0,
    timeout_effective: int = 0,
    eh_count: int = 0,
    eh_near_boundary: int = 0,
    eh_logging: int = 0,
    pc_count: int = 0,
    pc_role_boundary: int = 0,
    pc_uncertainty: int = 0,
) -> dict:
    """Create a synthetic behavioral record for testing."""
    findings = []
    if has_finding:
        findings.append({
            "finding_id": finding_id,
            "severity": severity,
            "manifestation_observed": manifested,
        })

    # Use override counts if provided, else derive from boolean flags
    _ov_count = ov_count if ov_count else (1 if has_output_validation else 0)
    _timeout_count = timeout_count if timeout_count else (1 if has_timeout else 0)
    _eh_count = eh_count if eh_count else (1 if has_error_handling else 0)
    _pc_count = pc_count if pc_count else (1 if has_prompt_constraints else 0)

    # Build semantic_analysis to control manifestation for semantic findings
    trust_elevation_rate = 0.8 if manifested else 0.1
    mean_chain_fidelity = 0.1 if manifested else 0.8
    fabrication_risk_rate = 0.5 if manifested else 0.0
    mean_stability_score = 0.1 if manifested else 0.8

    # Topology info
    node_count = 4 if topology_class == "pipeline" else (5 if topology_class == "hub_and_spoke" else 3)
    has_branching = topology_class in ("hub_and_spoke", "branching")

    return {
        "repo_full_name": repo_name,
        "findings": findings,
        "failure_modes": findings,
        "defensive_patterns": {
            "summary": {
                "output_validation": {
                    "count": _ov_count,
                    "at_boundary_count": ov_at_boundary,
                    "pydantic_count": ov_pydantic,
                    "typed_dict_count": 0,
                },
                "timeout_iteration_guards": {
                    "count": _timeout_count,
                    "near_boundary_count": timeout_near_boundary,
                    "effective_count": timeout_effective,
                },
                "exception_handling_topology": {
                    "count": _eh_count,
                    "near_boundary_count": eh_near_boundary,
                    "bare_except_count": 0,
                    "has_logging_count": eh_logging,
                },
                "concurrency_controls": {"count": 0, "near_shared_state_count": 0},
                "rate_limiting_backoff": {"count": 0, "has_exponential_count": 0},
                "input_sanitization": {"count": 0, "from_external_count": 0},
                "prompt_constraints": {
                    "count": _pc_count,
                    "role_boundary_count": pc_role_boundary,
                    "prohibition_count": 0,
                    "uncertainty_instruction_count": pc_uncertainty,
                },
            },
            "patterns": [],
        },
        "semantic_analysis": {
            "aggregate_scores": {
                "trust_elevation_rate": trust_elevation_rate,
                "mean_chain_fidelity": mean_chain_fidelity,
                "fabrication_risk_rate": fabrication_risk_rate,
                "mean_stability_score": mean_stability_score,
            },
            "nodes": [],
        },
        "topology_analysis": {
            "node_count": node_count,
            "edge_count": node_count - 1,
            "has_branching": has_branching,
            "nodes": (
                [{"fan_out": 3}] if topology_class == "hub_and_spoke"
                else [{"fan_out": 1}]
            ),
        },
    }


# =========================================================================
# Tests: Partitioning
# =========================================================================


class TestPartitioning:
    """Test Q1-Q4 partitioning."""

    def test_partitions_correctly(self):
        """Records with finding+manifestation go to Q1, finding only to Q2."""
        records = [
            make_record(repo_name="r1", has_finding=True, manifested=True),   # Q1
            make_record(repo_name="r2", has_finding=True, manifested=False),  # Q2
            make_record(repo_name="r3", has_finding=False, manifested=True),  # Q3
            make_record(repo_name="r4", has_finding=False, manifested=False), # Q4
        ]
        q = partition_repos("STRAT-SD-001", records)
        assert len(q["q1"]) == 1
        assert len(q["q2"]) == 1
        assert len(q["q3"]) == 1
        assert len(q["q4"]) == 1

    def test_empty_records(self):
        """Empty record list returns empty quadrants."""
        q = partition_repos("STRAT-SD-001", [])
        assert len(q["q1"]) == 0
        assert len(q["q2"]) == 0

    def test_all_q1(self):
        """All records with finding + manifestation go to Q1."""
        records = [
            make_record(repo_name=f"r{i}", has_finding=True, manifested=True)
            for i in range(10)
        ]
        q = partition_repos("STRAT-SD-001", records)
        assert len(q["q1"]) == 10
        assert len(q["q2"]) == 0

    def test_has_pattern_with_sub_features(self):
        """has_pattern correctly detects sub-features."""
        record = make_record(
            has_output_validation=True,
            ov_count=2,
            ov_at_boundary=1,
            ov_pydantic=1,
        )
        assert has_pattern(record, "output_validation") is True
        assert has_pattern(record, "output_validation_at_boundary") is True
        assert has_pattern(record, "pydantic_validation") is True

    def test_has_pattern_missing(self):
        """has_pattern returns False for absent patterns."""
        record = make_record()
        assert has_pattern(record, "output_validation") is False
        assert has_pattern(record, "timeout_at_boundary") is False


# =========================================================================
# Tests: Pattern Differential
# =========================================================================


class TestPatternDifferential:
    """Test defensive pattern differential computation."""

    def test_detects_significant_differential(self):
        """When Q2 has much higher pattern prevalence, detect as significant."""
        # Q1: manifested (mostly WITHOUT output_validation)
        q1 = [
            make_record(repo_name=f"q1_{i}", has_output_validation=False, ov_count=0)
            for i in range(50)
        ]
        q1 += [
            make_record(repo_name=f"q1_ov_{i}", has_output_validation=True, ov_count=1)
            for i in range(6)
        ]  # 10.7%

        # Q2: survived (mostly WITH output_validation)
        q2 = [
            make_record(repo_name=f"q2_ov_{i}", has_output_validation=True, ov_count=2)
            for i in range(37)
        ]  # 67.3%
        q2 += [
            make_record(repo_name=f"q2_{i}", has_output_validation=False, ov_count=0)
            for i in range(18)
        ]

        candidates = compute_pattern_differential(q1, q2, min_n=5, p_threshold=0.05)
        assert len(candidates) > 0, "Should find at least one significant pattern"
        # Output validation should be among candidates
        ov_candidates = [c for c in candidates if c["pattern"] == "output_validation"]
        assert len(ov_candidates) > 0, "output_validation should be a top candidate"

    def test_ignores_small_samples(self):
        """With fewer records than min_n, returns empty."""
        q1 = [make_record(repo_name=f"q1_{i}") for i in range(3)]
        q2 = [make_record(repo_name=f"q2_{i}") for i in range(3)]
        candidates = compute_pattern_differential(q1, q2, min_n=5, p_threshold=0.05)
        assert len(candidates) == 0

    def test_no_differential_when_equal(self):
        """When Q1 and Q2 have same prevalence, no candidates emerge."""
        q1 = [make_record(repo_name=f"q1_{i}", has_timeout=True, timeout_count=1) for i in range(30)]
        q2 = [make_record(repo_name=f"q2_{i}", has_timeout=True, timeout_count=1) for i in range(30)]
        candidates = compute_pattern_differential(q1, q2, min_n=5, p_threshold=0.05)
        # With equal prevalence in both groups, timeout should not appear
        timeout_cands = [c for c in candidates if c["pattern"] == "timeout_iteration_guards"]
        assert len(timeout_cands) == 0


# =========================================================================
# Tests: Topology Conditional
# =========================================================================


class TestTopologyConditional:
    """Test topology-conditional remediation breakdown."""

    def test_enriches_candidates_with_topology(self):
        """Candidates are enriched with topology_conditional and implementation_patterns."""
        q1 = [make_record(repo_name=f"q1_{i}", topology_class="pipeline", has_output_validation=False) for i in range(10)]
        q2 = [make_record(repo_name=f"q2_{i}", topology_class="pipeline", has_output_validation=True, ov_count=2) for i in range(10)]

        candidates = [{
            "pattern": "output_validation",
            "description": "Output validation",
            "q1_prevalence": 0.0,
            "q2_prevalence": 1.0,
            "odds_ratio": 10.0,
            "p_value": 0.001,
            "confidence": "high",
            "expected_manifestation_reduction": 1.0,
            "sample_sizes": {"q1": 10, "q2": 10},
        }]

        enriched = compute_topology_conditional("STRAT-SD-001", q1, q2, candidates)
        assert len(enriched) == 1
        assert "topology_conditional" in enriched[0]
        assert "implementation_patterns" in enriched[0]
        assert len(enriched[0]["implementation_patterns"]) > 0


# =========================================================================
# Tests: Priority Scoring
# =========================================================================


class TestPriorityScoring:
    """Test remediation priority scoring."""

    def test_higher_severity_higher_priority(self):
        """Critical findings should get higher priority scores."""
        candidate = {
            "expected_manifestation_reduction": 0.5,
            "odds_ratio": 3.0,
            "implementation_patterns": [{"complexity": "low"}],
        }
        score_critical = compute_priority_score(candidate, "critical", [])
        score_low = compute_priority_score(candidate, "low", [])
        assert score_critical > score_low

    def test_interaction_multiplier(self):
        """Findings with co-occurring patterns get priority boost."""
        candidate = {
            "expected_manifestation_reduction": 0.5,
            "odds_ratio": 3.0,
            "implementation_patterns": [{"complexity": "low"}],
        }
        interactions = [{"interacting_finding": "STRAT-DC-001", "co_occurrence_rate": 0.5}]
        score_with = compute_priority_score(candidate, "high", interactions)
        score_without = compute_priority_score(candidate, "high", [])
        assert score_with > score_without

    def test_higher_effort_lower_priority(self):
        """Higher complexity effort reduces priority score."""
        candidate_easy = {
            "expected_manifestation_reduction": 0.5,
            "odds_ratio": 3.0,
            "implementation_patterns": [{"complexity": "low"}],
        }
        candidate_hard = {
            "expected_manifestation_reduction": 0.5,
            "odds_ratio": 3.0,
            "implementation_patterns": [{"complexity": "high"}],
        }
        score_easy = compute_priority_score(candidate_easy, "high", [])
        score_hard = compute_priority_score(candidate_hard, "high", [])
        assert score_easy > score_hard

    def test_zero_reduction_uses_fallback(self):
        """When expected_reduction is 0, falls back to odds_ratio-based estimate."""
        candidate = {
            "expected_manifestation_reduction": 0,
            "odds_ratio": 5.0,
            "implementation_patterns": [{"complexity": "low"}],
        }
        score = compute_priority_score(candidate, "medium", [])
        assert score > 0


# =========================================================================
# Tests: Cross-Pattern Interactions
# =========================================================================


class TestCrossPatternInteractions:
    """Test cross-pattern interaction detection."""

    def test_detects_co_occurrences(self):
        """Findings with high co-occurrence rate are detected."""
        records = []
        for i in range(20):
            rec = make_record(
                repo_name=f"repo_{i}",
                has_finding=True,
                finding_id="STRAT-SD-001",
            )
            # Add co-occurring finding
            rec["findings"].append({"finding_id": "STRAT-CE-001", "severity": "high"})
            rec["failure_modes"] = rec["findings"]
            records.append(rec)

        interactions = compute_cross_pattern_interactions("STRAT-SD-001", records)
        assert len(interactions) >= 1
        assert any(i["interacting_finding"] == "STRAT-CE-001" for i in interactions)

    def test_no_interactions_when_no_co_occurrences(self):
        """No interactions when findings don't co-occur above threshold."""
        records = []
        for i in range(20):
            records.append(make_record(
                repo_name=f"repo_{i}",
                has_finding=True,
                finding_id="STRAT-SD-001",
            ))
        interactions = compute_cross_pattern_interactions("STRAT-SD-001", records)
        assert len(interactions) == 0


# =========================================================================
# Tests: Rationale Generation
# =========================================================================


class TestRationaleGeneration:
    """Test rationale text generation."""

    def test_generates_nonempty_rationale(self):
        """Rationale is generated with relevant details."""
        candidate = {
            "pattern": "output_validation",
            "odds_ratio": 3.5,
            "p_value": 0.005,
            "topology_conditional": {
                "pipeline": {"reduction": 0.4, "n": 20, "p": 0.01},
            },
        }
        interactions = [{"interacting_finding": "STRAT-CE-001", "co_occurrence_rate": 0.6}]
        rationale = generate_rationale(candidate, interactions)
        assert len(rationale) > 0
        assert "3.5" in rationale  # Odds ratio
        assert "pipeline" in rationale  # Topology

    def test_rationale_empty_candidate(self):
        """Rationale handles candidate with minimal data."""
        candidate = {"pattern": "output_validation"}
        rationale = generate_rationale(candidate, [])
        # Should not crash, may be empty string
        assert isinstance(rationale, str)


# =========================================================================
# Tests: Statistical Functions
# =========================================================================


class TestStatisticalFunctions:
    """Test chi-squared and t-test implementations."""

    def test_chi2_manual_basic(self):
        """Manual chi-squared with clear separation returns low p-value."""
        table = [[10, 40], [40, 10]]
        chi2, p = chi2_manual(table)
        assert chi2 > 0
        assert p < 0.05

    def test_chi2_manual_equal(self):
        """Equal distribution returns chi2=0."""
        table = [[25, 25], [25, 25]]
        chi2, p = chi2_manual(table)
        assert chi2 == 0.0
        assert p == 1.0

    def test_chi2_manual_empty(self):
        """All zeros returns 0, 1."""
        chi2, p = chi2_manual([[0, 0], [0, 0]])
        assert chi2 == 0.0
        assert p == 1.0

    def test_chi2_contingency_wrapper(self):
        """chi2_contingency wrapper returns same direction as manual."""
        table = [[5, 45], [45, 5]]
        chi2, p = chi2_contingency(table)
        assert chi2 > 0
        assert p < 0.05

    def test_ttest_ind_different_means(self):
        """T-test detects significantly different means."""
        vals1 = [1.0, 2.0, 1.5, 2.5, 1.0] * 10
        vals2 = [8.0, 9.0, 8.5, 9.5, 8.0] * 10
        t, p = ttest_ind(vals1, vals2)
        assert abs(t) > 1.0
        assert p < 0.05

    def test_ttest_ind_small_samples(self):
        """T-test with < 2 samples returns no significance."""
        import math
        t, p = ttest_ind([1.0], [2.0])
        # With scipy, may return NaN; without scipy, manual fallback returns 1.0
        assert p == 1.0 or math.isnan(p)


# =========================================================================
# Run
# =========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
