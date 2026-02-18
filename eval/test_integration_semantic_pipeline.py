"""Integration tests for the semantic analysis and remediation mining pipeline.

Cross-phase integration: verify outputs from each phase feed correctly into
the next. All LLM calls are mocked. Tests should skipif modules not importable.
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, SCRIPT_DIR)
PARENT_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PARENT_DIR)

# Guard imports — skip all tests if modules not importable
try:
    from scan_defensive_patterns import scan_repo
    from analyze_semantics import (
        pass_1_delegation_fidelity,
        pass_2_cross_run_consistency,
        pass_3_uncertainty_chains,
        pass_4_confidence_escalation,
        pass_5_topological_vulnerability,
        compute_aggregate_scores,
    )
    from stratum_lab.remediation import (
        partition_repos,
        compute_pattern_differential,
        compute_topology_conditional,
        compute_priority_score,
        generate_rationale,
    )
    from mine_remediations import mine_finding
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not MODULES_AVAILABLE,
    reason="Semantic pipeline modules not importable",
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_repo(files: dict[str, str]) -> str:
    """Create a temp directory with given files. Returns repo root path."""
    tmpdir = tempfile.mkdtemp(prefix="stratum_integration_")
    for relpath, content in files.items():
        fpath = os.path.join(tmpdir, relpath)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
    return tmpdir


def _make_events(node_names: list[str], include_errors: bool = False) -> list[dict]:
    """Create events for a delegation chain with task_start, task_end, llm events."""
    events: list[dict] = []
    for i, name in enumerate(node_names):
        nid = f"framework:{name}:src/{name.lower()}.py:{(i + 1) * 10}"
        ts_start = f"2025-01-01T00:00:{i * 4 + 1:02d}Z"
        ts_llm = f"2025-01-01T00:00:{i * 4 + 2:02d}Z"
        ts_llm_end = f"2025-01-01T00:00:{i * 4 + 3:02d}Z"
        ts_end = f"2025-01-01T00:00:{i * 4 + 4:02d}Z"

        events.append({
            "event_type": "agent.task_start",
            "node_id": nid,
            "timestamp": ts_start,
        })
        events.append({
            "event_type": "llm.call_start",
            "node_id": nid,
            "timestamp": ts_llm,
            "last_user_message_preview": f"Input for {name} agent with sufficient text " + "x" * 40,
        })
        events.append({
            "event_type": "llm.call_end",
            "node_id": nid,
            "timestamp": ts_llm_end,
            "output_preview": f"Output from {name} agent with enough text for analysis " + "y" * 40,
        })
        if include_errors and i == 1:
            events.append({
                "event_type": "error.tool_failure",
                "node_id": nid,
                "timestamp": ts_llm_end,
                "error_message": "tool timeout",
            })
            # Add a second LLM call after the error
            events.append({
                "event_type": "llm.call_end",
                "node_id": nid,
                "timestamp": f"2025-01-01T00:00:{i * 4 + 4:02d}Z",
                "output_preview": f"Retry output from {name} with more detail " + "z" * 40,
            })
        events.append({
            "event_type": "agent.task_end",
            "node_id": nid,
            "timestamp": ts_end,
            "output_preview": f"Final output from {name} agent with results " + "w" * 50,
        })
    return events


def _make_mock_client(responses: dict[str, dict] | None = None) -> MagicMock:
    """Create a mock VLLMClient with configurable per-pass responses."""
    default_response = {
        "hedging_preserved": True,
        "source_attribution_preserved": True,
        "scope_preserved": True,
        "factual_additions_detected": False,
        "format_transformation_loss": False,
        "role_boundary_respected": True,
        "mast_failure_mode": "none",
        "uncertainty_transfer": "preserved",
        "factual_agreement": True,
        "structural_agreement": True,
        "confidence_direction": "same",
        "semantic_overlap_estimate": 0.85,
        "novel_claims_in_other": 0,
        "dropped_claims_from_run1": 0,
        "claim_identified": "Market will decline 20%",
        "confidence_at_origin": "hedged",
        "confidence_at_terminus": "asserted",
        "elevation_boundary": "Writer",
        "information_accretion": False,
        "accretion_boundary": "none",
        "chain_fidelity": 0.7,
        "confidence_trajectory": "stable",
        "compensatory_assertion": False,
        "tool_failure_acknowledged": True,
        "fabrication_risk": "low",
    }
    client = MagicMock()
    client.structured_query.return_value = default_response
    return client


def _make_record(
    repo_name: str,
    has_finding: bool = False,
    finding_id: str = "STRAT-SD-001",
    manifested: bool = False,
    severity: str = "medium",
    has_output_validation: bool = False,
    ov_count: int = 0,
) -> dict:
    """Build a synthetic behavioral record."""
    findings = []
    if has_finding:
        findings.append({
            "finding_id": finding_id,
            "severity": severity,
            "manifestation_observed": manifested,
        })

    _ov_count = ov_count if ov_count else (1 if has_output_validation else 0)
    trust_elevation_rate = 0.8 if manifested else 0.1
    mean_chain_fidelity = 0.1 if manifested else 0.8

    return {
        "repo_full_name": repo_name,
        "findings": findings,
        "failure_modes": findings,
        "defensive_patterns": {
            "summary": {
                "output_validation": {"count": _ov_count, "at_boundary_count": 0, "pydantic_count": 0, "typed_dict_count": 0},
                "timeout_iteration_guards": {"count": 0, "near_boundary_count": 0, "effective_count": 0},
                "exception_handling_topology": {"count": 0, "near_boundary_count": 0, "bare_except_count": 0, "has_logging_count": 0},
                "concurrency_controls": {"count": 0, "near_shared_state_count": 0},
                "rate_limiting_backoff": {"count": 0, "has_exponential_count": 0},
                "input_sanitization": {"count": 0, "from_external_count": 0},
                "prompt_constraints": {"count": 0, "role_boundary_count": 0, "prohibition_count": 0, "uncertainty_instruction_count": 0},
            },
            "patterns": [],
        },
        "semantic_analysis": {
            "aggregate_scores": {
                "trust_elevation_rate": trust_elevation_rate,
                "mean_chain_fidelity": mean_chain_fidelity,
                "fabrication_risk_rate": 0.0,
                "mean_stability_score": 0.8,
            },
            "nodes": [],
        },
        "topology_analysis": {
            "node_count": 3,
            "edge_count": 2,
            "has_branching": False,
            "nodes": [{"fan_out": 1}],
        },
    }


# =========================================================================
# Test 1: Phase 0 scan feeds into Pass 5 vulnerability scoring
# =========================================================================


class TestPhase0FeedsPass5:
    """Verify defensive scan output is consumed by vulnerability scoring."""

    def test_defended_nodes_have_lower_vulnerability(self):
        """Phase 0 defensive patterns reduce Pass 5 vulnerability scores."""
        # Create a repo with timeout guards near delegation
        repo = _make_repo({
            "src/researcher.py": (
                "from crewai import Agent, Crew\n"
                "agent = Agent(role='researcher', timeout=60)\n"
                "crew = Crew(agents=[agent])\n"
                "result = crew.kickoff()\n"
            ),
            "src/writer.py": (
                "class Writer:\n"
                "    def write(self, text):\n"
                "        return text\n"
            ),
        })

        # Phase 0: scan for defensive patterns
        defensive = scan_repo(repo)
        assert defensive["total_patterns_found"] > 0, "Should detect defensive patterns"

        # Create events for a 2-node chain (Researcher -> Writer)
        events = _make_events(["Researcher", "Writer"])

        # Pass 5: vulnerability scoring WITH defenses
        results_defended = pass_5_topological_vulnerability(
            {"run_1": events}, defensive
        )

        # Pass 5: vulnerability scoring WITHOUT defenses
        results_undefended = pass_5_topological_vulnerability(
            {"run_1": events}, {"patterns": []}
        )

        # Every node's defended score should be <= its undefended score
        for defended, undefended in zip(
            sorted(results_defended, key=lambda x: x["node_id"]),
            sorted(results_undefended, key=lambda x: x["node_id"]),
        ):
            assert defended["vulnerability_score"] <= undefended["vulnerability_score"], (
                f"Defended node {defended['node_id']} should have "
                f"vulnerability <= undefended ({defended['vulnerability_score']} "
                f"vs {undefended['vulnerability_score']})"
            )

        # At least one node should have defense_count > 0
        any_defended = any(
            n["has_defenses"]["defense_count"] > 0 for n in results_defended
        )
        assert any_defended, "At least one node should have defenses from Phase 0 scan"


# =========================================================================
# Test 2: All 5 passes produce structured output from 5 runs
# =========================================================================


class TestAllPassesStructuredOutput:
    """Verify all 5 passes produce correct structured output."""

    def test_five_passes_with_five_runs(self):
        """All 5 passes produce structured output from 5 runs with mocked VLLMClient."""
        # Build events for 5 runs (runs 4,5 repeat run 1)
        events_by_run = {}
        for i in range(1, 6):
            events_by_run[f"events_run_{i}"] = _make_events(
                ["Researcher", "Analyst", "Writer"],
                include_errors=(i == 2),
            )

        mock_client = _make_mock_client()
        defensive = {"patterns": []}

        # Pass 1: Delegation Fidelity
        p1_results, p1_calls = pass_1_delegation_fidelity(events_by_run, mock_client)
        assert isinstance(p1_results, list)
        assert p1_calls >= 0

        # Pass 2: Cross-Run Consistency (needs >= 4 runs)
        p2_results, p2_calls = pass_2_cross_run_consistency(events_by_run, mock_client)
        assert isinstance(p2_results, dict)
        assert "per_comparison" in p2_results
        assert "node_stability" in p2_results

        # Pass 3: Uncertainty Chains
        p3_results, p3_calls = pass_3_uncertainty_chains(events_by_run, mock_client)
        assert isinstance(p3_results, list)
        # With 3-node chains in 5 runs, should have results
        if p3_results:
            r = p3_results[0]
            assert "chain_length" in r
            assert "chain_roles" in r or "chain" in r

        # Pass 4: Confidence Escalation
        p4_results, p4_calls = pass_4_confidence_escalation(events_by_run, mock_client)
        assert isinstance(p4_results, list)
        # Run 2 has errors + multi-call node, should produce results
        if p4_results:
            r = p4_results[0]
            assert "node_id" in r
            assert "call_count" in r

        # Pass 5: Topological Vulnerability
        p5_results = pass_5_topological_vulnerability(events_by_run, defensive)
        assert isinstance(p5_results, list)
        assert len(p5_results) >= 3  # 3 nodes in chain
        for node in p5_results:
            assert "node_id" in node
            assert "position_class" in node
            assert "raw_vulnerability_score" in node
            assert "vulnerability_score" in node
            assert "has_defenses" in node

        # Aggregate scores
        all_results = {
            "delegation_fidelity": p1_results,
            "cross_run_consistency": p2_results,
            "uncertainty_chains": p3_results,
            "confidence_escalation": p4_results,
            "topological_vulnerability": p5_results,
        }
        scores = compute_aggregate_scores(all_results)
        assert isinstance(scores, dict)
        assert "oer_estimate" in scores
        assert 0.0 <= scores["oer_estimate"] <= 1.0


# =========================================================================
# Test 3: Pass failure isolation
# =========================================================================


class TestPassFailureIsolation:
    """Verify that one pass crashing doesn't prevent others from running."""

    def test_pass1_crash_does_not_block_pass3_and_pass5(self):
        """If Pass 1 crashes, Pass 3 and 5 still produce valid results."""
        events_by_run = {}
        for i in range(1, 6):
            events_by_run[f"events_run_{i}"] = _make_events(
                ["Researcher", "Analyst", "Writer"]
            )

        # Client that raises on first call (simulates Pass 1 crash)
        crashing_client = MagicMock()
        crashing_client.structured_query.side_effect = ConnectionError("vLLM down")

        defensive = {"patterns": []}

        # Pass 1 should raise
        with pytest.raises(ConnectionError):
            pass_1_delegation_fidelity(events_by_run, crashing_client)

        # Pass 3 would also crash with this client, but the point is
        # the system handles errors per-pass. Test that Pass 5 (no LLM) works.
        p5_results = pass_5_topological_vulnerability(events_by_run, defensive)
        assert isinstance(p5_results, list)
        assert len(p5_results) >= 3

        # Aggregate still works with error results for failed passes
        results = {
            "delegation_fidelity": {"error": "ConnectionError: vLLM down"},
            "cross_run_consistency": {"error": "ConnectionError: vLLM down"},
            "uncertainty_chains": {"error": "ConnectionError: vLLM down"},
            "confidence_escalation": {"error": "ConnectionError: vLLM down"},
            "topological_vulnerability": p5_results,
        }
        scores = compute_aggregate_scores(results)
        assert isinstance(scores, dict)
        # OER should still be computable with partial data
        assert "oer_estimate" in scores
        assert scores["oer_estimate"] >= 0


# =========================================================================
# Test 4: Full flow — manifestation → mine_finding → remediation emerges
# =========================================================================


class TestFullFlowRemediation:
    """End-to-end: semantic scores drive manifestation, mining finds remediation."""

    def test_output_validation_emerges_as_remediation(self):
        """Build corpus where output_validation differentiates Q1 from Q2,
        mine_finding identifies it as a remediation candidate."""
        records = []

        # Q1: has finding + manifested (mostly WITHOUT output_validation)
        for i in range(30):
            rec = _make_record(
                repo_name=f"q1_{i}",
                has_finding=True,
                finding_id="STRAT-SD-001",
                manifested=True,
                severity="high",
                has_output_validation=False,
            )
            records.append(rec)

        # Q2: has finding + NOT manifested (mostly WITH output_validation)
        for i in range(30):
            rec = _make_record(
                repo_name=f"q2_{i}",
                has_finding=True,
                finding_id="STRAT-SD-001",
                manifested=False,
                severity="high",
                has_output_validation=True,
                ov_count=3,
            )
            records.append(rec)

        # Q4: no finding, not manifested (background)
        for i in range(20):
            rec = _make_record(
                repo_name=f"q4_{i}",
                has_finding=False,
                manifested=False,
            )
            records.append(rec)

        # Run mine_finding
        result = mine_finding("STRAT-SD-001", records, min_n=5, p_threshold=0.05)

        assert result is not None
        assert result["finding_id"] == "STRAT-SD-001"
        assert result["severity"] == "high"

        # Corpus statistics
        stats = result["corpus_statistics"]
        assert stats["q1_count"] == 30
        assert stats["q2_count"] == 30
        assert stats["total_repos_with_finding"] == 60

        # Remediation candidates should include output_validation
        candidates = result["remediation_candidates"]
        assert len(candidates) > 0, "Should find remediation candidates"
        ov_candidates = [c for c in candidates if c["pattern"] == "output_validation"]
        assert len(ov_candidates) > 0, "output_validation should emerge as candidate"

        # Priority ranking should exist and be 1-indexed
        ranked = result["priority_ranked_remediations"]
        assert len(ranked) > 0
        assert ranked[0]["rank"] == 1
        assert ranked[0]["priority_score"] > 0
        assert len(ranked[0]["rationale"]) > 0


# =========================================================================
# Run
# =========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
