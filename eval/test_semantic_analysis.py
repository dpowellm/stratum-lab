"""Tests for semantic analysis pipeline.

Tests with synthetic delegation data. Does NOT require vLLM — mocks LLM
responses via unittest.mock.
"""
from __future__ import annotations

import json
import os
import sys

import pytest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, SCRIPT_DIR)
PARENT_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PARENT_DIR)

from analyze_semantics import (
    pass_1_delegation_fidelity,
    pass_2_cross_run_consistency,
    pass_3_uncertainty_chains,
    pass_4_confidence_escalation,
    pass_5_topological_vulnerability,
    compute_aggregate_scores,
    extract_delegation_edges,
    extract_node_outputs,
    build_delegation_chain,
)
from stratum_lab.semantic import (
    compute_delegation_fidelity_score,
    compute_stability_score,
    validate_pass1_response,
    validate_pass2_response,
    validate_pass3_response,
    validate_pass4_response,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_delegation_events(
    source_output: str,
    target_input: str,
    source_id: str = "framework:Researcher:src/researcher.py:10",
    target_id: str = "framework:Writer:src/writer.py:20",
) -> list[dict]:
    """Create synthetic events simulating a delegation from source to target."""
    return [
        {
            "event_type": "agent.task_end",
            "node_id": source_id,
            "timestamp": "2025-01-01T00:00:01Z",
            "output_preview": source_output,
        },
        {
            "event_type": "llm.call_start",
            "node_id": target_id,
            "timestamp": "2025-01-01T00:00:02Z",
            "last_user_message_preview": target_input,
        },
        {
            "event_type": "agent.task_start",
            "node_id": target_id,
            "timestamp": "2025-01-01T00:00:02Z",
        },
    ]


def make_chain_events(node_names: list[str]) -> list[dict]:
    """Create events for a linear delegation chain A -> B -> C."""
    events: list[dict] = []
    for i, name in enumerate(node_names):
        nid = f"framework:{name}:src/{name.lower()}.py:{(i + 1) * 10}"
        events.append({
            "event_type": "agent.task_start",
            "node_id": nid,
            "timestamp": f"2025-01-01T00:00:{i * 2 + 1:02d}Z",
        })
        events.append({
            "event_type": "agent.task_end",
            "node_id": nid,
            "timestamp": f"2025-01-01T00:00:{i * 2 + 2:02d}Z",
            "output_preview": f"Output from {name} agent " + "x" * 50,
        })
    return events


def make_mock_client(response: dict) -> MagicMock:
    """Create a mock VLLMClient that returns a fixed response."""
    client = MagicMock()
    client.structured_query.return_value = response
    return client


def make_consistency_events(node_id: str, output: str) -> list[dict]:
    """Create events for a single run with one agent producing output."""
    return [
        {
            "event_type": "agent.task_start",
            "node_id": node_id,
            "timestamp": "2025-01-01T00:00:01Z",
        },
        {
            "event_type": "agent.task_end",
            "node_id": node_id,
            "timestamp": "2025-01-01T00:00:02Z",
            "output_preview": output,
        },
        {
            "event_type": "llm.call_end",
            "node_id": node_id,
            "timestamp": "2025-01-01T00:00:02Z",
            "output_preview": output,
        },
    ]


# =========================================================================
# Tests: Pass 1 - Delegation Fidelity
# =========================================================================


class TestPass1DelegationFidelity:
    """Test delegation fidelity analysis."""

    def test_detects_hedging_loss(self):
        """When source has hedging that target drops, hedging_preserved should be false."""
        events = make_delegation_events(
            source_output="The market might decline approximately 20%",
            target_input="The market will decline 20%",
        )
        mock_client = make_mock_client({
            "hedging_preserved": False,
            "factual_additions_detected": False,
            "mast_failure_mode": "information_undersupply",
            "uncertainty_transfer": "attenuated",
        })
        results, calls = pass_1_delegation_fidelity({"run_1": events}, mock_client)
        assert len(results) == 1
        assert results[0]["hedging_preserved"] is False
        assert calls == 1

    def test_skips_short_outputs(self):
        """Edges with very short output/input are skipped."""
        events = make_delegation_events(source_output="OK", target_input="OK")
        mock_client = make_mock_client({})
        results, calls = pass_1_delegation_fidelity({"run_1": events}, mock_client)
        assert len(results) == 0
        assert calls == 0

    def test_handles_llm_parse_error(self):
        """If LLM returns unparseable response, result includes parse_error."""
        events = make_delegation_events(
            source_output="A" * 50,
            target_input="B" * 50,
        )
        mock_client = MagicMock()
        mock_client.structured_query.return_value = {"parse_error": "invalid json"}
        results, calls = pass_1_delegation_fidelity({"run_1": events}, mock_client)
        assert len(results) == 1
        assert "parse_error" in results[0]

    def test_multiple_edges_multiple_runs(self):
        """Multiple runs with delegation edges produce results per edge."""
        events_run1 = make_delegation_events(
            source_output="Research shows significant correlation " + "x" * 30,
            target_input="There is a significant correlation " + "y" * 30,
        )
        events_run2 = make_delegation_events(
            source_output="Analysis indicates possible trend " + "x" * 30,
            target_input="The trend is confirmed " + "y" * 30,
        )
        mock_client = make_mock_client({
            "hedging_preserved": True,
            "factual_additions_detected": False,
            "mast_failure_mode": "none",
            "uncertainty_transfer": "preserved",
        })
        results, calls = pass_1_delegation_fidelity(
            {"run_1": events_run1, "run_2": events_run2}, mock_client
        )
        assert len(results) == 2
        assert calls == 2


# =========================================================================
# Tests: Pass 2 - Cross-Run Consistency
# =========================================================================


class TestPass2Consistency:
    """Test cross-run consistency analysis."""

    def test_computes_stability_score(self):
        """Stability score is mean of factual, structural, semantic agreement."""
        # Direct test of the scoring function
        score = compute_stability_score(True, True, 0.8)
        expected = (1.0 + 1.0 + 0.8) / 3.0
        assert abs(score - expected) < 0.001

        score_low = compute_stability_score(False, False, 0.2)
        expected_low = (0.0 + 0.0 + 0.2) / 3.0
        assert abs(score_low - expected_low) < 0.001
        assert score > score_low

    def test_handles_missing_repeat_runs(self):
        """If fewer than 4 runs, returns empty results."""
        node_id = "framework:Agent:src/agent.py:10"
        events_by_run = {
            "events_run_1": make_consistency_events(node_id, "Output run 1 " + "x" * 40),
            "events_run_2": make_consistency_events(node_id, "Output run 2 " + "x" * 40),
        }
        mock_client = make_mock_client({
            "factual_agreement": True,
            "structural_agreement": True,
            "semantic_overlap_estimate": 0.9,
            "novel_claims_in_other": 0,
            "dropped_claims_from_run1": 0,
        })
        result, calls = pass_2_cross_run_consistency(events_by_run, mock_client)
        assert isinstance(result, dict)
        assert result["per_comparison"] == []
        assert calls == 0

    def test_with_sufficient_runs(self):
        """With 5 runs, compares run 1 vs runs 4+."""
        node_id = "framework:Agent:src/agent.py:10"
        events_by_run = {}
        for i in range(1, 6):
            events_by_run[f"events_run_{i}"] = make_consistency_events(
                node_id, f"Output from run {i} with enough text " + "x" * 30
            )

        mock_client = make_mock_client({
            "factual_agreement": True,
            "structural_agreement": False,
            "confidence_direction": "same",
            "semantic_overlap_estimate": 0.7,
            "novel_claims_in_other": 2,
            "dropped_claims_from_run1": 1,
        })
        result, calls = pass_2_cross_run_consistency(events_by_run, mock_client)
        assert calls >= 1
        assert len(result["node_stability"]) >= 1
        stability = result["node_stability"][0]
        assert 0.0 <= stability["stability_score"] <= 1.0


# =========================================================================
# Tests: Pass 5 - Topological Vulnerability
# =========================================================================


class TestPass5Vulnerability:
    """Test topological vulnerability scoring."""

    def test_terminal_node_higher_vulnerability(self):
        """Terminal nodes get 1.5x multiplier per Sherlock."""
        events = make_chain_events(["A", "B", "C"])
        defensive = {"patterns": []}
        results = pass_5_topological_vulnerability({"run_1": events}, defensive)

        terminal = [r for r in results if r["position_class"] == "terminal"]
        initial = [r for r in results if r["position_class"] == "initial"]
        assert len(terminal) >= 1, "Should have a terminal node"
        assert len(initial) >= 1, "Should have an initial node"
        # Terminal raw score should be higher than intermediate base
        # (terminal gets 1.5x, initial gets 1.3x but also has downstream reach)
        assert terminal[0]["raw_vulnerability_score"] > 0.3  # Above base

    def test_defenses_reduce_vulnerability(self):
        """Nodes with defensive patterns get reduced vulnerability score."""
        events = make_chain_events(["A", "B"])
        defensive = {
            "patterns": [{
                "file_path": "src/a_agent.py",
                "pattern_category": "timeout_iteration_guards",
                "near_delegation_boundary": True,
                "pattern_detail": {
                    "parameter": "timeout",
                    "value": 60,
                    "value_assessment": "effective",
                },
            }]
        }
        results = pass_5_topological_vulnerability({"run_1": events}, defensive)
        # The "A" node should benefit from timeout defense
        a_node = [r for r in results if "A" in r["node_id"]]
        assert len(a_node) >= 1
        a = a_node[0]
        assert a["vulnerability_score"] <= a["raw_vulnerability_score"]

    def test_isolated_nodes_low_vulnerability(self):
        """Nodes with no edges get low vulnerability."""
        events = [{
            "event_type": "agent.task_end",
            "node_id": "framework:Solo:src/solo.py:10",
            "timestamp": "2025-01-01T00:00:01Z",
            "output_preview": "Solo agent output",
        }]
        defensive = {"patterns": []}
        results = pass_5_topological_vulnerability({"run_1": events}, defensive)
        assert len(results) == 1
        assert results[0]["position_class"] == "isolated"
        assert results[0]["raw_vulnerability_score"] == 0.3  # Base score only


# =========================================================================
# Tests: Aggregate Scores
# =========================================================================


class TestAggregateScores:
    """Test aggregate score computation."""

    def test_oer_estimate_increases_with_trust_elevation(self):
        """Higher trust elevation rate should increase OER estimate."""
        # Low trust elevation
        results_low = {
            "delegation_fidelity": [
                {"hedging_preserved": True, "factual_additions_detected": False,
                 "mast_failure_mode": "none"},
            ] * 10,
            "cross_run_consistency": {"node_stability": []},
            "uncertainty_chains": [],
            "confidence_escalation": [],
            "topological_vulnerability": [],
        }
        scores_low = compute_aggregate_scores(results_low)

        # High trust elevation
        results_high = {
            "delegation_fidelity": [
                {"hedging_preserved": False, "factual_additions_detected": True,
                 "mast_failure_mode": "information_oversupply"},
            ] * 10,
            "cross_run_consistency": {"node_stability": []},
            "uncertainty_chains": [],
            "confidence_escalation": [],
            "topological_vulnerability": [],
        }
        scores_high = compute_aggregate_scores(results_high)

        assert scores_high["oer_estimate"] > scores_low["oer_estimate"]

    def test_handles_empty_passes(self):
        """If all passes return errors, aggregate scores gracefully degrade."""
        results = {
            "delegation_fidelity": {"error": "failed"},
            "cross_run_consistency": {"error": "failed"},
            "uncertainty_chains": {"error": "failed"},
            "confidence_escalation": {"error": "failed"},
            "topological_vulnerability": {"error": "failed"},
        }
        scores = compute_aggregate_scores(results)
        # OER should still be computed (with defaults), should be >= 0
        assert "oer_estimate" in scores
        assert scores["oer_estimate"] >= 0

    def test_mast_failure_distribution(self):
        """MAST failure distribution is computed from delegation fidelity results."""
        results = {
            "delegation_fidelity": [
                {"hedging_preserved": True, "factual_additions_detected": False,
                 "mast_failure_mode": "none"},
                {"hedging_preserved": False, "factual_additions_detected": True,
                 "mast_failure_mode": "information_oversupply"},
                {"hedging_preserved": False, "factual_additions_detected": False,
                 "mast_failure_mode": "information_undersupply"},
            ],
            "cross_run_consistency": {"node_stability": []},
            "uncertainty_chains": [],
            "confidence_escalation": [],
            "topological_vulnerability": [],
        }
        scores = compute_aggregate_scores(results)
        assert "mast_failure_distribution" in scores
        dist = scores["mast_failure_distribution"]
        assert abs(sum(dist.values()) - 1.0) < 0.01


# =========================================================================
# Tests: Scoring Functions
# =========================================================================


class TestScoringFunctions:
    """Test the shared scoring functions in stratum_lab.semantic."""

    def test_delegation_fidelity_perfect(self):
        """Perfect fidelity: all positive signals true, no negatives."""
        response = {
            "hedging_preserved": True,
            "source_attribution_preserved": True,
            "scope_preserved": True,
            "role_boundary_respected": True,
            "factual_additions_detected": False,
            "format_transformation_loss": False,
        }
        score = compute_delegation_fidelity_score(response)
        assert score == 1.0

    def test_delegation_fidelity_poor(self):
        """Poor fidelity: no positive signals, both negatives."""
        response = {
            "hedging_preserved": False,
            "source_attribution_preserved": False,
            "scope_preserved": False,
            "role_boundary_respected": False,
            "factual_additions_detected": True,
            "format_transformation_loss": True,
        }
        score = compute_delegation_fidelity_score(response)
        assert score == 0.0  # Should be clamped to 0

    def test_stability_score_range(self):
        """Stability score is always in [0, 1]."""
        assert 0.0 <= compute_stability_score(False, False, 0.0) <= 1.0
        assert 0.0 <= compute_stability_score(True, True, 1.0) <= 1.0
        assert 0.0 <= compute_stability_score(True, False, 0.5) <= 1.0


# =========================================================================
# Tests: Validators
# =========================================================================


class TestValidators:
    """Test response validators fill defaults for missing fields."""

    def test_validate_pass1_fills_defaults(self):
        """Pass 1 validator fills missing fields with defaults."""
        result = validate_pass1_response({"hedging_preserved": True})
        assert result["hedging_preserved"] is True
        assert result["mast_failure_mode"] == "unknown"
        assert result["uncertainty_transfer"] == "unknown"

    def test_validate_pass2_fills_defaults(self):
        """Pass 2 validator fills missing fields."""
        result = validate_pass2_response({})
        assert result["semantic_overlap_estimate"] == 0.5
        assert result["novel_claims_in_other"] == 0

    def test_validate_pass3_fills_defaults(self):
        """Pass 3 validator fills missing fields."""
        result = validate_pass3_response({"chain_fidelity": 0.8})
        assert result["chain_fidelity"] == 0.8
        assert result["information_accretion"] is False

    def test_validate_pass4_fills_defaults(self):
        """Pass 4 validator fills missing fields."""
        result = validate_pass4_response({})
        assert result["confidence_trajectory"] == "stable"
        assert result["fabrication_risk"] == "none"


# =========================================================================
# Tests: Helper Functions
# =========================================================================


class TestHelperFunctions:
    """Test event extraction helpers."""

    def test_extract_delegation_edges_explicit(self):
        """Explicit delegation pairs are extracted."""
        events = [
            {"event_type": "delegation.initiated", "delegation_id": "d1",
             "source_node": "A", "timestamp": "2025-01-01T00:00:01Z"},
            {"event_type": "delegation.completed", "delegation_id": "d1",
             "target_node": "B", "timestamp": "2025-01-01T00:00:02Z"},
        ]
        edges = extract_delegation_edges(events)
        assert len(edges) >= 1
        assert "initiated" in edges[0]

    def test_extract_node_outputs(self):
        """Extracts last output preview per node."""
        events = [
            {"event_type": "agent.task_end", "node_id": "A",
             "output_preview": "first", "timestamp": "1"},
            {"event_type": "agent.task_end", "node_id": "A",
             "output_preview": "second", "timestamp": "2"},
        ]
        outputs = extract_node_outputs(events)
        assert outputs["A"] == "second"

    def test_build_delegation_chain(self):
        """Chain is built from task_end events in order."""
        events = make_chain_events(["Alpha", "Beta", "Gamma"])
        chain = build_delegation_chain(events)
        assert len(chain) == 3
        assert "Alpha" in chain[0]["node_id"]
        assert "Gamma" in chain[2]["node_id"]


# =========================================================================
# Run
# =========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
