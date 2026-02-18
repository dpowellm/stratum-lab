"""Tests for audit readiness and DTMC risk model (V4 research enrichment).

Exercises stratum_lab/audit_readiness.py and scripts/compute_risk_model.py.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, SCRIPT_DIR)
PARENT_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PARENT_DIR)

from stratum_lab.audit_readiness import (
    compute_event_completeness,
    assess_separation_of_duties,
    compute_decision_traceability,
    map_findings_to_regulations,
    compute_audit_readiness,
)
from compute_risk_model import (
    extract_state_sequences,
    learn_dtmc,
    estimate_violation_probability,
    compute_finding_violation_probabilities,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_edges(pairs: list[tuple]) -> list[dict]:
    return [
        {"edge_id": f"{src}->{tgt}", "source": src, "target": tgt}
        for src, tgt in pairs
    ]


def _make_nodes(names: list[str]) -> list[dict]:
    return [{"node_id": n} for n in names]


def _make_paired_events(n_complete: int, n_orphaned: int) -> dict:
    """Create events_by_run with matched and unmatched start/end pairs.

    Includes task, delegation, and LLM event pairs so overall_completeness
    reflects all three categories.
    """
    events: list[dict] = []
    for i in range(n_complete):
        events.append({"event_type": "agent.task_start", "node_id": f"agent_{i}",
                        "timestamp": f"2026-02-16T10:00:{i:02d}Z"})
        events.append({"event_type": "agent.task_end", "node_id": f"agent_{i}",
                        "status": "success", "output_preview": "done",
                        "timestamp": f"2026-02-16T10:01:{i:02d}Z"})
        events.append({"event_type": "delegation.initiated", "delegation_id": f"d_{i}",
                        "timestamp": f"2026-02-16T10:02:{i:02d}Z"})
        events.append({"event_type": "delegation.completed", "delegation_id": f"d_{i}",
                        "timestamp": f"2026-02-16T10:03:{i:02d}Z"})
        events.append({"event_type": "llm.call_start", "node_id": f"llm_{i}",
                        "timestamp": f"2026-02-16T10:04:{i:02d}Z"})
        events.append({"event_type": "llm.call_end", "node_id": f"llm_{i}",
                        "timestamp": f"2026-02-16T10:05:{i:02d}Z"})
    for i in range(n_orphaned):
        events.append({"event_type": "agent.task_start", "node_id": f"orphan_{i}",
                        "timestamp": f"2026-02-16T10:06:{i:02d}Z"})
        events.append({"event_type": "delegation.initiated", "delegation_id": f"d_orphan_{i}",
                        "timestamp": f"2026-02-16T10:07:{i:02d}Z"})
        events.append({"event_type": "llm.call_start", "node_id": f"llm_orphan_{i}",
                        "timestamp": f"2026-02-16T10:08:{i:02d}Z"})
    return {"run_1": events}


def _make_findings(ids: list[str]) -> list[dict]:
    return [{"finding_id": fid, "severity": "medium"} for fid in ids]


# ===========================================================================
# AUDIT READINESS TESTS (10)
# ===========================================================================
class TestEventCompleteness:
    def test_event_completeness_perfect(self):
        """All starts matched → overall_completeness = 1.0."""
        events_by_run = _make_paired_events(5, 0)
        result = compute_event_completeness(events_by_run)
        assert result["task_completion_rate"] == 1.0
        assert result["overall_completeness"] == 1.0
        assert result["orphaned_starts"] == 0

    def test_event_completeness_partial(self):
        """50% matched → overall_completeness reflects partial matching."""
        events_by_run = _make_paired_events(2, 2)
        result = compute_event_completeness(events_by_run)
        assert result["task_completion_rate"] == pytest.approx(0.5, abs=0.01)
        # 2 orphaned × 3 event types (task, delegation, llm) = 6
        assert result["orphaned_starts"] == 6


class TestSeparationOfDuties:
    def test_separation_self_review(self):
        """Node 'ReviewAgent' with output → self_review_detected."""
        nodes = _make_nodes(["ReviewAgent", "Writer"])
        edges = _make_edges([("ReviewAgent", "Writer"), ("Writer", "ReviewAgent")])
        tool_regs = {"ReviewAgent": ["validate_output"], "Writer": ["write_file"]}
        result = assess_separation_of_duties(nodes, edges, tool_regs)
        assert result["self_review_detected"] is True
        assert "ReviewAgent" in result["self_review_nodes"]

    def test_separation_good(self):
        """Generator → Reviewer → Writer pipeline → separation_score high."""
        nodes = _make_nodes(["Generator", "Reviewer", "Writer"])
        edges = _make_edges([("Generator", "Reviewer"), ("Reviewer", "Writer")])
        tool_regs = {
            "Generator": ["generate_report"],
            "Reviewer": ["validate_output"],
            "Writer": ["write_file"],
        }
        result = assess_separation_of_duties(nodes, edges, tool_regs)
        assert result["separation_score"] >= 0.5


class TestDecisionTraceability:
    def test_traceability_with_routing(self):
        """Has routing.decision events → has_routing_decisions = True."""
        events_by_run = {
            "run_1": [
                {"event_type": "routing.decision", "node_id": "router"},
                {"event_type": "agent.task_end", "node_id": "A", "output_preview": "result"},
            ]
        }
        result = compute_decision_traceability(events_by_run)
        assert result["has_routing_decisions"] is True
        assert result["routing_decisions_logged"] >= 1

    def test_traceability_no_outputs(self):
        """No output_preview on any event → traceability_fraction low."""
        events_by_run = {
            "run_1": [
                {"event_type": "agent.task_end", "node_id": "A"},
                {"event_type": "llm.call_end", "node_id": "B"},
            ]
        }
        result = compute_decision_traceability(events_by_run)
        assert result["traceability_fraction"] == 0.0


class TestRegulatoryMapping:
    def test_regulatory_mapping_strat_hc(self):
        """STRAT-HC-001 → eu_ai_act, gdpr."""
        findings = _make_findings(["STRAT-HC-001"])
        result = map_findings_to_regulations(findings)
        assert len(result) == 1
        assert "eu_ai_act" in result[0]["frameworks_implicated"]
        assert "gdpr" in result[0]["frameworks_implicated"]

    def test_regulatory_mapping_strat_pl(self):
        """STRAT-PL-001 → gdpr, hipaa."""
        findings = _make_findings(["STRAT-PL-001"])
        result = map_findings_to_regulations(findings)
        assert "gdpr" in result[0]["frameworks_implicated"]
        assert "hipaa" in result[0]["frameworks_implicated"]


class TestAuditReadinessScore:
    def test_audit_readiness_score_bounds(self):
        """Score always in [0.0, 1.0]."""
        events_by_run = _make_paired_events(3, 1)
        nodes = _make_nodes(["A", "B"])
        edges = _make_edges([("A", "B")])
        tool_regs = {"A": ["search"], "B": ["write_file"]}
        findings = _make_findings(["STRAT-DC-001", "STRAT-HC-001"])
        result = compute_audit_readiness(events_by_run, nodes, edges, tool_regs, findings)
        assert 0.0 <= result["audit_readiness_score"] <= 1.0
        assert "frameworks_at_risk" in result

    def test_frameworks_at_risk(self):
        """Multiple findings → frameworks_at_risk non-empty."""
        events_by_run = _make_paired_events(2, 0)
        nodes = _make_nodes(["A"])
        edges = []
        tool_regs = {"A": ["search"]}
        findings = _make_findings(["STRAT-DC-001", "STRAT-PL-001", "STRAT-AU-001"])
        result = compute_audit_readiness(events_by_run, nodes, edges, tool_regs, findings)
        assert len(result["frameworks_at_risk"]) > 0


# ===========================================================================
# RISK MODEL TESTS (8)
# ===========================================================================
class TestStateSequenceExtraction:
    def test_state_sequence_extraction(self):
        """Events → correct state string sequence."""
        events = [
            {"event_type": "agent.task_start", "node_id": "fw:Planner:main.py:10",
             "timestamp": "2026-02-16T10:00:00Z"},
            {"event_type": "delegation.initiated", "source_node_id": "fw:Planner:main.py:10",
             "target_node_id": "fw:Executor:run.py:20",
             "timestamp": "2026-02-16T10:00:01Z"},
            {"event_type": "agent.task_end", "node_id": "fw:Executor:run.py:20",
             "status": "success",
             "timestamp": "2026-02-16T10:00:02Z"},
            {"event_type": "error.runtime", "node_id": "fw:Monitor:err.py:5",
             "timestamp": "2026-02-16T10:00:03Z"},
        ]
        seq = extract_state_sequences(events)
        assert len(seq) == 4
        assert seq[0] == "start:Planner"
        assert "delegate:" in seq[1]
        assert seq[2] == "success:Executor"
        assert seq[3] == "error:Monitor"


class TestDTMCLearning:
    def test_dtmc_learning_basic(self):
        """3 sequences → transition_matrix rows sum to ~1.0."""
        sequences = [
            ["start:A", "delegate:A\u2192B", "success:B"],
            ["start:A", "error:A", "fail:A"],
            ["start:A", "delegate:A\u2192B", "error:B"],
        ]
        dtmc = learn_dtmc(sequences)
        assert dtmc["total_sequences"] == 3
        assert len(dtmc["states"]) > 0

        for state, row in dtmc["transition_matrix"].items():
            row_sum = sum(row.values())
            assert row_sum == pytest.approx(1.0, abs=0.01)

    def test_dtmc_laplace_smoothing(self):
        """Unseen transitions get non-zero probability."""
        sequences = [
            ["start:A", "success:A"],
        ]
        dtmc = learn_dtmc(sequences, laplace_alpha=1.0)
        # The transition start:A → success:A should be dominant
        # But all other transitions should be non-zero due to Laplace
        row = dtmc["transition_matrix"]["start:A"]
        assert row["success:A"] > 0
        # Check that unseen transitions are non-zero
        for state in dtmc["states"]:
            assert row[state] > 0


class TestViolationProbability:
    def test_violation_probability_safe(self):
        """All transitions safe → probability ≈ 0.0 (only Laplace noise)."""
        sequences = [
            ["start:A", "success:A"],
            ["start:A", "success:A"],
            ["start:A", "success:A"],
        ]
        dtmc = learn_dtmc(sequences, laplace_alpha=0.01)
        p = estimate_violation_probability(dtmc, ["error:X", "fail:X"], "start:A", horizon=5)
        # With Laplace alpha very small and no error states in data,
        # probability should be very low
        assert p < 0.3

    def test_violation_probability_risky(self):
        """Direct path to error → probability > 0.5."""
        sequences = [
            ["start:A", "error:A"],
            ["start:A", "error:A"],
            ["start:A", "error:A"],
            ["start:A", "error:A"],
            ["start:A", "success:A"],
        ]
        dtmc = learn_dtmc(sequences, laplace_alpha=0.1)
        p = estimate_violation_probability(dtmc, ["error:A"], "start:A", horizon=5)
        assert p > 0.5

    def test_violation_probability_unknown_state(self):
        """Unknown current_state → returns 0.0."""
        sequences = [["start:A", "success:A"]]
        dtmc = learn_dtmc(sequences)
        p = estimate_violation_probability(dtmc, ["error:A"], "nonexistent_state", horizon=5)
        assert p == 0.0


class TestPACBound:
    def test_pac_confidence_bound(self):
        """Bound decreases with more sequences."""
        seq_small = [["start:A", "error:A"] for _ in range(5)]
        seq_large = [["start:A", "error:A"] for _ in range(100)]

        dtmc_small = learn_dtmc(seq_small)
        dtmc_large = learn_dtmc(seq_large)

        probs_small = compute_finding_violation_probabilities(dtmc_small)
        probs_large = compute_finding_violation_probabilities(dtmc_large)

        # Confidence bound should be smaller for larger sample
        bound_small = probs_small["STRAT-DC-001"]["confidence_bound"]
        bound_large = probs_large["STRAT-DC-001"]["confidence_bound"]
        assert bound_large < bound_small


class TestFindingViolationProbabilities:
    def test_finding_violation_probabilities(self):
        """Returns dict keyed by finding_id, each value has mean/max/min."""
        sequences = [
            ["start:A", "delegate:A\u2192B", "error:B"],
            ["start:A", "success:A"],
        ]
        dtmc = learn_dtmc(sequences)
        result = compute_finding_violation_probabilities(dtmc)

        assert "STRAT-DC-001" in result
        assert "STRAT-HC-001" in result
        assert "STRAT-CE-001" in result

        for fid, data in result.items():
            assert "mean_violation_probability" in data
            assert "max_violation_probability" in data
            assert "min_violation_probability" in data
            assert "confidence_bound" in data
            assert "horizon" in data
            assert 0.0 <= data["mean_violation_probability"] <= 1.0
