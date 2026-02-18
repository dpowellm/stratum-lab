"""Tests for privacy topology analysis (V4 research enrichment).

Exercises stratum_lab/privacy.py: data domain classification, information
fan-in, cross-domain flow detection, and end-to-end privacy topology scoring.
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

from stratum_lab.privacy import (
    classify_data_domain,
    compute_node_data_domains,
    compute_information_fan_in,
    detect_cross_domain_flows,
    compute_privacy_topology,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_edges(pairs: list[tuple]) -> list[dict]:
    """Create edge dicts with edge_id, source, target."""
    return [
        {"edge_id": f"{src}->{tgt}", "source": src, "target": tgt}
        for src, tgt in pairs
    ]


def _make_nodes(names: list[str]) -> list[dict]:
    """Create node dicts with node_id."""
    return [{"node_id": n} for n in names]


def _make_state_events(accesses: list[tuple]) -> list[dict]:
    """Create state.access events from (accessor, key) tuples."""
    return [
        {"event_type": "state.access", "accessor_node": acc, "state_key": key}
        for acc, key in accesses
    ]


# ---------------------------------------------------------------------------
# Domain classification tests
# ---------------------------------------------------------------------------
class TestClassifyDataDomain:
    def test_classify_single_domain(self):
        result = classify_data_domain("email_sender")
        assert "communication_content" in result

    def test_classify_multi_domain(self):
        result = classify_data_domain("user_health_record")
        assert "personal_identifiable" in result
        assert "health_medical" in result

    def test_classify_generic_fallback(self):
        result = classify_data_domain("process_data")
        assert result == ["generic"]


# ---------------------------------------------------------------------------
# Fan-in tests
# ---------------------------------------------------------------------------
class TestFanIn:
    def test_fan_in_single_source(self):
        nodes = _make_nodes(["A", "B"])
        edges = _make_edges([("A", "B")])
        node_domains = {"A": ["financial"], "B": ["financial"]}
        fi = compute_information_fan_in("B", edges, node_domains)
        assert fi["fan_in_risk"] == "low"
        assert fi["incoming_edge_count"] == 1

    def test_fan_in_high_risk(self):
        nodes = _make_nodes(["A", "B", "C", "D"])
        edges = _make_edges([("A", "D"), ("B", "D"), ("C", "D")])
        node_domains = {
            "A": ["personal_identifiable"],
            "B": ["financial"],
            "C": ["health_medical"],
            "D": ["generic"],
        }
        fi = compute_information_fan_in("D", edges, node_domains)
        assert fi["fan_in_risk"] == "high"
        assert fi["domain_count"] >= 3
        assert fi["heterogeneous"] is True


# ---------------------------------------------------------------------------
# Cross-domain flow tests
# ---------------------------------------------------------------------------
class TestCrossDomainFlows:
    def test_cross_domain_flow_detected(self):
        edges = _make_edges([("fin_agent", "pii_agent")])
        node_domains = {
            "fin_agent": ["financial"],
            "pii_agent": ["personal_identifiable"],
        }
        flows = detect_cross_domain_flows(edges, node_domains)
        assert len(flows) == 1
        assert flows[0]["cross_domain"] is True
        assert "personal_identifiable" in flows[0]["new_domains_at_target"]

    def test_no_cross_domain_same_type(self):
        edges = _make_edges([("fin_a", "fin_b")])
        node_domains = {
            "fin_a": ["financial"],
            "fin_b": ["financial"],
        }
        flows = detect_cross_domain_flows(edges, node_domains)
        assert len(flows) == 0


# ---------------------------------------------------------------------------
# Score and finding tests
# ---------------------------------------------------------------------------
class TestPrivacyScoring:
    def test_privacy_exposure_score_bounds(self):
        nodes = _make_nodes(["A", "B", "C", "D", "E"])
        edges = _make_edges([("A", "C"), ("B", "C"), ("C", "D"), ("D", "E")])
        tool_regs = {
            "A": ["fetch_user_email"],
            "B": ["query_payment"],
            "C": ["process_data"],
            "D": ["lookup_patient"],
            "E": ["store_result"],
        }
        state_events = _make_state_events([
            ("C", "user_profile"),
            ("D", "patient_records"),
        ])
        result = compute_privacy_topology(nodes, edges, tool_regs, state_events)
        assert 0.0 <= result["privacy_exposure_score"] <= 1.0

    def test_finding_triggered_above_threshold(self):
        # Build a system with lots of cross-domain flows to push score > 0.3
        nodes = _make_nodes(["pii", "fin", "health", "cred", "hub"])
        edges = _make_edges([
            ("pii", "hub"), ("fin", "hub"), ("health", "hub"), ("cred", "hub"),
        ])
        tool_regs = {
            "pii": ["search_user_profile"],
            "fin": ["query_transaction"],
            "health": ["fetch_patient_diagnosis"],
            "cred": ["retrieve_api_key"],
            "hub": ["process_data"],
        }
        result = compute_privacy_topology(nodes, edges, tool_regs, [])
        assert result["finding_triggered"] is True
        assert result["privacy_exposure_score"] > 0.3

    def test_finding_not_triggered_below(self):
        nodes = _make_nodes(["A", "B"])
        edges = _make_edges([("A", "B")])
        tool_regs = {"A": ["read_file"], "B": ["write_file"]}
        result = compute_privacy_topology(nodes, edges, tool_regs, [])
        assert result["finding_triggered"] is False


# ---------------------------------------------------------------------------
# Node data domain tests
# ---------------------------------------------------------------------------
class TestNodeDataDomains:
    def test_node_data_domains_union(self):
        domains = compute_node_data_domains(
            "agent_1",
            ["fetch_user_email", "query_payment"],
            [],
        )
        assert "personal_identifiable" in domains or "communication_content" in domains
        assert "financial" in domains

    def test_empty_tool_registrations(self):
        nodes = _make_nodes(["A", "B"])
        edges = _make_edges([("A", "B")])
        tool_regs = {}
        result = compute_privacy_topology(nodes, edges, tool_regs, [])
        for nid, domains in result["node_domains"].items():
            assert "generic" in domains

    def test_state_key_classification(self):
        domains = classify_data_domain("patient_records")
        assert "health_medical" in domains


# ---------------------------------------------------------------------------
# Full topology + edge cases
# ---------------------------------------------------------------------------
class TestFullPrivacyTopology:
    def test_full_privacy_topology(self):
        """4-node system with mixed domains."""
        nodes = _make_nodes(["user_agent", "payment_agent", "coordinator", "writer"])
        edges = _make_edges([
            ("user_agent", "coordinator"),
            ("payment_agent", "coordinator"),
            ("coordinator", "writer"),
        ])
        tool_regs = {
            "user_agent": ["fetch_user_profile", "search_user_email"],
            "payment_agent": ["query_transaction", "get_balance"],
            "coordinator": ["process_data"],
            "writer": ["write_file"],
        }
        state_events = _make_state_events([
            ("coordinator", "user_name"),
            ("coordinator", "account_balance"),
        ])

        result = compute_privacy_topology(nodes, edges, tool_regs, state_events)

        # Coordinator should have fan-in from multiple domains
        fan_in_nodes = [fi["node_id"] for fi in result["fan_in_analysis"]]
        assert "coordinator" in fan_in_nodes or result["cross_domain_edge_count"] > 0

        # Should have cross-domain flows
        assert result["cross_domain_edge_count"] >= 1

        # All expected keys present
        assert "node_domains" in result
        assert "privacy_exposure_score" in result
        assert "finding_triggered" in result

    def test_compute_privacy_handles_missing_nodes(self):
        """Edge references a node not in the node list — should not crash."""
        nodes = _make_nodes(["A"])
        edges = _make_edges([("A", "B"), ("C", "A")])
        tool_regs = {"A": ["read_data"]}
        result = compute_privacy_topology(nodes, edges, tool_regs, [])
        assert "privacy_exposure_score" in result

    def test_heterogeneous_detection(self):
        """2+ non-generic domains → heterogeneous = True."""
        edges = _make_edges([("pii_src", "hub"), ("fin_src", "hub")])
        node_domains = {
            "pii_src": ["personal_identifiable"],
            "fin_src": ["financial"],
            "hub": ["generic"],
        }
        fi = compute_information_fan_in("hub", edges, node_domains)
        assert fi["heterogeneous"] is True
        assert fi["domain_count"] >= 2
