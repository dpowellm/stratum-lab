"""Tests for permission blast radius and cost risk scoring (V4 research enrichment).

Exercises stratum_lab/permissions.py and stratum_lab/cost_risk.py.
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

from stratum_lab.permissions import (
    classify_tool_permissions,
    build_direct_permissions,
    compute_transitive_permissions,
    find_permission_asymmetries,
    compute_unused_permissions,
    compute_permission_blast_radius,
)
from stratum_lab.cost_risk import (
    compute_token_amplification,
    compute_tool_call_density,
    compute_retry_waste,
    compute_latency_profile,
    compute_cost_risk,
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


def _make_tool_regs(mapping: dict) -> dict:
    """Create tool_registrations dict."""
    return dict(mapping)


def _make_runtime_calls(mapping: dict) -> dict:
    """Create runtime_tool_calls dict."""
    return dict(mapping)


def _make_run_events(events: list[dict]) -> list[dict]:
    """Create event list with synthetic timestamps."""
    result = []
    for i, evt in enumerate(events):
        e = dict(evt)
        if "timestamp" not in e:
            e["timestamp"] = f"2026-02-16T10:00:{i:02d}.000Z"
        result.append(e)
    return result


# ===========================================================================
# PERMISSION TESTS (10)
# ===========================================================================
class TestClassifyToolPermissions:
    def test_classify_read_tool(self):
        result = classify_tool_permissions("search_documents")
        assert "read_data" in result

    def test_classify_write_tool(self):
        result = classify_tool_permissions("update_database")
        assert "write_data" in result
        assert "database" in result

    def test_classify_dangerous_tool(self):
        result = classify_tool_permissions("execute_shell")
        assert "execute_code" in result


class TestTransitiveClosure:
    def test_transitive_closure_simple(self):
        """A→B, B has 'database' → A effective includes 'database'."""
        tool_regs = {"A": ["search_data"], "B": ["sql_query"]}
        direct = build_direct_permissions(tool_regs)
        edges = _make_edges([("A", "B")])
        transitive = compute_transitive_permissions(direct, edges)
        assert "database" in transitive["A"]["effective_permissions"]

    def test_transitive_closure_chain(self):
        """A→B→C, C has 'system_admin' → A effective includes 'system_admin'."""
        tool_regs = {"A": ["read_file"], "B": ["process_data"], "C": ["admin_deploy"]}
        direct = build_direct_permissions(tool_regs)
        edges = _make_edges([("A", "B"), ("B", "C")])
        transitive = compute_transitive_permissions(direct, edges)
        assert "system_admin" in transitive["A"]["effective_permissions"]


class TestEscalation:
    def test_escalation_detected(self):
        """A delegates to B, B has 'execute_code' not in A."""
        tool_regs = {"A": ["search_data"], "B": ["execute_script"]}
        direct = build_direct_permissions(tool_regs)
        edges = _make_edges([("A", "B")])
        transitive = compute_transitive_permissions(direct, edges)
        assert "execute_code" in transitive["A"]["escalated_permissions"]

    def test_no_escalation_same_perms(self):
        """A and B have same permissions → no escalation."""
        tool_regs = {"A": ["search_data"], "B": ["find_data"]}
        direct = build_direct_permissions(tool_regs)
        edges = _make_edges([("A", "B")])
        transitive = compute_transitive_permissions(direct, edges)
        assert transitive["A"]["escalation_count"] == 0


class TestAsymmetries:
    def test_asymmetry_risk_critical(self):
        """Escalation includes 'system_admin' → risk_level = 'critical'."""
        tool_regs = {"A": ["search_data"], "B": ["admin_deploy"]}
        direct = build_direct_permissions(tool_regs)
        edges = _make_edges([("A", "B")])
        transitive = compute_transitive_permissions(direct, edges)
        asymmetries = find_permission_asymmetries(transitive, edges)
        assert len(asymmetries) >= 1
        assert asymmetries[0]["risk_level"] == "critical"


class TestUnusedPermissions:
    def test_unused_permissions_detected(self):
        """Agent registers 5 tool categories, uses 2 → low utilization."""
        tool_regs = {"A": ["search_data", "update_database", "execute_shell", "http_request", "admin_deploy"]}
        direct = build_direct_permissions(tool_regs)
        runtime = {"A": ["search_data", "update_database"]}
        unused = compute_unused_permissions(direct, runtime)
        assert unused["A"]["utilization_rate"] < 1.0


class TestPermissionBlastRadius:
    def test_permission_blast_radius_score(self):
        """System with critical asymmetries → score > 0.3."""
        nodes = _make_nodes(["reader", "executor", "admin"])
        edges = _make_edges([("reader", "executor"), ("executor", "admin")])
        tool_regs = {
            "reader": ["search_data"],
            "executor": ["execute_shell"],
            "admin": ["admin_deploy", "sudo_command"],
        }
        runtime = {"reader": ["search_data"], "executor": ["execute_shell"], "admin": ["admin_deploy"]}
        result = compute_permission_blast_radius(nodes, edges, tool_regs, runtime)
        assert result["permission_risk_score"] > 0.0
        assert result["finding_triggered"] is True or result["critical_asymmetries"] > 0


# ===========================================================================
# COST RISK TESTS (10)
# ===========================================================================
class TestTokenAmplification:
    def test_token_amplification_basic(self):
        """2-node chain, 100 input → 2000 total → ratio = 20.0."""
        events = _make_run_events([
            {"event_type": "llm.call_end", "node_id": "A", "prompt_tokens": 100, "completion_tokens": 200},
            {"event_type": "delegation.initiated", "source_node_id": "A", "target_node_id": "B", "delegation_id": "d1"},
            {"event_type": "llm.call_end", "node_id": "B", "prompt_tokens": 800, "completion_tokens": 900},
        ])
        result = compute_token_amplification([events])
        assert len(result["chains"]) >= 1
        assert result["max_amplification_ratio"] > 1.0

    def test_token_amplification_single_node(self):
        """No delegation chain → no amplification entry."""
        events = _make_run_events([
            {"event_type": "llm.call_end", "node_id": "A", "prompt_tokens": 100, "completion_tokens": 50},
        ])
        result = compute_token_amplification([events])
        assert len(result["chains"]) == 0


class TestToolCallDensity:
    def test_tool_call_density(self):
        """Node with 10 LLM calls and 1 delegation → density = 10.0."""
        events = _make_run_events(
            [{"event_type": "llm.call_end", "node_id": "A"} for _ in range(10)]
            + [{"event_type": "delegation.initiated", "source_node_id": "A", "target_node_id": "B", "source_node": "A"}]
        )
        result = compute_tool_call_density([events])
        assert "A" in result["per_node"]
        assert result["per_node"]["A"]["mean_llm_calls"] == 10.0


class TestRetryWaste:
    def test_retry_waste_detection(self):
        """3 errors out of 10 calls → retry_rate ≈ 0.3."""
        events = _make_run_events(
            [{"event_type": "llm.call_end", "node_id": "A", "prompt_tokens": 100} for _ in range(10)]
            + [{"event_type": "error.llm_api", "node_id": "A"} for _ in range(3)]
        )
        result = compute_retry_waste([events])
        assert "A" in result["per_node"]
        assert result["per_node"]["A"]["retry_rate"] == pytest.approx(0.3, abs=0.05)

    def test_retry_waste_zero_errors(self):
        """No errors → retry_rate = 0.0, wasted_tokens = 0."""
        events = _make_run_events([
            {"event_type": "llm.call_end", "node_id": "A", "prompt_tokens": 100},
        ])
        result = compute_retry_waste([events])
        assert result["corpus_retry_rate"] == 0.0
        assert result["total_estimated_wasted_tokens"] == 0


class TestLatencyProfile:
    def test_latency_profile(self):
        """Bottleneck node has highest mean_latency_ms."""
        events = _make_run_events([
            {"event_type": "llm.call_end", "node_id": "fast", "latency_ms": 100},
            {"event_type": "llm.call_end", "node_id": "slow", "latency_ms": 5000},
            {"event_type": "llm.call_end", "node_id": "fast", "latency_ms": 150},
        ])
        result = compute_latency_profile([events])
        assert result["bottleneck_node"] == "slow"
        assert result["bottleneck_latency_ms"] == 5000.0


class TestCostRiskScoring:
    def test_cost_risk_score_bounds(self):
        """Score always in [0.0, 1.0]."""
        events = _make_run_events([
            {"event_type": "llm.call_end", "node_id": "A", "prompt_tokens": 1000, "completion_tokens": 500, "latency_ms": 200},
            {"event_type": "delegation.initiated", "source_node_id": "A", "target_node_id": "B", "delegation_id": "d1"},
            {"event_type": "llm.call_end", "node_id": "B", "prompt_tokens": 5000, "completion_tokens": 3000, "latency_ms": 1000},
        ])
        result = compute_cost_risk([events], {"run1": events})
        assert 0.0 <= result["cost_risk_score"] <= 1.0

    def test_cost_finding_triggered(self):
        """High amplification + high retry → finding_triggered = True."""
        events = _make_run_events(
            [{"event_type": "llm.call_end", "node_id": "A", "prompt_tokens": 50, "completion_tokens": 50, "latency_ms": 100}]
            + [{"event_type": "delegation.initiated", "source_node_id": "A", "target_node_id": "B", "delegation_id": "d1"}]
            + [{"event_type": "llm.call_end", "node_id": "B", "prompt_tokens": 5000, "completion_tokens": 5000, "latency_ms": 500} for _ in range(5)]
            + [{"event_type": "error.llm_api", "node_id": "B"} for _ in range(10)]
        )
        result = compute_cost_risk([events], {"run1": events})
        assert result["cost_risk_score"] > 0.0

    def test_cost_finding_not_triggered(self):
        """Low amplification, no retries → finding_triggered = False."""
        events = _make_run_events([
            {"event_type": "llm.call_end", "node_id": "A", "prompt_tokens": 100, "completion_tokens": 50, "latency_ms": 100},
        ])
        result = compute_cost_risk([events], {"run1": events})
        assert result["finding_triggered"] is False

    def test_monthly_cost_multiplier(self):
        """Check multiplier formula: amp_ratio * (1 + retry_rate)."""
        events = _make_run_events([
            {"event_type": "llm.call_end", "node_id": "A", "prompt_tokens": 100, "completion_tokens": 100, "latency_ms": 100},
            {"event_type": "delegation.initiated", "source_node_id": "A", "target_node_id": "B", "delegation_id": "d1"},
            {"event_type": "llm.call_end", "node_id": "B", "prompt_tokens": 500, "completion_tokens": 500, "latency_ms": 200},
        ])
        result = compute_cost_risk([events], {"run1": events})
        amp = result["token_amplification"]["max_amplification_ratio"]
        retry = result["retry_waste"]["corpus_retry_rate"]
        expected = round(amp * (1 + retry), 2)
        assert result["estimated_monthly_cost_multiplier"] == expected
