"""Integration tests for the stratum-cli → stratum-lab bridge.

Tests the full chain:
  1. Scan result (stratum-cli format) → bridge → scorer-ready dict
  2. Patcher-format events → cost_risk → correct field extraction
  3. Node ID matching between structural and runtime IDs
"""
from __future__ import annotations

import os
import sys

import pytest

PARENT_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PARENT_DIR)

from stratum_lab.bridge.scan_schema import (
    adapt_scan_result,
    _extract_graph_edges,
    _extract_taxonomy_preconditions,
    _compute_risk_surface,
    _resolve_archetype_id,
)
from stratum_lab.selection.scorer import (
    compute_structural_value,
    compute_runnability,
    score_repo,
)
from stratum_lab.selection.schema import validate_selection_input
from stratum_lab.cost_risk import (
    compute_token_amplification,
    compute_tool_call_density,
    compute_retry_waste,
    compute_latency_profile,
    compute_cost_risk,
)
from stratum_lab.node_ids import (
    match_runtime_to_structural,
    structural_agent_id,
    runtime_node_id,
)


# ---------------------------------------------------------------------------
# Fixtures: realistic scan result (stratum-cli format)
# ---------------------------------------------------------------------------

REALISTIC_SCAN_RESULT = {
    "repo_id": "test-org/multi-agent-crew",
    "repo_url": "https://github.com/test-org/multi-agent-crew",
    "repo_full_name": "test-org/multi-agent-crew",
    "graph": {
        "nodes": {
            "agent_researcher": {
                "node_type": "agent",
                "node_name": "Researcher",
                "source_file": "agents.py",
                "line_number": 10,
            },
            "agent_writer": {
                "node_type": "agent",
                "node_name": "Writer",
                "source_file": "agents.py",
                "line_number": 30,
            },
            "cap_search_tool": {
                "node_type": "capability",
                "node_name": "SearchTool",
            },
            "ext_openai_api": {
                "node_type": "external",
                "node_name": "OpenAI API",
            },
        },
        "edges": {
            "e1": {
                "source": "agent_researcher",
                "target": "agent_writer",
                "edge_type": "delegates_to",
            },
            "e2": {
                "source": "agent_researcher",
                "target": "cap_search_tool",
                "edge_type": "calls",
            },
            "e3": {
                "source": "agent_writer",
                "target": "ext_openai_api",
                "edge_type": "sends_to",
            },
        },
    },
    "findings": [
        {"finding_id": "STRAT-DC-001", "severity": "high"},
        {"finding_id": "STRAT-SI-001", "severity": "medium"},
    ],
    "detected_frameworks": ["crewai"],
    "agent_definitions": [
        {"name": "Researcher", "tool_names": ["search_web", "read_file"]},
        {"name": "Writer", "tool_names": ["write_file"]},
    ],
    "control_inventory": {
        "present_controls": ["timeout"],
        "absent_controls": ["rate_limit", "output_validation"],
    },
    "files": ["main.py", "agents.py", "tasks.py", "requirements.txt"],
    "metadata": {"has_entry_point": True, "has_requirements": True},
}


# Realistic patcher-format events (nested payload, source_node dict)
PATCHER_FORMAT_EVENTS = [
    {
        "event_id": "evt-001",
        "timestamp_ns": 1000000000,
        "run_id": "run-1",
        "repo_id": "test-org/multi-agent-crew",
        "event_type": "llm.call_end",
        "source_node": {"node_type": "agent", "node_id": "crewai:Researcher:agents.py:10", "node_name": "Researcher"},
        "payload": {
            "input_tokens": 500,
            "output_tokens": 200,
            "latency_ms": 1500.0,
            "model_requested": "gpt-4",
            "model_actual": "Qwen/Qwen2.5-72B-Instruct",
        },
        "timestamp": "2026-02-16T10:00:00.000Z",
    },
    {
        "event_id": "evt-002",
        "timestamp_ns": 2000000000,
        "run_id": "run-1",
        "repo_id": "test-org/multi-agent-crew",
        "event_type": "delegation.initiated",
        "source_node": {"node_type": "agent", "node_id": "crewai:Researcher:agents.py:10", "node_name": "Researcher"},
        "target_node": {"node_type": "agent", "node_id": "crewai:Writer:agents.py:30", "node_name": "Writer"},
        "edge_type": "delegates_to",
        "payload": {"delegator": "Researcher", "delegate": "Writer"},
        "timestamp": "2026-02-16T10:00:01.000Z",
    },
    {
        "event_id": "evt-003",
        "timestamp_ns": 3000000000,
        "run_id": "run-1",
        "repo_id": "test-org/multi-agent-crew",
        "event_type": "llm.call_end",
        "source_node": {"node_type": "agent", "node_id": "crewai:Writer:agents.py:30", "node_name": "Writer"},
        "payload": {
            "input_tokens": 2000,
            "output_tokens": 1000,
            "latency_ms": 3500.0,
            "model_requested": "gpt-4",
            "model_actual": "Qwen/Qwen2.5-72B-Instruct",
        },
        "timestamp": "2026-02-16T10:00:05.000Z",
    },
    {
        "event_id": "evt-004",
        "timestamp_ns": 4000000000,
        "run_id": "run-1",
        "repo_id": "test-org/multi-agent-crew",
        "event_type": "tool.call_end",
        "source_node": {"node_type": "agent", "node_id": "crewai:Researcher:agents.py:10", "node_name": "Researcher"},
        "payload": {"tool_name": "search_web", "latency_ms": 200.0},
        "timestamp": "2026-02-16T10:00:02.000Z",
    },
]


# ===========================================================================
# BRIDGE TESTS
# ===========================================================================

class TestAdaptScanResult:
    def test_adapt_produces_scorer_fields(self):
        """Bridge output has all fields the scorer expects."""
        adapted = adapt_scan_result(REALISTIC_SCAN_RESULT)
        assert "agent_definitions" in adapted
        assert "graph_edges" in adapted
        assert "taxonomy_preconditions" in adapted
        assert "risk_surface" in adapted
        assert "archetype_id" in adapted
        assert "detected_frameworks" in adapted
        assert "detected_entry_point" in adapted
        assert "detected_requirements" in adapted

    def test_adapt_produces_schema_fields(self):
        """Bridge output passes selection schema validation."""
        adapted = adapt_scan_result(REALISTIC_SCAN_RESULT)
        valid, missing = validate_selection_input(adapted)
        assert valid, f"Missing fields: {missing}"

    def test_graph_edges_extracted(self):
        """Graph edges are extracted from graph.edges dict."""
        edges = _extract_graph_edges(REALISTIC_SCAN_RESULT)
        assert len(edges) == 3
        edge_types = {e["edge_type"] for e in edges}
        assert "delegates_to" in edge_types

    def test_taxonomy_preconditions_mapped(self):
        """Finding IDs are mapped back to precondition names."""
        preconditions = _extract_taxonomy_preconditions(REALISTIC_SCAN_RESULT)
        assert "unbounded_delegation_depth" in preconditions
        assert "no_error_propagation_strategy" in preconditions

    def test_risk_surface_computed(self):
        """Risk surface metrics are computed from edges."""
        edges = _extract_graph_edges(REALISTIC_SCAN_RESULT)
        risk = _compute_risk_surface(REALISTIC_SCAN_RESULT, edges)
        assert risk["max_delegation_depth"] >= 1
        assert "shared_state_conflict_count" in risk
        assert "feedback_loop_count" in risk
        assert risk["trust_boundary_crossing_count"] >= 1  # ext_openai_api edge

    def test_archetype_resolution(self):
        """Archetype ID is resolved for multi-agent system."""
        aid = _resolve_archetype_id(REALISTIC_SCAN_RESULT)
        assert isinstance(aid, int)
        assert aid > 0

    def test_scorer_accepts_adapted(self):
        """The scorer functions accept the adapted dict without error."""
        adapted = adapt_scan_result(REALISTIC_SCAN_RESULT)
        sv = compute_structural_value(adapted)
        assert sv > 0
        rn = compute_runnability(adapted)
        assert rn > 0

    def test_score_repo_end_to_end(self):
        """Full scoring pipeline: scan result → bridge → score."""
        adapted = adapt_scan_result(REALISTIC_SCAN_RESULT)
        result = score_repo(adapted, archetype_counts={}, selection_target=100)
        assert result["selection_score"] > 0
        assert result["framework"] == "crewai"
        assert result["agent_count"] == 2

    def test_empty_scan_result(self):
        """Bridge handles empty/minimal scan result gracefully."""
        adapted = adapt_scan_result({})
        assert adapted["agent_count"] == 0
        assert adapted["graph_edges"] == []
        assert adapted["taxonomy_preconditions"] == []


# ===========================================================================
# PATCHER-FORMAT EVENT TESTS (cost_risk reads correct fields)
# ===========================================================================

class TestPatcherEventFormat:
    def test_token_amplification_nested_events(self):
        """Token amplification works with patcher-format nested events."""
        result = compute_token_amplification([PATCHER_FORMAT_EVENTS])
        assert len(result["chains"]) >= 1
        chain = result["chains"][0]
        assert chain["total_chain_tokens"] > 0
        assert chain["amplification_ratio"] > 1.0

    def test_tool_call_density_nested_events(self):
        """Tool call density works with patcher-format nested events."""
        result = compute_tool_call_density([PATCHER_FORMAT_EVENTS])
        # Should see the Researcher node (has LLM calls and tool calls)
        researcher_id = "crewai:Researcher:agents.py:10"
        assert researcher_id in result["per_node"]
        assert result["per_node"][researcher_id]["mean_llm_calls"] >= 1

    def test_retry_waste_nested_events(self):
        """Retry waste works with patcher-format nested events."""
        events_with_error = PATCHER_FORMAT_EVENTS + [
            {
                "event_type": "error.llm_api",
                "source_node": {"node_type": "agent", "node_id": "crewai:Writer:agents.py:30", "node_name": "Writer"},
                "timestamp": "2026-02-16T10:00:06.000Z",
            },
        ]
        result = compute_retry_waste([events_with_error])
        writer_id = "crewai:Writer:agents.py:30"
        assert writer_id in result["per_node"]

    def test_latency_profile_nested_events(self):
        """Latency profile reads payload.latency_ms correctly."""
        result = compute_latency_profile([PATCHER_FORMAT_EVENTS])
        writer_id = "crewai:Writer:agents.py:30"
        assert writer_id in result["per_node_latency"]
        assert result["per_node_latency"][writer_id]["mean_latency_ms"] == 3500.0
        assert result["bottleneck_node"] == writer_id

    def test_cost_risk_full_pipeline_nested(self):
        """Full cost risk pipeline works with patcher-format events."""
        result = compute_cost_risk(
            [PATCHER_FORMAT_EVENTS],
            {"run-1": PATCHER_FORMAT_EVENTS},
        )
        assert 0.0 <= result["cost_risk_score"] <= 1.0
        assert "token_amplification" in result
        assert "latency_profile" in result


# ===========================================================================
# NODE ID MATCHING TESTS
# ===========================================================================

class TestNodeIdMatching:
    def test_runtime_to_structural_exact(self):
        """Runtime 'crewai:Researcher:...' matches structural 'agent_researcher'."""
        structural_nodes = {
            "agent_researcher": {"node_type": "agent", "node_name": "Researcher"},
            "agent_writer": {"node_type": "agent", "node_name": "Writer"},
        }
        rt_id = runtime_node_id("crewai", "Researcher", "agents.py", 10)
        matched = match_runtime_to_structural(rt_id, structural_nodes)
        assert matched == "agent_researcher"

    def test_runtime_to_structural_source_file(self):
        """Source file + line match when name doesn't match directly."""
        structural_nodes = {
            "agent_custom_name": {
                "node_type": "agent",
                "source_file": "agents.py",
                "line_number": 10,
            },
        }
        rt_id = runtime_node_id("crewai", "MyAgent", "agents.py", 10)
        matched = match_runtime_to_structural(rt_id, structural_nodes)
        assert matched == "agent_custom_name"

    def test_runtime_to_structural_no_match(self):
        """Returns None when no structural node matches."""
        structural_nodes = {
            "agent_researcher": {"node_type": "agent"},
        }
        rt_id = runtime_node_id("crewai", "CompletelyDifferent", "other.py", 99)
        matched = match_runtime_to_structural(rt_id, structural_nodes)
        assert matched is None

    def test_structural_agent_id_consistency(self):
        """structural_agent_id produces correct format."""
        assert structural_agent_id("Researcher") == "agent_researcher"
        assert structural_agent_id("MyCustomAgent") == "agent_my_custom_agent"
        assert structural_agent_id("writer") == "agent_writer"
