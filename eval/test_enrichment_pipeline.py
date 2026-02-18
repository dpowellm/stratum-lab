"""Integration tests for the V4 enrichment pipeline.

Exercises compute_enrichments.py orchestration, cross-module data flow,
finding registration, corpus risk model, and mass-scan edge cases.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, SCRIPT_DIR)
PARENT_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PARENT_DIR)

from compute_enrichments import (
    compute_all_enrichments,
    load_events,
    load_behavioral_record,
    load_data_topology,
    extract_runtime_tool_calls,
    main as enrichments_main,
)
from compute_risk_model import (
    compute_corpus_risk_model,
    extract_state_sequences,
    learn_dtmc,
)
from stratum_lab.privacy import compute_privacy_topology, classify_data_domain
from stratum_lab.permissions import (
    compute_permission_blast_radius,
    classify_tool_permissions,
    compute_transitive_permissions,
    build_direct_permissions,
)
from stratum_lab.cost_risk import compute_token_amplification, compute_cost_risk
from stratum_lab.audit_readiness import (
    compute_audit_readiness,
    compute_event_completeness,
    map_findings_to_regulations,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_chain_events(
    chain: list[str],
    tokens_per_node: int = 500,
    include_errors: bool = False,
    include_tools: list[str] | None = None,
) -> list[dict]:
    """Generate events for a delegation chain."""
    events: list[dict] = []
    ts = [0]

    def next_ts() -> str:
        ts[0] += 1
        s = ts[0]
        return f"2026-01-01T{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}Z"

    for i, node_id in enumerate(chain):
        role = node_id.split(":")[1] if ":" in node_id else node_id
        input_source = chain[i - 1] if i > 0 else "user"

        events.append({
            "event_type": "agent.task_start",
            "node_id": node_id,
            "input_source": input_source,
            "timestamp": next_ts(),
        })
        events.append({
            "event_type": "llm.call_start",
            "node_id": node_id,
            "model": "test-model",
            "message_count": 3,
            "timestamp": next_ts(),
        })

        if include_tools:
            for tool_name in include_tools:
                events.append({
                    "event_type": "tool.call_start",
                    "node_id": node_id,
                    "tool_name": tool_name,
                    "timestamp": next_ts(),
                })
                events.append({
                    "event_type": "tool.call_end",
                    "node_id": node_id,
                    "tool_name": tool_name,
                    "timestamp": next_ts(),
                })

        events.append({
            "event_type": "llm.call_end",
            "node_id": node_id,
            "output_preview": f"Output from {role}",
            "prompt_tokens": tokens_per_node,
            "completion_tokens": tokens_per_node // 2,
            "total_tokens": int(tokens_per_node * 1.5),
            "latency_ms": 500,
            "timestamp": next_ts(),
        })

        if include_errors and i == 1:
            events.append({
                "event_type": "error.tool_failure",
                "node_id": node_id,
                "error_type": "timeout",
                "error_message": "Tool timed out",
                "timestamp": next_ts(),
            })

        events.append({
            "event_type": "agent.task_end",
            "node_id": node_id,
            "output_preview": f"Output from {role}",
            "status": "success",
            "output_hash": "abc123",
            "output_type": "str",
            "output_size_bytes": 100,
            "timestamp": next_ts(),
        })

        if i < len(chain) - 1:
            events.append({
                "event_type": "delegation.initiated",
                "delegation_id": f"del_{i}",
                "source_node_id": node_id,
                "target_node_id": chain[i + 1],
                "timestamp": next_ts(),
            })
            events.append({
                "event_type": "delegation.completed",
                "delegation_id": f"del_{i}",
                "status": "success",
                "timestamp": next_ts(),
            })

    return events


def _make_results_dir(
    nodes=None, edges=None, findings=None,
    events_per_run=None, n_runs=5,
    tool_registrations=None, state_keys=None,
    include_data_topology=True,
) -> str:
    """Create a synthetic results directory mirroring pipeline output."""
    tmpdir = tempfile.mkdtemp(prefix="stratum_test_enrich_")

    if nodes is None:
        nodes = [
            {"node_id": "crewai:Researcher:main.py:10"},
            {"node_id": "crewai:Writer:main.py:20"},
        ]
    if edges is None:
        edges = [
            {"edge_id": "e1",
             "source": "crewai:Researcher:main.py:10",
             "target": "crewai:Writer:main.py:20",
             "edge_type": "delegates_to"},
        ]

    # behavioral_record.json — uses keys compute_enrichments.py reads
    node_activation = {n["node_id"]: {"activated": True} for n in nodes}
    behavioral_record = {
        "schema_version": "v6",
        "edge_validation": {},
        "emergent_edges": edges,
        "node_activation": node_activation,
        "failure_modes": findings or [],
        "semantic_analysis": None,
        "defensive_patterns": None,
        "research_enrichments": None,
    }
    with open(os.path.join(tmpdir, "behavioral_record.json"), "w", encoding="utf-8") as f:
        json.dump(behavioral_record, f, indent=2)

    # data_topology.json
    if include_data_topology:
        data_topology = {
            "tool_registrations": tool_registrations or {},
            "state_keys": state_keys or [],
            "tools_found": sum(len(v) for v in (tool_registrations or {}).values()),
            "patterns_detected": {
                "decorator_tools": 0, "constructor_tools": 0,
                "mcp_tools": 0, "api_patterns": 0,
                "database_patterns": 0, "file_patterns": 0,
            },
        }
        with open(os.path.join(tmpdir, "data_topology.json"), "w", encoding="utf-8") as f:
            json.dump(data_topology, f, indent=2)

    # raw_events/events_run_*.jsonl
    raw_dir = os.path.join(tmpdir, "raw_events")
    os.makedirs(raw_dir, exist_ok=True)

    if events_per_run is None:
        default_chain = [n["node_id"] for n in nodes[:2]] if len(nodes) >= 2 \
            else [nodes[0]["node_id"]] if nodes else ["crewai:Agent:main.py:1"]
        events_per_run = [_make_chain_events(default_chain) for _ in range(n_runs)]

    for i, events in enumerate(events_per_run):
        fpath = os.path.join(raw_dir, f"events_run_{i + 1}.jsonl")
        with open(fpath, "w", encoding="utf-8") as f:
            for evt in events:
                f.write(json.dumps(evt) + "\n")

    return tmpdir


# ===========================================================================
# Section A: Enrichment Orchestrator (7 tests)
# ===========================================================================
class TestComputeAllEnrichments:
    def test_all_four_enrichments_complete(self):
        """Standard 2-node chain with tool registrations → 4/4 enrichments."""
        results_dir = _make_results_dir(
            tool_registrations={
                "crewai:Researcher:main.py:10": ["search_documents", "web_search"],
            },
            state_keys=["research_notes"],
        )
        try:
            result = compute_all_enrichments(results_dir)
            assert result["enrichments_completed"] == 4
            assert result["enrichments_failed"] == 0
            for key in ("privacy_topology", "permission_blast_radius",
                        "cost_risk", "audit_readiness"):
                assert isinstance(result[key], dict)
                assert "error" not in result[key]
            assert isinstance(result["enrichment_timestamp"], str)
            assert len(result["enrichment_timestamp"]) > 0
        finally:
            shutil.rmtree(results_dir)

    def test_missing_data_topology_doesnt_crash(self):
        """No data_topology.json → enrichments still run."""
        results_dir = _make_results_dir(include_data_topology=False)
        try:
            result = compute_all_enrichments(results_dir)
            assert isinstance(result["enrichments_completed"], int)
            assert result["enrichments_completed"] >= 0
            # Cost risk and audit readiness should still work (event-based)
        finally:
            shutil.rmtree(results_dir)

    def test_missing_behavioral_record(self):
        """No behavioral_record.json → no uncaught exception."""
        tmpdir = tempfile.mkdtemp(prefix="stratum_test_enrich_")
        raw_dir = os.path.join(tmpdir, "raw_events")
        os.makedirs(raw_dir)
        events = _make_chain_events([
            "crewai:A:main.py:1", "crewai:B:main.py:2",
        ])
        with open(os.path.join(raw_dir, "events_run_1.jsonl"), "w") as f:
            for evt in events:
                f.write(json.dumps(evt) + "\n")
        try:
            result = compute_all_enrichments(tmpdir)
            assert isinstance(result, dict)
            assert "enrichments_completed" in result
        finally:
            shutil.rmtree(tmpdir)

    def test_crash_isolation(self):
        """One enrichment crashes → others still complete."""
        results_dir = _make_results_dir()
        try:
            with patch(
                "compute_enrichments.compute_privacy_topology",
                side_effect=RuntimeError("injected crash"),
            ):
                result = compute_all_enrichments(results_dir)
            assert "error" in result["privacy_topology"]
            assert "injected crash" in result["privacy_topology"]["error"]
            assert result["enrichments_failed"] >= 1
            assert result["enrichments_completed"] >= 3
            # Other three should be fine
            for key in ("permission_blast_radius", "cost_risk", "audit_readiness"):
                assert "error" not in result[key]
        finally:
            shutil.rmtree(results_dir)

    def test_empty_event_streams(self):
        """Empty event files → no crash."""
        results_dir = _make_results_dir(
            events_per_run=[[] for _ in range(5)],
        )
        try:
            result = compute_all_enrichments(results_dir)
            assert isinstance(result, dict)
            for key in ("privacy_topology", "permission_blast_radius",
                        "cost_risk", "audit_readiness"):
                assert isinstance(result[key], dict)
        finally:
            shutil.rmtree(results_dir)

    def test_writes_enrichments_json(self):
        """main() writes enrichments.json and patches behavioral_record."""
        results_dir = _make_results_dir()
        try:
            with patch("sys.argv", ["compute_enrichments.py", results_dir]):
                enrichments_main()
            enrichments_path = os.path.join(results_dir, "enrichments.json")
            assert os.path.exists(enrichments_path)
            with open(enrichments_path, "r") as f:
                data = json.load(f)
            assert "enrichments_completed" in data
            # behavioral_record.json should be patched
            with open(os.path.join(results_dir, "behavioral_record.json"), "r") as f:
                record = json.load(f)
            assert "research_enrichments" in record
        finally:
            shutil.rmtree(results_dir)

    def test_extract_runtime_tool_calls(self):
        """tool.call_end events → per-node tool name mapping."""
        events_by_run = {
            "run_1": [
                {"event_type": "tool.call_end", "node_id": "A", "tool_name": "search"},
                {"event_type": "tool.call_end", "node_id": "A", "tool_name": "fetch"},
                {"event_type": "tool.call_end", "node_id": "B", "tool_name": "write"},
            ],
        }
        result = extract_runtime_tool_calls(events_by_run)
        assert "search" in result["A"]
        assert "fetch" in result["A"]
        assert "write" in result["B"]


# ===========================================================================
# Section B: Cross-Module Data Flow (5 tests)
# ===========================================================================
class TestDataFlow:
    def test_tool_registrations_to_privacy_domains(self):
        """Tool names → correct data domain classification → cross-domain detected."""
        node_a = "crewai:AgentA:main.py:10"
        node_b = "crewai:AgentB:main.py:20"
        nodes = [{"node_id": node_a}, {"node_id": node_b}]
        edges = [{"edge_id": "e1", "source": node_a, "target": node_b}]
        tool_regs = {
            node_a: ["search_medical_records", "lookup_patient"],
            node_b: ["process_payment", "check_balance"],
        }
        result = compute_privacy_topology(nodes, edges, tool_regs, [])
        assert "health_medical" in result["node_domains"][node_a]
        assert "financial" in result["node_domains"][node_b]
        assert len(result["cross_domain_flows"]) > 0

    def test_tool_registrations_to_permission_blast(self):
        """Delegation to node with execute/write → escalation detected."""
        node_a = "crewai:AgentA:main.py:10"
        node_b = "crewai:AgentB:main.py:20"
        nodes = [{"node_id": node_a}, {"node_id": node_b}]
        edges = [{"edge_id": "e1", "source": node_a, "target": node_b}]
        tool_regs = {
            node_a: ["search_documents"],
            node_b: ["execute_shell_command", "write_database"],
        }
        result = compute_permission_blast_radius(nodes, edges, tool_regs, {})
        a_perms = result["transitive_permissions"][node_a]
        assert "execute_code" in a_perms["effective_permissions"]
        assert "database" in a_perms["effective_permissions"]
        assert result["total_escalation_count"] > 0

    def test_state_keys_to_privacy_domains(self):
        """State access events feed privacy domain classification."""
        node_a = "crewai:AgentA:main.py:10"
        node_b = "crewai:AgentB:main.py:20"
        node_c = "crewai:AgentC:main.py:30"
        node_d = "crewai:AgentD:main.py:40"
        nodes = [{"node_id": n} for n in [node_a, node_b, node_d, node_c]]
        edges = [
            {"edge_id": "e1", "source": node_a, "target": node_c},
            {"edge_id": "e2", "source": node_b, "target": node_c},
            {"edge_id": "e3", "source": node_d, "target": node_c},
        ]
        state_events = [
            {"accessor_node": node_a, "state_key": "patient_diagnosis",
             "event_type": "state.access"},
            {"accessor_node": node_b, "state_key": "salary_data",
             "event_type": "state.access"},
            {"accessor_node": node_d, "state_key": "user_profile",
             "event_type": "state.access"},
        ]
        result = compute_privacy_topology(nodes, edges, {}, state_events)
        assert result["privacy_exposure_score"] > 0
        # AgentC should have high fan-in from 3 heterogeneous domains
        fan_in_nodes = {fi["node_id"] for fi in result["fan_in_analysis"]}
        assert node_c in fan_in_nodes

    def test_events_to_cost_risk(self):
        """Token counts in events → correct amplification ratio."""
        node1 = "crewai:Node1:main.py:10"
        node2 = "crewai:Node2:main.py:20"
        node3 = "crewai:Node3:main.py:30"
        events = [
            {"event_type": "llm.call_end", "node_id": node1,
             "prompt_tokens": 100, "completion_tokens": 200,
             "latency_ms": 100, "timestamp": "2026-01-01T00:00:01Z"},
            {"event_type": "delegation.initiated",
             "source_node_id": node1, "target_node_id": node2,
             "timestamp": "2026-01-01T00:00:02Z"},
            {"event_type": "llm.call_end", "node_id": node2,
             "prompt_tokens": 400, "completion_tokens": 300,
             "latency_ms": 100, "timestamp": "2026-01-01T00:00:03Z"},
            {"event_type": "delegation.initiated",
             "source_node_id": node2, "target_node_id": node3,
             "timestamp": "2026-01-01T00:00:04Z"},
            {"event_type": "llm.call_end", "node_id": node3,
             "prompt_tokens": 600, "completion_tokens": 500,
             "latency_ms": 100, "timestamp": "2026-01-01T00:00:05Z"},
        ]
        # Total = (100+200) + (400+300) + (600+500) = 2100
        # first_input = 100, ratio = 2100/100 = 21.0
        result = compute_token_amplification([events])
        assert result["max_amplification_ratio"] == pytest.approx(21.0, abs=1.0)
        assert result["high_amplification_chains"] >= 1

    def test_events_to_audit_completeness(self):
        """Orphaned task starts → correct completion rate."""
        events: list[dict] = []
        for i in range(10):
            events.append({
                "event_type": "agent.task_start",
                "node_id": f"agent_{i}",
                "timestamp": f"2026-01-01T00:00:{i:02d}Z",
            })
        for i in range(6):
            events.append({
                "event_type": "agent.task_end",
                "node_id": f"agent_{i}",
                "status": "success",
                "timestamp": f"2026-01-01T00:01:{i:02d}Z",
            })
        result = compute_event_completeness({"run_1": events})
        assert result["task_completion_rate"] == pytest.approx(0.6, abs=0.01)
        assert result["orphaned_starts"] >= 4


# ===========================================================================
# Section C: New Finding Registration (3 tests)
# ===========================================================================
class TestNewFindings:
    def test_privacy_finding_triggered(self):
        """High privacy exposure → finding_triggered = True."""
        # 3 nodes with different domains converging at a target
        nodes = [{"node_id": n} for n in ["A", "B", "C", "Target"]]
        edges = [
            {"edge_id": "e1", "source": "A", "target": "Target"},
            {"edge_id": "e2", "source": "B", "target": "Target"},
            {"edge_id": "e3", "source": "C", "target": "Target"},
        ]
        tool_regs = {
            "A": ["search_patient_records"],       # health_medical
            "B": ["process_payment"],              # financial
            "C": ["lookup_user_profile"],           # personal_identifiable
            "Target": ["generate_report"],          # generic
        }
        result = compute_privacy_topology(nodes, edges, tool_regs, [])
        assert result["finding_triggered"] is True
        assert result["privacy_exposure_score"] > 0.3

    def test_permission_finding_triggered(self):
        """Critical escalation → finding_triggered = True."""
        nodes = [{"node_id": "A"}, {"node_id": "B"}]
        edges = [{"edge_id": "e1", "source": "A", "target": "B"}]
        tool_regs = {
            "A": ["search_documents"],
            "B": ["execute_shell_command", "admin_deploy"],
        }
        result = compute_permission_blast_radius(nodes, edges, tool_regs, {})
        assert result["finding_triggered"] is True
        assert result["permission_risk_score"] > 0.3

    def test_all_four_v4_findings_via_enrichments(self):
        """All 4 new STRAT findings recognized by map_findings_to_regulations."""
        findings = [
            {"finding_id": "STRAT-PL-001", "severity": "high"},
            {"finding_id": "STRAT-PE-001", "severity": "high"},
            {"finding_id": "STRAT-CR-001", "severity": "medium"},
            {"finding_id": "STRAT-AU-001", "severity": "high"},
        ]
        result = map_findings_to_regulations(findings)
        assert len(result) == 4
        fids = {r["finding_id"] for r in result}
        assert fids == {"STRAT-PL-001", "STRAT-PE-001", "STRAT-CR-001", "STRAT-AU-001"}


# ===========================================================================
# Section D: Regulatory Mapping (2 tests)
# ===========================================================================
class TestRegulatoryMapping:
    def test_privacy_finding_maps_to_gdpr_hipaa(self):
        """STRAT-PL-001 → gdpr, hipaa."""
        result = map_findings_to_regulations(
            [{"finding_id": "STRAT-PL-001", "severity": "high"}],
        )
        assert "gdpr" in result[0]["frameworks_implicated"]
        assert "hipaa" in result[0]["frameworks_implicated"]

    def test_permission_finding_maps_to_sox(self):
        """STRAT-PE-001 → sox."""
        result = map_findings_to_regulations(
            [{"finding_id": "STRAT-PE-001", "severity": "high"}],
        )
        assert "sox" in result[0]["frameworks_implicated"]


# ===========================================================================
# Section E: Corpus Risk Model Integration (4 tests)
# ===========================================================================
def _make_corpus_dir(
    n_repos: int = 3,
    n_runs: int = 5,
    error_repo_indices: set[int] | None = None,
) -> str:
    """Create a corpus directory with multiple repos for risk model tests."""
    tmpdir = tempfile.mkdtemp(prefix="stratum_test_corpus_")
    chain = ["crewai:Researcher:main.py:10", "crewai:Writer:main.py:20"]

    for r in range(n_repos):
        repo_dir = os.path.join(tmpdir, f"repo_{r + 1}")
        os.makedirs(repo_dir)

        record = {"schema_version": "v6", "findings": [], "failure_modes": []}
        with open(os.path.join(repo_dir, "behavioral_record.json"), "w") as f:
            json.dump(record, f)

        include_err = error_repo_indices is not None and r in error_repo_indices
        for run_idx in range(n_runs):
            events = _make_chain_events(chain, include_errors=include_err)
            fpath = os.path.join(repo_dir, f"events_run_{run_idx + 1}.jsonl")
            with open(fpath, "w") as f:
                for evt in events:
                    f.write(json.dumps(evt) + "\n")

    return tmpdir


class TestCorpusRiskModel:
    def test_corpus_model_from_directory(self):
        """3 repos × 5 runs → correct corpus statistics."""
        corpus_dir = _make_corpus_dir(n_repos=3, n_runs=5)
        try:
            result = compute_corpus_risk_model(corpus_dir)
            assert result["corpus_statistics"]["total_repos"] == 3
            assert result["corpus_statistics"]["total_sequences"] >= 15
            assert result["corpus_statistics"]["unique_states"] >= 2
            assert "states" in result["dtmc"] or "states_count" in result["dtmc"]
            assert isinstance(result["violation_probabilities"], dict)
        finally:
            shutil.rmtree(corpus_dir)

    def test_corpus_model_empty_directory(self):
        """Empty directory → no crash, zero repos."""
        tmpdir = tempfile.mkdtemp(prefix="stratum_test_corpus_")
        try:
            result = compute_corpus_risk_model(tmpdir)
            assert result["corpus_statistics"]["total_repos"] == 0
            assert result["corpus_statistics"]["total_sequences"] == 0
        finally:
            shutil.rmtree(tmpdir)

    def test_corpus_model_includes_error_repos(self):
        """Mix of clean and error repos → error states in DTMC."""
        corpus_dir = _make_corpus_dir(n_repos=3, n_runs=5, error_repo_indices={2})
        try:
            result = compute_corpus_risk_model(corpus_dir)
            assert result["corpus_statistics"]["total_repos"] == 3
            states = result["dtmc"].get("states", [])
            error_states = [s for s in states if s.startswith("error:")]
            assert len(error_states) > 0
        finally:
            shutil.rmtree(corpus_dir)

    def test_corpus_model_top_transitions(self):
        """Identical chains → top_transitions is a non-empty list."""
        corpus_dir = _make_corpus_dir(n_repos=5, n_runs=5)
        try:
            result = compute_corpus_risk_model(corpus_dir)
            top = result["corpus_statistics"]["top_transitions"]
            assert len(top) > 0
            for entry in top:
                assert len(entry) == 3  # [from, to, probability]
                assert 0.0 < entry[2] <= 1.0
        finally:
            shutil.rmtree(corpus_dir)


# ===========================================================================
# Section F: Mass Scan Edge Cases (5 tests)
# ===========================================================================
class TestMassScanEdgeCases:
    def test_generic_tool_names_classify_as_generic(self):
        """Generic tool names → ['generic'], no false domain matches."""
        for name in ("process_data", "handle_input", "run_pipeline"):
            domains = classify_data_domain(name)
            assert domains == ["generic"], f"{name} classified as {domains}"

    def test_hub_spoke_permission_blast_radius(self):
        """Hub delegates to 3 spokes → hub inherits all spoke permissions."""
        hub = "Hub"
        spoke_a = "Spoke_A"
        spoke_b = "Spoke_B"
        spoke_c = "Spoke_C"
        nodes = [{"node_id": n} for n in [hub, spoke_a, spoke_b, spoke_c]]
        edges = [
            {"edge_id": "e1", "source": hub, "target": spoke_a},
            {"edge_id": "e2", "source": hub, "target": spoke_b},
            {"edge_id": "e3", "source": hub, "target": spoke_c},
        ]
        tool_regs = {
            hub: ["list_items"],
            spoke_a: ["execute_python"],
            spoke_b: ["update_records", "write_file"],
            spoke_c: ["send_notification"],
        }
        result = compute_permission_blast_radius(nodes, edges, tool_regs, {})
        hub_trans = result["transitive_permissions"][hub]
        effective = set(hub_trans["effective_permissions"])
        assert effective >= {"read_data", "execute_code", "write_data",
                             "file_system", "user_communication"}
        assert hub_trans["escalation_count"] >= 4

    def test_amplification_threshold_boundary(self):
        """Low cost → not triggered; high cost → triggered."""
        # Low: 2-node chain, modest tokens, 1 LLM call each
        lo_events = [
            {"event_type": "llm.call_end", "node_id": "crewai:A:m.py:1",
             "prompt_tokens": 500, "completion_tokens": 250,
             "latency_ms": 100, "timestamp": "2026-01-01T00:00:01Z"},
            {"event_type": "delegation.initiated",
             "source_node_id": "crewai:A:m.py:1",
             "target_node_id": "crewai:B:m.py:2",
             "timestamp": "2026-01-01T00:00:02Z"},
            {"event_type": "llm.call_end", "node_id": "crewai:B:m.py:2",
             "prompt_tokens": 500, "completion_tokens": 250,
             "latency_ms": 100, "timestamp": "2026-01-01T00:00:03Z"},
        ]
        lo_result = compute_cost_risk([lo_events], {"run_1": lo_events})
        assert lo_result["finding_triggered"] is False

        # High: massive amplification + high density
        hi_events: list[dict] = []
        ts_ctr = 0

        def hi_ts():
            nonlocal ts_ctr
            ts_ctr += 1
            return f"2026-01-01T00:{ts_ctr // 60:02d}:{ts_ctr % 60:02d}Z"

        # Node1: small input
        hi_events.append({"event_type": "llm.call_end",
                          "node_id": "crewai:N1:m.py:1",
                          "prompt_tokens": 10, "completion_tokens": 5,
                          "latency_ms": 100, "timestamp": hi_ts()})
        hi_events.append({"event_type": "delegation.initiated",
                          "source_node_id": "crewai:N1:m.py:1",
                          "target_node_id": "crewai:N2:m.py:2",
                          "timestamp": hi_ts()})
        # Node2: 10 large LLM calls → high density + amplification
        for _ in range(10):
            hi_events.append({"event_type": "llm.call_end",
                              "node_id": "crewai:N2:m.py:2",
                              "prompt_tokens": 200, "completion_tokens": 200,
                              "latency_ms": 200, "timestamp": hi_ts()})

        hi_result = compute_cost_risk([hi_events], {"run_1": hi_events})
        assert hi_result["finding_triggered"] is True

    def test_single_node_no_delegation(self):
        """Single node, no edges → no crash, no escalation, no amplification."""
        node = "crewai:Solo:main.py:10"
        events = [
            {"event_type": "agent.task_start", "node_id": node,
             "timestamp": "2026-01-01T00:00:01Z"},
            {"event_type": "llm.call_start", "node_id": node,
             "model": "test-model", "timestamp": "2026-01-01T00:00:02Z"},
            {"event_type": "llm.call_end", "node_id": node,
             "prompt_tokens": 100, "completion_tokens": 50,
             "latency_ms": 200, "timestamp": "2026-01-01T00:00:03Z"},
            {"event_type": "agent.task_end", "node_id": node,
             "status": "success", "output_preview": "done",
             "timestamp": "2026-01-01T00:00:04Z"},
        ]
        results_dir = _make_results_dir(
            nodes=[{"node_id": node}],
            edges=[],
            events_per_run=[events for _ in range(3)],
            tool_registrations={node: ["search"]},
        )
        try:
            result = compute_all_enrichments(results_dir)
            assert result["enrichments_completed"] >= 3
            privacy = result["privacy_topology"]
            if "error" not in privacy:
                assert len(privacy.get("cross_domain_flows", [])) == 0
            perms = result["permission_blast_radius"]
            if "error" not in perms:
                assert perms.get("total_escalation_count", 0) == 0
            cost = result["cost_risk"]
            if "error" not in cost:
                amp = cost.get("token_amplification", {})
                assert amp.get("high_amplification_chains", 0) == 0
        finally:
            shutil.rmtree(results_dir)

    def test_large_fan_in_privacy_convergence(self):
        """5 domain-specific nodes → 1 aggregator → high fan-in + triggered."""
        health = "crewai:HealthAgent:main.py:10"
        finance = "crewai:FinanceAgent:main.py:20"
        personal = "crewai:PersonalAgent:main.py:30"
        creds = "crewai:CredsAgent:main.py:40"
        comms = "crewai:CommsAgent:main.py:50"
        aggregator = "crewai:Aggregator:main.py:60"

        nodes = [{"node_id": n} for n in
                 [health, finance, personal, creds, comms, aggregator]]
        edges = [
            {"edge_id": f"e{i}", "source": src, "target": aggregator}
            for i, src in enumerate([health, finance, personal, creds, comms])
        ]
        tool_regs = {
            health: ["search_patient_records"],
            finance: ["check_balance"],
            personal: ["lookup_user_profile"],
            creds: ["fetch_api_key"],
            comms: ["send_email_notification"],
            aggregator: ["generate_report"],
        }

        result = compute_privacy_topology(nodes, edges, tool_regs, [])

        # Aggregator should be in fan_in_analysis with high risk
        agg_fan_in = [fi for fi in result["fan_in_analysis"]
                      if fi["node_id"] == aggregator]
        assert len(agg_fan_in) == 1
        assert agg_fan_in[0]["fan_in_risk"] == "high"
        assert agg_fan_in[0]["domain_count"] >= 5
        assert agg_fan_in[0]["heterogeneous"] is True

        assert result["privacy_exposure_score"] > 0.5
        assert result["finding_triggered"] is True
