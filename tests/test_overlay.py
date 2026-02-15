"""Tests validating overlay bug fixes from Part 1 and Part 2 features."""
import pytest
from stratum_lab.overlay.enricher import enrich_graph, compute_node_behavioral_overlay
from stratum_lab.overlay.edges import detect_emergent_edges, detect_dead_edges
from stratum_lab.node_ids import normalize_name


class TestCapabilityNodesGetBehavioralData:
    """Bug fix 1A: capability nodes should have non-zero behavioral data."""

    def test_capability_node_activation_count(self, sample_structural_graph, sample_run_records):
        enriched = enrich_graph(sample_structural_graph, sample_run_records)
        cap_ws = enriched["nodes"]["cap_web_search"]["behavioral"]
        assert cap_ws["activation_count"] > 0, "WebSearch capability should be activated"

    def test_capability_node_llm(self, sample_structural_graph, sample_run_records):
        enriched = enrich_graph(sample_structural_graph, sample_run_records)
        cap_llm = enriched["nodes"]["cap_llm_call"]["behavioral"]
        assert cap_llm["activation_count"] > 0, "LLM capability should be activated"


class TestDataStoreNodesGetBehavioralData:
    """Bug fix 1A: data store nodes should have non-zero data."""

    def test_data_store_activation(self, sample_structural_graph, sample_run_records):
        enriched = enrich_graph(sample_structural_graph, sample_run_records)
        ds = enriched["nodes"]["ds_shared_memory"]["behavioral"]
        assert ds["activation_count"] > 0, "SharedMemory data store should be activated"


class TestExternalServiceNodesGetData:
    def test_external_node_activation(self, sample_structural_graph, sample_run_records):
        enriched = enrich_graph(sample_structural_graph, sample_run_records)
        ext = enriched["nodes"]["ext_web_api"]["behavioral"]
        assert ext["activation_count"] > 0, "WebAPI external node should be activated"


class TestGuardrailNodesGetData:
    def test_guardrail_activation(self, sample_structural_graph, sample_run_records):
        enriched = enrich_graph(sample_structural_graph, sample_run_records)
        guard = enriched["nodes"]["guard_quality"]["behavioral"]
        assert guard["activation_count"] > 0, "QualityGuard should be activated"

    def test_guardrail_effectiveness(self, sample_structural_graph, sample_run_records):
        enriched = enrich_graph(sample_structural_graph, sample_run_records)
        guard = enriched["nodes"]["guard_quality"]["behavioral"]
        ge = guard.get("guardrail_effectiveness")
        assert ge is not None, "guardrail_effectiveness should be populated"
        assert ge["trigger_count"] == 3


class TestActivationRateIs0To1:
    """Bug fix 1C: activation_rate should be 0.0-1.0."""

    def test_node_activation_rate_range(self, sample_structural_graph, sample_run_records):
        enriched = enrich_graph(sample_structural_graph, sample_run_records)
        for node_id, node_data in enriched["nodes"].items():
            rate = node_data["behavioral"]["activation_rate"]
            assert 0.0 <= rate <= 1.0, f"Node {node_id} activation_rate={rate} not in [0,1]"

    def test_edge_activation_rate_range(self, sample_structural_graph, sample_run_records):
        enriched = enrich_graph(sample_structural_graph, sample_run_records)
        for edge_id, edge_data in enriched["edges"].items():
            rate = edge_data["behavioral"]["activation_rate"]
            assert 0.0 <= rate <= 1.0, f"Edge {edge_id} activation_rate={rate} not in [0,1]"


class TestDecisionBehaviorPopulated:
    """Bug fix 1D: decision_behavior should be populated for nodes with decision.made events."""

    def test_decision_behavior_not_none(self, sample_structural_graph, sample_run_records):
        enriched = enrich_graph(sample_structural_graph, sample_run_records)
        manager = enriched["nodes"]["agent_manager"]["behavioral"]
        assert manager["decision_behavior"] is not None, "Manager should have decision_behavior"
        assert manager["decision_behavior"]["decisions_made"] == 3
        assert manager["decision_behavior"]["decision_entropy"] > 0


class TestStructuralPredictionMatch:
    """Bug fix 1E: structural_prediction_match should be populated."""

    def test_prediction_match_on_error_node(self, sample_structural_graph, sample_run_records):
        enriched = enrich_graph(sample_structural_graph, sample_run_records)
        researcher = enriched["nodes"]["agent_researcher"]["behavioral"]
        # Researcher has error_handling: {strategy: "fail_silent"} and error events with handling: "fail_silent"
        match = researcher["error_behavior"]["structural_prediction_match"]
        assert match is True, f"Prediction should match, got {match}"


class TestErrorPropagationThroughEdges:
    """Bug fix 1F: error propagation should update edge error_crossings."""

    def test_at_least_one_edge_has_error_crossings(self, sample_structural_graph, sample_run_records):
        enriched = enrich_graph(sample_structural_graph, sample_run_records)
        has_errors = False
        for edge_id, edge_data in enriched["edges"].items():
            if edge_data["behavioral"]["error_crossings"]["errors_traversed"] > 0:
                has_errors = True
                break
        assert has_errors, "At least one edge should have non-zero error_crossings"


class TestEdgeTraversalCounts:
    """Bug fix 1B: edges should have non-zero traversal when events reference them."""

    def test_delegation_edges_traversed(self, sample_structural_graph, sample_run_records):
        enriched = enrich_graph(sample_structural_graph, sample_run_records)
        e1 = enriched["edges"]["e1"]["behavioral"]
        assert e1["traversal_count"] > 0, "Delegation edge e1 should be traversed"

    def test_uses_edges_traversed(self, sample_structural_graph, sample_run_records):
        enriched = enrich_graph(sample_structural_graph, sample_run_records)
        e4 = enriched["edges"]["e4"]["behavioral"]
        assert e4["traversal_count"] > 0, "Uses edge e4 should be traversed"


class TestUnmappedEventsZero:
    """All event types exercised should result in 0 unmapped events."""

    def test_unmapped_node_events_zero(self, sample_structural_graph, sample_run_records):
        enriched = enrich_graph(sample_structural_graph, sample_run_records)
        assert enriched["unmapped_events"]["node_events"] == 0, \
            f"Expected 0 unmapped node events, got {enriched['unmapped_events']['node_events']}"


# ============================================================================
# Part 2A: Cross-run determinism metrics
# ============================================================================

class TestDeterminismMetrics:
    """Part 2A: determinism metrics should be present on enriched nodes."""

    def test_determinism_present_on_active_nodes(self, sample_structural_graph, sample_run_records):
        enriched = enrich_graph(sample_structural_graph, sample_run_records)
        researcher = enriched["nodes"]["agent_researcher"]["behavioral"]
        det = researcher.get("determinism")
        assert det is not None, "Active node should have determinism metrics"
        assert "same_input_activation_consistency" in det
        assert "same_input_path_consistency" in det
        assert "cross_input_variance" in det

    def test_same_input_consistency_high_for_identical_runs(self, sample_structural_graph, sample_run_records):
        """Runs 000 and 001 use same input_hash and have identical events, so consistency should be 1.0."""
        enriched = enrich_graph(sample_structural_graph, sample_run_records)
        researcher = enriched["nodes"]["agent_researcher"]["behavioral"]
        det = researcher["determinism"]
        assert det["same_input_activation_consistency"] == 1.0

    def test_same_input_path_consistency_high(self, sample_structural_graph, sample_run_records):
        """Same-input runs have identical activation counts, so path consistency should be 1.0."""
        enriched = enrich_graph(sample_structural_graph, sample_run_records)
        researcher = enriched["nodes"]["agent_researcher"]["behavioral"]
        det = researcher["determinism"]
        assert det["same_input_path_consistency"] == 1.0

    def test_cross_input_variance_low_for_uniform_behavior(self, sample_structural_graph, sample_run_records):
        """All runs activate the same nodes equally, so cross-input variance should be low."""
        enriched = enrich_graph(sample_structural_graph, sample_run_records)
        researcher = enriched["nodes"]["agent_researcher"]["behavioral"]
        det = researcher["determinism"]
        assert det["cross_input_variance"] < 0.5

    def test_determinism_none_for_single_run(self, sample_structural_graph):
        """With only 1 run, determinism should be None."""
        single_run = [{
            "run_id": "run_000",
            "repo_id": "test_repo",
            "framework": "crewai",
            "events": [{
                "event_id": "e1", "timestamp_ns": 1000, "run_id": "run_000",
                "repo_id": "test_repo", "event_type": "agent.task_start",
                "source_node": {"node_type": "agent", "node_id": "agent_researcher", "node_name": "Researcher"},
            }],
            "metadata": {"input_hash": "input_001"},
        }]
        enriched = enrich_graph(sample_structural_graph, single_run)
        det = enriched["nodes"]["agent_researcher"]["behavioral"]["determinism"]
        assert det is None


# ============================================================================
# Part 2B: Execution path signatures
# ============================================================================

class TestExecutionPaths:
    """Part 2B: execution paths and path analysis should be present."""

    def test_execution_paths_present(self, sample_structural_graph, sample_run_records):
        enriched = enrich_graph(sample_structural_graph, sample_run_records)
        assert "execution_paths" in enriched
        assert len(enriched["execution_paths"]) == 3  # 3 runs

    def test_execution_path_has_run_id(self, sample_structural_graph, sample_run_records):
        enriched = enrich_graph(sample_structural_graph, sample_run_records)
        for p in enriched["execution_paths"]:
            assert "run_id" in p
            assert "execution_path" in p

    def test_execution_path_steps_have_edge_fields(self, sample_structural_graph, sample_run_records):
        enriched = enrich_graph(sample_structural_graph, sample_run_records)
        for p in enriched["execution_paths"]:
            for step in p["execution_path"]:
                assert "edge_id" in step
                assert "timestamp_ns" in step
                assert "source" in step
                assert "target" in step
                assert "edge_type" in step

    def test_execution_paths_non_empty_for_edge_events(self, sample_structural_graph, sample_run_records):
        """Runs with delegation/tool/data events should produce non-empty execution paths."""
        enriched = enrich_graph(sample_structural_graph, sample_run_records)
        for p in enriched["execution_paths"]:
            assert len(p["execution_path"]) > 0, f"Run {p['run_id']} should have edge traversals"


class TestPathAnalysis:
    """Part 2B: path analysis aggregate should be computed."""

    def test_path_analysis_present(self, sample_structural_graph, sample_run_records):
        enriched = enrich_graph(sample_structural_graph, sample_run_records)
        pa = enriched.get("path_analysis")
        assert pa is not None
        assert "distinct_paths" in pa
        assert "dominant_path_frequency" in pa
        assert "path_divergence_points" in pa
        assert "conditional_edge_activation_rates" in pa

    def test_distinct_paths_at_least_one(self, sample_structural_graph, sample_run_records):
        enriched = enrich_graph(sample_structural_graph, sample_run_records)
        pa = enriched["path_analysis"]
        assert pa["distinct_paths"] >= 1

    def test_dominant_path_frequency_in_range(self, sample_structural_graph, sample_run_records):
        enriched = enrich_graph(sample_structural_graph, sample_run_records)
        pa = enriched["path_analysis"]
        assert 0.0 <= pa["dominant_path_frequency"] <= 1.0


# ============================================================================
# Part 2C: Tool failure impact chains
# ============================================================================

class TestFailureImpact:
    """Part 2C: failure_impact should appear on capability nodes with failures."""

    def test_no_failure_impact_when_no_failures(self, sample_structural_graph, sample_run_records):
        """Sample events have no tool failures, so failure_impact should not be present."""
        enriched = enrich_graph(sample_structural_graph, sample_run_records)
        cap = enriched["nodes"]["cap_web_search"]["behavioral"]
        ms = cap.get("model_sensitivity", {})
        assert "failure_impact" not in ms

    def test_failure_impact_present_when_failures_exist(self, sample_structural_graph):
        """Build events with tool failures and verify failure_impact is populated."""
        events = []
        base_ts = 1000000000000

        # Agent starts
        events.append({
            "event_id": "e_agent_start", "timestamp_ns": base_ts, "run_id": "run_fail",
            "repo_id": "test_repo", "event_type": "agent.task_start",
            "source_node": {"node_type": "agent", "node_id": "agent_researcher", "node_name": "Researcher"},
        })
        # Tool invocation
        events.append({
            "event_id": "e_tool_inv", "timestamp_ns": base_ts + 1000, "run_id": "run_fail",
            "repo_id": "test_repo", "event_type": "tool.invoked",
            "source_node": {"node_type": "capability", "node_id": "cap_web_search", "node_name": "WebSearch"},
            "payload": {"agent_id": "Researcher", "tool_name": "WebSearch"},
        })
        # Tool failure
        events.append({
            "event_id": "e_tool_fail", "timestamp_ns": base_ts + 2000, "run_id": "run_fail",
            "repo_id": "test_repo", "event_type": "tool.completed",
            "source_node": {"node_type": "capability", "node_id": "cap_web_search", "node_name": "WebSearch"},
            "parent_event_id": "e_tool_inv",
            "payload": {"agent_id": "Researcher", "tool_name": "WebSearch", "status": "failure"},
        })
        # Downstream agent proceeds without noticing (silent degradation)
        events.append({
            "event_id": "e_writer_start", "timestamp_ns": base_ts + 5000, "run_id": "run_fail",
            "repo_id": "test_repo", "event_type": "agent.task_start",
            "source_node": {"node_type": "agent", "node_id": "agent_writer", "node_name": "Writer"},
        })
        events.append({
            "event_id": "e_writer_end", "timestamp_ns": base_ts + 8000, "run_id": "run_fail",
            "repo_id": "test_repo", "event_type": "agent.task_end",
            "source_node": {"node_type": "agent", "node_id": "agent_writer", "node_name": "Writer"},
            "parent_event_id": "e_writer_start",
        })

        run_records = [{
            "run_id": "run_fail",
            "repo_id": "test_repo",
            "framework": "crewai",
            "events": events,
            "metadata": {"input_hash": "input_fail"},
        }]

        enriched = enrich_graph(sample_structural_graph, run_records)
        cap = enriched["nodes"]["cap_web_search"]["behavioral"]
        ms = cap.get("model_sensitivity", {})
        fi = ms.get("failure_impact")
        assert fi is not None, "failure_impact should be present after tool failure"
        assert "downstream_degradation_count" in fi
        assert "cascade_to_other_agents" in fi
        assert "recovery_observed" in fi
        assert "avg_recovery_time_ms" in fi
        assert "silent_degradation_count" in fi
        assert fi["silent_degradation_count"] >= 1, "Writer proceeded after failure = silent degradation"
