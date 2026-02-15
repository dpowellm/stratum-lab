"""Tests for the product query layer (Part 3)."""
import json
import pytest
import tempfile
from pathlib import Path

from stratum_lab.query.fingerprint import compute_graph_fingerprint
from stratum_lab.query.matcher import match_against_dataset, Match
from stratum_lab.query.predictor import predict_risks, RiskPrediction
from stratum_lab.query.report import generate_risk_report


# ---------------------------------------------------------------------------
# Synthetic knowledge-base fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_kb(tmp_path) -> Path:
    """Create a temporary knowledge-base directory populated with the JSON
    files that the query layer loads.

    Contents:
      - patterns.json: 2 patterns (shared_state_without_arbitration, linear_delegation_chain)
      - taxonomy_probabilities.json: probability data for shared_state_no_arbitration
      - fingerprints.json: 3 sample fingerprints from repos
      - normalization.json: min/max for each of 13 feature dimensions
      - fragility_map.json: fragility entries
      - framework_comparisons.json: comparison data
    """
    kb = tmp_path / "kb"
    kb.mkdir()

    # -- patterns.json -------------------------------------------------------
    patterns = [
        {
            "pattern_id": "pat_shared_state_abc123",
            "pattern_name": "shared_state_without_arbitration",
            "structural_signature": {
                "data_store": "ds_shared",
                "writers": ["agent_a", "agent_b"],
                "writer_count": 2,
                "has_arbitration": False,
            },
            "prevalence": {
                "repos_count": 5,
                "total_repos": 20,
                "prevalence_rate": 0.25,
                "repo_ids": ["repo_001", "repo_003", "repo_006", "repo_009", "repo_012"],
            },
            "behavioral_distribution": {
                "failure_rate": 0.6,
                "confidence_interval_95": [0.35, 0.82],
                "sample_size": 5,
                "repos_with_failure": 3,
                "failure_modes": {"propagate": 2, "retry": 1},
                "avg_error_rate": 0.15,
                "avg_latency_p50_ms": 450.0,
            },
            "fragility_data": {
                "avg_tool_call_failure_rate": 0.08,
                "quality_dependent_rate": 0.3,
                "sample_count": 5,
            },
            "risk_assessment": {
                "risk_score": 0.45,
                "risk_level": "medium",
            },
        },
        {
            "pattern_id": "pat_linear_chain_def456",
            "pattern_name": "linear_delegation_chain",
            "structural_signature": {
                "chain": ["agent_manager", "agent_alpha", "agent_beta"],
                "chain_length": 3,
                "root": "agent_manager",
                "leaf": "agent_beta",
            },
            "prevalence": {
                "repos_count": 3,
                "total_repos": 20,
                "prevalence_rate": 0.15,
                "repo_ids": ["repo_002", "repo_005", "repo_010"],
            },
            "behavioral_distribution": {
                "failure_rate": 0.33,
                "confidence_interval_95": [0.10, 0.65],
                "sample_size": 3,
                "repos_with_failure": 1,
                "failure_modes": {"propagate": 1},
                "avg_error_rate": 0.05,
                "avg_latency_p50_ms": 320.0,
            },
            "fragility_data": {
                "avg_tool_call_failure_rate": 0.04,
                "quality_dependent_rate": 0.1,
                "sample_count": 3,
            },
            "risk_assessment": {
                "risk_score": 0.28,
                "risk_level": "low",
            },
        },
    ]
    (kb / "patterns.json").write_text(json.dumps(patterns), encoding="utf-8")

    # -- taxonomy_probabilities.json -----------------------------------------
    taxonomy_probs = {
        "shared_state_no_arbitration": {
            "probability": 0.67,
            "confidence_interval": [0.38, 0.88],
            "sample_size": 3,
            "manifested_count": 2,
            "severity_when_manifested": {
                "severity_label": "high",
                "description": "Data corruption observed",
            },
        },
        "unbounded_delegation_depth": {
            "probability": 0.40,
            "confidence_interval": [0.15, 0.70],
            "sample_size": 5,
            "manifested_count": 2,
            "severity_when_manifested": {
                "severity_label": "medium",
                "description": "Cascade timeout",
            },
        },
    }
    (kb / "taxonomy_probabilities.json").write_text(
        json.dumps(taxonomy_probs), encoding="utf-8"
    )

    # -- fingerprints.json ---------------------------------------------------
    # 3 sample fingerprints using the matcher's internal feature schema.
    # The matcher uses _extract_feature_vector which produces a 13-element
    # vector based on _FEATURE_KEYS.
    fingerprints = [
        {
            "repo_id": "repo_001",
            "nodes": {
                "a1": {"structural": {"node_type": "agent"}},
                "a2": {"structural": {"node_type": "agent"}},
                "ds": {"structural": {"node_type": "data_store"}},
                "ext": {"structural": {"node_type": "external"}},
            },
            "edges": {
                "e1": {"structural": {"edge_type": "writes_to", "source": "a1", "target": "ds"}},
                "e2": {"structural": {"edge_type": "writes_to", "source": "a2", "target": "ds"}},
                "e3": {"structural": {"edge_type": "calls", "source": "a1", "target": "ext"}},
            },
            "motifs": [
                {
                    "motif_name": "shared_state_without_arbitration",
                    "structural_signature": {"writer_count": 2},
                },
            ],
            "taxonomy_preconditions": ["shared_state_no_arbitration"],
        },
        {
            "repo_id": "repo_002",
            "nodes": {
                "mgr": {"structural": {"node_type": "agent"}},
                "w1": {"structural": {"node_type": "agent"}},
                "w2": {"structural": {"node_type": "agent"}},
                "w3": {"structural": {"node_type": "agent"}},
            },
            "edges": {
                "e1": {"structural": {"edge_type": "delegates_to", "source": "mgr", "target": "w1"}},
                "e2": {"structural": {"edge_type": "delegates_to", "source": "w1", "target": "w2"}},
                "e3": {"structural": {"edge_type": "delegates_to", "source": "w2", "target": "w3"}},
            },
            "motifs": [
                {
                    "motif_name": "linear_delegation_chain",
                    "structural_signature": {"chain_length": 4},
                },
            ],
            "taxonomy_preconditions": ["unbounded_delegation_depth"],
        },
        {
            "repo_id": "repo_003",
            "nodes": {
                "hub": {"structural": {"node_type": "agent"}},
                "s1": {"structural": {"node_type": "agent"}},
                "s2": {"structural": {"node_type": "agent"}},
                "s3": {"structural": {"node_type": "agent"}},
                "g": {"structural": {"node_type": "guardrail"}},
            },
            "edges": {
                "e1": {"structural": {"edge_type": "delegates_to", "source": "hub", "target": "s1"}},
                "e2": {"structural": {"edge_type": "delegates_to", "source": "hub", "target": "s2"}},
                "e3": {"structural": {"edge_type": "delegates_to", "source": "hub", "target": "s3"}},
                "e4": {"structural": {"edge_type": "filtered_by", "source": "hub", "target": "g"}},
            },
            "motifs": [
                {
                    "motif_name": "hub_and_spoke",
                    "structural_signature": {"spoke_count": 3},
                },
            ],
            "taxonomy_preconditions": ["single_point_of_failure"],
        },
    ]
    (kb / "fingerprints.json").write_text(
        json.dumps(fingerprints), encoding="utf-8"
    )

    # -- normalization.json --------------------------------------------------
    # The matcher normalizes a 13-element vector using lists of min/max.
    normalization = {
        "min": [0.0] * 13,
        "max": [10.0, 10.0, 5.0, 5.0, 20.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 3.0],
    }
    (kb / "normalization.json").write_text(
        json.dumps(normalization), encoding="utf-8"
    )

    # -- fragility_map.json --------------------------------------------------
    fragility_map = [
        {
            "structural_position": "hub_node",
            "sensitivity_score": 0.55,
            "avg_tool_call_failure_rate": 0.12,
            "max_tool_call_failure_rate": 0.25,
            "affected_repos_count": 4,
            "total_nodes_analyzed": 8,
            "quality_dependent_rate": 0.4,
            "avg_retry_activations": 1.2,
            "top_fragile_nodes": [],
        },
        {
            "structural_position": "chain_node",
            "sensitivity_score": 0.35,
            "avg_tool_call_failure_rate": 0.07,
            "max_tool_call_failure_rate": 0.15,
            "affected_repos_count": 3,
            "total_nodes_analyzed": 6,
            "quality_dependent_rate": 0.2,
            "avg_retry_activations": 0.8,
            "top_fragile_nodes": [],
        },
        {
            "structural_position": "connector_node",
            "sensitivity_score": 0.42,
            "avg_tool_call_failure_rate": 0.09,
            "max_tool_call_failure_rate": 0.18,
            "affected_repos_count": 2,
            "total_nodes_analyzed": 4,
            "quality_dependent_rate": 0.25,
            "avg_retry_activations": 0.5,
            "top_fragile_nodes": [],
        },
    ]
    (kb / "fragility_map.json").write_text(
        json.dumps(fragility_map), encoding="utf-8"
    )

    # -- framework_comparisons.json ------------------------------------------
    framework_comparisons = [
        {
            "motif_name": "shared_state_without_arbitration",
            "frameworks_compared": ["crewai", "langgraph"],
            "per_framework": {
                "crewai": {
                    "framework": "crewai",
                    "repos_count": 3,
                    "behavioral_distribution": {
                        "failure_rate": 0.67,
                        "confidence_interval_95": [0.30, 0.92],
                        "sample_size": 3,
                    },
                },
                "langgraph": {
                    "framework": "langgraph",
                    "repos_count": 2,
                    "behavioral_distribution": {
                        "failure_rate": 0.50,
                        "confidence_interval_95": [0.15, 0.85],
                        "sample_size": 2,
                    },
                },
            },
        },
    ]
    (kb / "framework_comparisons.json").write_text(
        json.dumps(framework_comparisons), encoding="utf-8"
    )

    return kb


# ---------------------------------------------------------------------------
# Synthetic structural graph for testing
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_structural_graph() -> dict:
    """A structural graph that triggers the shared_state_without_arbitration
    motif (2 agents writing to a data store without arbitration) and includes
    taxonomy preconditions suitable for risk prediction.
    """
    return {
        "repo_id": "test_repo_query",
        "framework": "crewai",
        "taxonomy_preconditions": [
            "shared_state_no_arbitration",
            "unbounded_delegation_depth",
        ],
        "nodes": {
            "agent_researcher": {
                "structural": {
                    "node_type": "agent",
                    "name": "Researcher",
                },
            },
            "agent_writer": {
                "structural": {
                    "node_type": "agent",
                    "name": "Writer",
                },
            },
            "agent_reviewer": {
                "structural": {
                    "node_type": "agent",
                    "name": "Reviewer",
                },
            },
            "agent_manager": {
                "structural": {
                    "node_type": "agent",
                    "name": "Manager",
                },
            },
            "cap_web_search": {
                "structural": {
                    "node_type": "capability",
                    "name": "WebSearch",
                },
            },
            "cap_file_writer": {
                "structural": {
                    "node_type": "capability",
                    "name": "FileWriter",
                },
            },
            "cap_llm_call": {
                "structural": {
                    "node_type": "capability",
                    "name": "LLMCall",
                },
            },
            "ds_shared_memory": {
                "structural": {
                    "node_type": "data_store",
                    "name": "SharedMemory",
                },
            },
            "ext_web_api": {
                "structural": {
                    "node_type": "external",
                    "name": "WebAPI",
                },
            },
            "guard_quality": {
                "structural": {
                    "node_type": "guardrail",
                    "name": "QualityGuard",
                },
            },
        },
        "edges": {
            "e1": {
                "structural": {
                    "edge_type": "delegates_to",
                    "source": "agent_manager",
                    "target": "agent_researcher",
                },
            },
            "e2": {
                "structural": {
                    "edge_type": "delegates_to",
                    "source": "agent_manager",
                    "target": "agent_writer",
                },
            },
            "e3": {
                "structural": {
                    "edge_type": "delegates_to",
                    "source": "agent_manager",
                    "target": "agent_reviewer",
                },
            },
            "e4": {
                "structural": {
                    "edge_type": "writes_to",
                    "source": "agent_researcher",
                    "target": "ds_shared_memory",
                },
            },
            "e5": {
                "structural": {
                    "edge_type": "writes_to",
                    "source": "agent_writer",
                    "target": "ds_shared_memory",
                },
            },
            "e6": {
                "structural": {
                    "edge_type": "calls",
                    "source": "cap_web_search",
                    "target": "ext_web_api",
                },
            },
            "e7": {
                "structural": {
                    "edge_type": "filtered_by",
                    "source": "agent_writer",
                    "target": "guard_quality",
                },
            },
            "e8": {
                "structural": {
                    "edge_type": "reads_from",
                    "source": "agent_reviewer",
                    "target": "ds_shared_memory",
                },
            },
            "e9": {
                "structural": {
                    "edge_type": "tool_of",
                    "source": "cap_file_writer",
                    "target": "agent_writer",
                },
            },
            "e10": {
                "structural": {
                    "edge_type": "feeds_into",
                    "source": "cap_llm_call",
                    "target": "agent_researcher",
                },
            },
        },
        "motifs": [
            {
                "motif_name": "shared_state_without_arbitration",
                "structural_signature": {
                    "data_store": "ds_shared_memory",
                    "writers": ["agent_researcher", "agent_writer"],
                    "writer_count": 2,
                    "has_arbitration": False,
                },
                "involved_nodes": ["agent_researcher", "agent_writer", "ds_shared_memory"],
                "involved_edges": ["e4", "e5"],
            },
        ],
    }


# ---------------------------------------------------------------------------
# Tests: match_against_dataset
# ---------------------------------------------------------------------------

class TestMatchAgainstDataset:
    """Tests for match_against_dataset."""

    def test_match_returns_motif_matches(self, sample_structural_graph, synthetic_kb):
        """A graph containing the shared_state_without_arbitration motif
        should produce at least one exact_motif match against the KB.
        """
        matches = match_against_dataset(sample_structural_graph, synthetic_kb)

        motif_matches = [m for m in matches if m.match_type == "exact_motif"]
        assert len(motif_matches) >= 1

        shared_state_match = next(
            (m for m in motif_matches
             if m.pattern_name == "shared_state_without_arbitration"),
            None,
        )
        assert shared_state_match is not None
        assert shared_state_match.similarity_score == 1.0
        assert shared_state_match.matched_repos == 5

    def test_match_returns_similarity_matches(self, sample_structural_graph, synthetic_kb):
        """Structural similarity matches should be present and ranked in
        descending order of similarity score.
        """
        matches = match_against_dataset(sample_structural_graph, synthetic_kb)

        sim_matches = [m for m in matches if m.match_type == "structural_similarity"]

        # We have 3 fingerprints in the KB, so we should get at least 1 match
        # (above the 0.1 threshold).
        assert len(sim_matches) >= 1

        # Verify descending order within the similarity matches
        scores = [m.similarity_score for m in sim_matches]
        assert scores == sorted(scores, reverse=True)

    def test_match_returns_list_of_match_objects(self, sample_structural_graph, synthetic_kb):
        """All returned items should be Match dataclass instances."""
        matches = match_against_dataset(sample_structural_graph, synthetic_kb)

        assert isinstance(matches, list)
        for m in matches:
            assert isinstance(m, Match)
            assert isinstance(m.pattern_id, str)
            assert isinstance(m.similarity_score, float)
            assert m.match_type in ("exact_motif", "structural_similarity", "archetype")

    def test_match_overall_sorted_by_score(self, sample_structural_graph, synthetic_kb):
        """The full match list should be sorted by similarity_score descending."""
        matches = match_against_dataset(sample_structural_graph, synthetic_kb)

        scores = [m.similarity_score for m in matches]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Tests: predict_risks
# ---------------------------------------------------------------------------

class TestPredictRisks:
    """Tests for predict_risks."""

    def test_predict_risks_non_empty(self, sample_structural_graph, synthetic_kb):
        """predict_risks should return at least 1 predicted risk when the
        graph has taxonomy preconditions that exist in the KB.
        """
        matches = match_against_dataset(sample_structural_graph, synthetic_kb)
        prediction = predict_risks(
            structural_graph=sample_structural_graph,
            matches=matches,
            taxonomy_preconditions=sample_structural_graph["taxonomy_preconditions"],
            knowledge_base_path=synthetic_kb,
        )

        assert isinstance(prediction, RiskPrediction)
        assert len(prediction.predicted_risks) >= 1

    def test_predict_overall_risk_score_in_range(self, sample_structural_graph, synthetic_kb):
        """The overall_risk_score must be in the range [0, 100]."""
        matches = match_against_dataset(sample_structural_graph, synthetic_kb)
        prediction = predict_risks(
            structural_graph=sample_structural_graph,
            matches=matches,
            taxonomy_preconditions=sample_structural_graph["taxonomy_preconditions"],
            knowledge_base_path=synthetic_kb,
        )

        assert 0.0 <= prediction.overall_risk_score <= 100.0

    def test_predict_returns_risk_prediction_dataclass(self, sample_structural_graph, synthetic_kb):
        """The return value must be a RiskPrediction dataclass with the
        expected attributes.
        """
        matches = match_against_dataset(sample_structural_graph, synthetic_kb)
        prediction = predict_risks(
            structural_graph=sample_structural_graph,
            matches=matches,
            taxonomy_preconditions=sample_structural_graph["taxonomy_preconditions"],
            knowledge_base_path=synthetic_kb,
        )

        assert hasattr(prediction, "archetype")
        assert hasattr(prediction, "predicted_risks")
        assert hasattr(prediction, "positive_signals")
        assert hasattr(prediction, "structural_only_risks")
        assert hasattr(prediction, "dataset_coverage")

    def test_predict_risks_have_remediation(self, sample_structural_graph, synthetic_kb):
        """Each predicted risk that has KB data should carry a non-empty
        remediation string for known preconditions.
        """
        matches = match_against_dataset(sample_structural_graph, synthetic_kb)
        prediction = predict_risks(
            structural_graph=sample_structural_graph,
            matches=matches,
            taxonomy_preconditions=sample_structural_graph["taxonomy_preconditions"],
            knowledge_base_path=synthetic_kb,
        )

        for risk in prediction.predicted_risks:
            # shared_state_no_arbitration and unbounded_delegation_depth
            # are both in the REMEDIATIONS dict
            assert isinstance(risk.remediation, str)
            assert len(risk.remediation) > 0

    def test_predict_positive_signals_for_guardrail(self, sample_structural_graph, synthetic_kb):
        """The sample graph has a guardrail node, so positive signals should
        include the guardrail-related signal.
        """
        matches = match_against_dataset(sample_structural_graph, synthetic_kb)
        prediction = predict_risks(
            structural_graph=sample_structural_graph,
            matches=matches,
            taxonomy_preconditions=sample_structural_graph["taxonomy_preconditions"],
            knowledge_base_path=synthetic_kb,
        )

        guardrail_signals = [
            s for s in prediction.positive_signals if "guardrail" in s.lower()
        ]
        assert len(guardrail_signals) >= 1

    def test_predict_dataset_coverage(self, sample_structural_graph, synthetic_kb):
        """dataset_coverage should report on preconditions queried vs. found."""
        matches = match_against_dataset(sample_structural_graph, synthetic_kb)
        prediction = predict_risks(
            structural_graph=sample_structural_graph,
            matches=matches,
            taxonomy_preconditions=sample_structural_graph["taxonomy_preconditions"],
            knowledge_base_path=synthetic_kb,
        )

        cov = prediction.dataset_coverage
        assert "preconditions_queried" in cov
        assert cov["preconditions_queried"] == 2
        assert "preconditions_with_data" in cov
        assert cov["preconditions_with_data"] >= 1


# ---------------------------------------------------------------------------
# Tests: generate_risk_report
# ---------------------------------------------------------------------------

class TestGenerateRiskReport:
    """Tests for generate_risk_report."""

    def _get_prediction(self, sample_structural_graph, synthetic_kb):
        """Helper to compute matches and prediction for report tests."""
        matches = match_against_dataset(sample_structural_graph, synthetic_kb)
        return predict_risks(
            structural_graph=sample_structural_graph,
            matches=matches,
            taxonomy_preconditions=sample_structural_graph["taxonomy_preconditions"],
            knowledge_base_path=synthetic_kb,
        )

    def test_report_json_valid(self, sample_structural_graph, synthetic_kb):
        """generate_risk_report with format='json' should return a dict
        with the expected top-level keys.
        """
        prediction = self._get_prediction(sample_structural_graph, synthetic_kb)
        report = generate_risk_report(
            prediction, sample_structural_graph, output_format="json"
        )

        assert isinstance(report, dict)
        expected_keys = {
            "executive_summary",
            "risk_details",
            "structural_only_risks",
            "architecture_analysis",
            "benchmark_comparison",
            "dataset_coverage",
            "methodology",
        }
        assert expected_keys == set(report.keys())

        # executive_summary should have an overall_risk_score
        es = report["executive_summary"]
        assert "overall_risk_score" in es
        assert "archetype" in es
        assert "risk_level" in es

        # risk_details should be a list
        assert isinstance(report["risk_details"], list)

        # The JSON should be serializable (no dataclasses or non-serializable objects)
        json.dumps(report)

    def test_report_markdown_valid(self, sample_structural_graph, synthetic_kb):
        """generate_risk_report with format='markdown' should return a
        string that starts with '# Stratum Risk Report'.
        """
        prediction = self._get_prediction(sample_structural_graph, synthetic_kb)
        report = generate_risk_report(
            prediction, sample_structural_graph, output_format="markdown"
        )

        assert isinstance(report, str)
        assert report.startswith("# Stratum Risk Report")
        assert "## Executive Summary" in report
        assert "## Risk Details" in report
        assert "## Architecture Analysis" in report
        assert "## Methodology" in report

    def test_report_markdown_contains_risk_info(self, sample_structural_graph, synthetic_kb):
        """The markdown report should contain information about the predicted
        risks (precondition IDs, severity, remediation).
        """
        prediction = self._get_prediction(sample_structural_graph, synthetic_kb)
        report = generate_risk_report(
            prediction, sample_structural_graph, output_format="markdown"
        )

        # At least one of the preconditions should appear
        assert "shared_state_no_arbitration" in report or "unbounded_delegation_depth" in report

    def test_report_json_risk_details_match_prediction(
        self, sample_structural_graph, synthetic_kb
    ):
        """The number of risk_details entries in the JSON report should equal
        the number of predicted_risks in the RiskPrediction.
        """
        prediction = self._get_prediction(sample_structural_graph, synthetic_kb)
        report = generate_risk_report(
            prediction, sample_structural_graph, output_format="json"
        )

        assert len(report["risk_details"]) == len(prediction.predicted_risks)


# ---------------------------------------------------------------------------
# End-to-end test
# ---------------------------------------------------------------------------

class TestEndToEndQueryFlow:
    """End-to-end: synthetic structural graph -> fingerprint -> match ->
    predict -> report.
    """

    def test_end_to_end_query_flow(self, sample_structural_graph, synthetic_kb):
        """Run the full query pipeline and verify that each stage produces
        valid, non-empty output that feeds into the next stage.
        """
        # Stage 1: Compute fingerprint
        fingerprint = compute_graph_fingerprint(sample_structural_graph)
        assert "feature_vector" in fingerprint
        assert len(fingerprint["feature_vector"]) == 20
        assert len(fingerprint["motifs"]) >= 1

        # Stage 2: Match against dataset
        matches = match_against_dataset(sample_structural_graph, synthetic_kb)
        assert isinstance(matches, list)
        assert len(matches) >= 1
        # At least one exact motif match should be present
        assert any(m.match_type == "exact_motif" for m in matches)

        # Stage 3: Predict risks
        prediction = predict_risks(
            structural_graph=sample_structural_graph,
            matches=matches,
            taxonomy_preconditions=sample_structural_graph["taxonomy_preconditions"],
            knowledge_base_path=synthetic_kb,
        )
        assert isinstance(prediction, RiskPrediction)
        assert 0.0 <= prediction.overall_risk_score <= 100.0
        assert len(prediction.predicted_risks) >= 1

        # Stage 4a: Generate JSON report
        json_report = generate_risk_report(
            prediction, sample_structural_graph, output_format="json"
        )
        assert isinstance(json_report, dict)
        assert "executive_summary" in json_report
        # Verify the JSON is fully serializable
        serialized = json.dumps(json_report)
        assert len(serialized) > 100  # non-trivial content

        # Stage 4b: Generate markdown report
        md_report = generate_risk_report(
            prediction, sample_structural_graph, output_format="markdown"
        )
        assert isinstance(md_report, str)
        assert "# Stratum Risk Report" in md_report
        assert len(md_report) > 200  # non-trivial content

        # Cross-check: JSON report risk_details count matches prediction
        assert len(json_report["risk_details"]) == len(prediction.predicted_risks)

        # Cross-check: executive summary risk score matches prediction
        assert (
            json_report["executive_summary"]["overall_risk_score"]
            == prediction.overall_risk_score
        )
