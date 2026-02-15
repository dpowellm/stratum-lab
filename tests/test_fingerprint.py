"""Tests for graph fingerprinting."""
import pytest
from stratum_lab.query.fingerprint import (
    compute_graph_fingerprint,
    normalize_feature_vector,
    compute_normalization_constants,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_structural_graph() -> dict:
    """A representative structural graph with 10 nodes and 10 edges.

    Nodes: agent_researcher, agent_writer, agent_reviewer, agent_manager,
           cap_web_search, cap_file_writer, cap_llm_call,
           ds_shared_memory, ext_web_api, guard_quality.
    Edges: e1-e10 covering delegates_to, writes_to, calls, filtered_by,
           reads_from, tool_of, feeds_into, gated_by.
    """
    return {
        "repo_id": "test_repo_fp",
        "framework": "crewai",
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
    }


@pytest.fixture()
def minimal_graph() -> dict:
    """A minimal graph with 2 agents and 1 edge -- structurally distinct
    from the sample_structural_graph.
    """
    return {
        "repo_id": "test_repo_minimal",
        "framework": "langgraph",
        "nodes": {
            "agent_a": {
                "structural": {
                    "node_type": "agent",
                    "name": "AgentA",
                },
            },
            "agent_b": {
                "structural": {
                    "node_type": "agent",
                    "name": "AgentB",
                },
            },
        },
        "edges": {
            "e1": {
                "structural": {
                    "edge_type": "delegates_to",
                    "source": "agent_a",
                    "target": "agent_b",
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Tests: compute_graph_fingerprint
# ---------------------------------------------------------------------------

class TestComputeGraphFingerprint:
    """Tests for compute_graph_fingerprint."""

    def test_returns_feature_vector_of_length_20(self, sample_structural_graph):
        """The fingerprint must contain a feature_vector list of exactly 20 floats."""
        fp = compute_graph_fingerprint(sample_structural_graph)

        assert "feature_vector" in fp
        assert isinstance(fp["feature_vector"], list)
        assert len(fp["feature_vector"]) == 20
        for element in fp["feature_vector"]:
            assert isinstance(element, float)

    def test_returns_expected_top_level_keys(self, sample_structural_graph):
        """The fingerprint dict must have the documented top-level keys."""
        fp = compute_graph_fingerprint(sample_structural_graph)

        expected_keys = {
            "feature_vector",
            "motifs",
            "topology_hash",
            "node_type_distribution",
            "edge_type_distribution",
            "structural_metrics",
        }
        assert expected_keys == set(fp.keys())

    def test_deterministic_same_graph(self, sample_structural_graph):
        """Calling compute_graph_fingerprint twice on the same graph must
        produce identical feature vectors and topology hashes.
        """
        fp1 = compute_graph_fingerprint(sample_structural_graph)
        fp2 = compute_graph_fingerprint(sample_structural_graph)

        assert fp1["feature_vector"] == fp2["feature_vector"]
        assert fp1["topology_hash"] == fp2["topology_hash"]
        assert fp1["motifs"] == fp2["motifs"]

    def test_different_graphs_produce_different_fingerprints(
        self, sample_structural_graph, minimal_graph
    ):
        """Two structurally different graphs must produce different
        feature vectors and topology hashes.
        """
        fp_large = compute_graph_fingerprint(sample_structural_graph)
        fp_small = compute_graph_fingerprint(minimal_graph)

        assert fp_large["feature_vector"] != fp_small["feature_vector"]
        assert fp_large["topology_hash"] != fp_small["topology_hash"]

    def test_structural_metrics_counts_match_graph(self, sample_structural_graph):
        """Structural metrics should reflect the actual node/edge counts."""
        fp = compute_graph_fingerprint(sample_structural_graph)
        metrics = fp["structural_metrics"]

        assert metrics["agent_count"] == 4
        assert metrics["capability_count"] == 3
        assert metrics["data_store_count"] == 1
        assert metrics["external_service_count"] == 1
        assert metrics["guardrail_count"] == 1
        assert metrics["total_edges"] == 10

    def test_motifs_detected(self, sample_structural_graph):
        """The sample graph should trigger at least the
        shared_state_without_arbitration motif (2 agents writing to
        ds_shared_memory with no arbitrated_by edge).
        """
        fp = compute_graph_fingerprint(sample_structural_graph)

        assert isinstance(fp["motifs"], list)
        assert "shared_state_without_arbitration" in fp["motifs"]

    def test_topology_hash_is_hex_sha256(self, sample_structural_graph):
        """topology_hash must be a valid 64-character hex string (SHA-256)."""
        fp = compute_graph_fingerprint(sample_structural_graph)
        h = fp["topology_hash"]

        assert isinstance(h, str)
        assert len(h) == 64
        # Must be valid hexadecimal
        int(h, 16)


# ---------------------------------------------------------------------------
# Tests: normalize_feature_vector
# ---------------------------------------------------------------------------

class TestNormalizeFeatureVector:
    """Tests for normalize_feature_vector."""

    def test_normalized_values_in_0_1(self, sample_structural_graph):
        """Every element of the normalized vector must be in [0, 1]."""
        fp = compute_graph_fingerprint(sample_structural_graph)
        raw_vector = fp["feature_vector"]

        # Build normalization constants that bracket the raw values
        constants = {}
        for i, val in enumerate(raw_vector):
            constants[str(i)] = {"min": val - 1.0, "max": val + 1.0}

        normalized = normalize_feature_vector(raw_vector, constants)

        assert len(normalized) == 20
        for val in normalized:
            assert 0.0 <= val <= 1.0

    def test_zero_range_maps_to_zero(self):
        """When min == max for a feature, the normalized value should be 0.0."""
        raw = [5.0] * 20
        constants = {str(i): {"min": 5.0, "max": 5.0} for i in range(20)}

        normalized = normalize_feature_vector(raw, constants)

        for val in normalized:
            assert val == 0.0

    def test_exact_min_maps_to_zero(self):
        """A value equal to the minimum should normalize to 0.0."""
        raw = [0.0] * 20
        constants = {str(i): {"min": 0.0, "max": 10.0} for i in range(20)}

        normalized = normalize_feature_vector(raw, constants)

        for val in normalized:
            assert val == pytest.approx(0.0)

    def test_exact_max_maps_to_one(self):
        """A value equal to the maximum should normalize to 1.0."""
        raw = [10.0] * 20
        constants = {str(i): {"min": 0.0, "max": 10.0} for i in range(20)}

        normalized = normalize_feature_vector(raw, constants)

        for val in normalized:
            assert val == pytest.approx(1.0)

    def test_values_clamped_above_max(self):
        """Values above max should be clamped to 1.0."""
        raw = [20.0] * 20
        constants = {str(i): {"min": 0.0, "max": 10.0} for i in range(20)}

        normalized = normalize_feature_vector(raw, constants)

        for val in normalized:
            assert val == 1.0

    def test_values_clamped_below_min(self):
        """Values below min should be clamped to 0.0."""
        raw = [-5.0] * 20
        constants = {str(i): {"min": 0.0, "max": 10.0} for i in range(20)}

        normalized = normalize_feature_vector(raw, constants)

        for val in normalized:
            assert val == 0.0


# ---------------------------------------------------------------------------
# Tests: compute_normalization_constants
# ---------------------------------------------------------------------------

class TestComputeNormalizationConstants:
    """Tests for compute_normalization_constants."""

    def test_returns_min_max_for_each_feature(self, sample_structural_graph, minimal_graph):
        """The result should have an entry for each of the 20 feature indices,
        with 'min' and 'max' keys.
        """
        fp1 = compute_graph_fingerprint(sample_structural_graph)
        fp2 = compute_graph_fingerprint(minimal_graph)

        constants = compute_normalization_constants([fp1, fp2])

        assert len(constants) == 20
        for i in range(20):
            key = str(i)
            assert key in constants
            assert "min" in constants[key]
            assert "max" in constants[key]
            assert constants[key]["min"] <= constants[key]["max"]

    def test_single_fingerprint_min_equals_max(self, sample_structural_graph):
        """With only one fingerprint, min and max should be equal for every feature."""
        fp = compute_graph_fingerprint(sample_structural_graph)

        constants = compute_normalization_constants([fp])

        for i in range(20):
            key = str(i)
            assert constants[key]["min"] == constants[key]["max"]

    def test_empty_list_returns_empty_dict(self):
        """An empty list of fingerprints should return an empty dict."""
        constants = compute_normalization_constants([])
        assert constants == {}

    def test_min_max_reflect_actual_extremes(self):
        """The min/max should reflect the actual minimum and maximum values
        across the provided fingerprints.
        """
        fp_low = {"feature_vector": [0.0] * 20}
        fp_mid = {"feature_vector": [5.0] * 20}
        fp_high = {"feature_vector": [10.0] * 20}

        constants = compute_normalization_constants([fp_low, fp_mid, fp_high])

        for i in range(20):
            key = str(i)
            assert constants[key]["min"] == pytest.approx(0.0)
            assert constants[key]["max"] == pytest.approx(10.0)
