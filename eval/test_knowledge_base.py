"""Evaluation script for the knowledge base (patterns, taxonomy, fragility).

Creates 20 synthetic enriched graphs spanning 3 frameworks (crewai, langgraph,
autogen), builds the pattern knowledge base, detects novel patterns, compares
frameworks, and builds the fragility map.

Prints:
  (a) Manifestation probabilities for present preconditions
  (b) Novel patterns detected
  (c) Framework comparison
  (d) Fragility map entries
"""

from __future__ import annotations

import json
import random
import sys
import os

# ---------------------------------------------------------------------------
# Path setup -- stratum_lab is NOT an installed package
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stratum_lab.knowledge.patterns import (
    build_pattern_knowledge_base,
    detect_novel_patterns,
    compare_frameworks,
)
from stratum_lab.knowledge.taxonomy import (
    TAXONOMY_PRECONDITIONS,
    compute_manifestation_probabilities,
)
from stratum_lab.knowledge.fragility import build_fragility_map

random.seed(42)

# =========================================================================
# 1.  Helpers to synthesise enriched graph dicts
# =========================================================================

# We draw taxonomy preconditions from the canonical list, plus semantic ones
PRECONDITION_POOL = TAXONOMY_PRECONDITIONS[:15] + [
    "unvalidated_semantic_chain",
    "classification_without_validation",
]


def _rand_latency() -> dict:
    p50 = round(random.uniform(50, 2000), 2)
    return {
        "p50": p50,
        "p95": round(p50 * random.uniform(1.3, 3.0), 2),
        "p99": round(p50 * random.uniform(2.0, 5.0), 2),
        "variance": round(random.uniform(10, 500), 2),
        "sample_count": random.randint(3, 30),
    }


def _rand_error_behavior(has_errors: bool) -> dict:
    if not has_errors:
        return {
            "errors_occurred": 0,
            "propagated_downstream": 0,
            "swallowed": 0,
            "propagation_rate": 0.0,
            "default_values_used": 0,
            "observed_error_handling": [],
            "structural_prediction_match": None,
        }
    errors = random.randint(1, 8)
    propagated = random.randint(0, errors)
    handling = random.sample(["retry", "propagate", "fallback", "ignore"], k=random.randint(1, 2))
    return {
        "errors_occurred": errors,
        "propagated_downstream": propagated,
        "swallowed": errors - propagated,
        "propagation_rate": round(propagated / max(errors, 1), 4),
        "default_values_used": random.randint(0, 2),
        "observed_error_handling": handling,
        "structural_prediction_match": random.choice([True, False, None]),
    }


def _rand_model_sensitivity(fragile: bool) -> dict:
    if fragile:
        failures = random.randint(2, 10)
        total = random.randint(failures, failures + 15)
        retries = random.randint(1, 5)
    else:
        total = random.randint(5, 20)
        failures = random.randint(0, 1)
        retries = 0
    return {
        "tool_call_failures": failures,
        "tool_call_failure_rate": round(failures / max(total, 1), 4),
        "retry_activations": retries,
        "quality_dependent": fragile,
    }


def _rand_node_behavioral(has_errors: bool = False, fragile: bool = False) -> dict:
    activation_count = random.randint(3, 30)
    total_runs = 3
    tasks = activation_count
    completed = tasks - random.randint(0, 2)
    failed = random.randint(0, 3) if has_errors else 0
    return {
        "activation_count": activation_count,
        "activation_rate": round(activation_count / total_runs, 4),
        "throughput": {
            "tasks_received": tasks,
            "completed": max(completed, 0),
            "failed": failed,
            "failure_rate": round(failed / max(tasks, 1), 4),
        },
        "latency": _rand_latency(),
        "error_behavior": _rand_error_behavior(has_errors),
        "decision_behavior": None,
        "model_sensitivity": _rand_model_sensitivity(fragile),
        "resource_usage": {
            "avg_tokens": round(random.uniform(200, 3000), 2),
            "avg_tool_calls": round(random.uniform(0, 5), 2),
            "avg_llm_calls": round(random.uniform(1, 8), 2),
            "avg_iterations": round(random.uniform(1, 4), 2),
        },
    }


def _make_enriched_graph(
    repo_index: int,
    framework: str,
    has_shared_state: bool = False,
    has_hub_spoke: bool = False,
    has_failures: bool = False,
    fragile_nodes: int = 0,
) -> dict:
    """Build one enriched graph dict with controllable motifs."""
    repo_id = f"repo_{repo_index:03d}"
    total_runs = 3

    # Pick some taxonomy preconditions for this repo
    n_preconditions = random.randint(2, 6)
    preconditions = random.sample(PRECONDITION_POOL, k=n_preconditions)

    # ---- Build nodes ----
    nodes: dict = {}
    edges: dict = {}
    edge_counter = 0

    # Always have a "Manager" agent
    manager_id = "agent_manager"
    nodes[manager_id] = {
        "structural": {
            "node_type": "agent",
            "name": "Manager",
            "taxonomy_preconditions": preconditions[:2],
        },
        "behavioral": _rand_node_behavioral(has_errors=has_failures, fragile=False),
    }

    # Worker agents
    worker_names = ["Alpha", "Beta", "Gamma", "Delta"]
    worker_ids = [f"agent_{n.lower()}" for n in worker_names]

    for idx, (wid, wname) in enumerate(zip(worker_ids, worker_names)):
        is_fragile = idx < fragile_nodes
        nodes[wid] = {
            "structural": {
                "node_type": "agent",
                "name": wname,
                "taxonomy_preconditions": preconditions[2:4] if idx == 0 else [],
            },
            "behavioral": _rand_node_behavioral(
                has_errors=(has_failures and idx == 0),
                fragile=is_fragile,
            ),
        }

    # Data store (for shared_state motif)
    ds_id = "ds_shared_state"
    nodes[ds_id] = {
        "structural": {"node_type": "data_store", "name": "SharedState", "taxonomy_preconditions": []},
        "behavioral": _rand_node_behavioral(),
    }

    # External service
    ext_id = "ext_third_party"
    nodes[ext_id] = {
        "structural": {"node_type": "external", "name": "ThirdPartyAPI", "taxonomy_preconditions": []},
        "behavioral": _rand_node_behavioral(),
    }

    # ---- Build edges ----
    # Manager delegates to workers
    if has_hub_spoke:
        # Manager delegates to all 4 workers -> hub_and_spoke (>= 3 spokes)
        for wid in worker_ids:
            edge_counter += 1
            edges[f"e{edge_counter}"] = {
                "structural": {"edge_type": "delegates_to", "source": manager_id, "target": wid},
                "behavioral": {
                    "traversal_count": random.randint(1, 5),
                    "activation_rate": round(random.random(), 4),
                    "never_activated": False,
                    "data_flow": {"avg_data_size_bytes": 512.0, "schema_mismatch_count": 0, "null_data_count": 0},
                    "error_crossings": {"errors_traversed": 0, "error_types": [], "downstream_impact": 0},
                    "latency_contribution_ms": round(random.uniform(5, 50), 2),
                    "conditional_behavior": None,
                },
            }
    else:
        # Linear chain: Manager -> Alpha -> Beta
        for i in range(min(2, len(worker_ids))):
            edge_counter += 1
            src = manager_id if i == 0 else worker_ids[i - 1]
            tgt = worker_ids[i]
            edges[f"e{edge_counter}"] = {
                "structural": {"edge_type": "delegates_to", "source": src, "target": tgt},
                "behavioral": {
                    "traversal_count": random.randint(1, 4),
                    "activation_rate": round(random.random(), 4),
                    "never_activated": False,
                    "data_flow": {"avg_data_size_bytes": 256.0, "schema_mismatch_count": 0, "null_data_count": 0},
                    "error_crossings": {"errors_traversed": 0, "error_types": [], "downstream_impact": 0},
                    "latency_contribution_ms": round(random.uniform(5, 30), 2),
                    "conditional_behavior": None,
                },
            }

    # Shared state motif: 2+ agents writing to same data store
    if has_shared_state:
        for wid in worker_ids[:3]:
            edge_counter += 1
            edges[f"e{edge_counter}"] = {
                "structural": {"edge_type": "writes_to", "source": wid, "target": ds_id},
                "behavioral": {
                    "traversal_count": random.randint(2, 6),
                    "activation_rate": round(random.random(), 4),
                    "never_activated": False,
                    "data_flow": {"avg_data_size_bytes": 2048.0, "schema_mismatch_count": 0, "null_data_count": 0},
                    "error_crossings": {"errors_traversed": 0, "error_types": [], "downstream_impact": 0},
                    "latency_contribution_ms": round(random.uniform(1, 10), 2),
                    "conditional_behavior": None,
                },
            }

    # Trust boundary crossing: worker -> external
    edge_counter += 1
    edges[f"e{edge_counter}"] = {
        "structural": {"edge_type": "calls", "source": worker_ids[0], "target": ext_id},
        "behavioral": {
            "traversal_count": random.randint(0, 3),
            "activation_rate": round(random.random(), 4),
            "never_activated": False,
            "data_flow": {"avg_data_size_bytes": 0.0, "schema_mismatch_count": 0, "null_data_count": 0},
            "error_crossings": {"errors_traversed": 0, "error_types": [], "downstream_impact": 0},
            "latency_contribution_ms": round(random.uniform(20, 200), 2),
            "conditional_behavior": None,
        },
    }

    return {
        "repo_id": repo_id,
        "framework": framework,
        "total_runs": total_runs,
        "taxonomy_preconditions": preconditions,
        "nodes": nodes,
        "edges": edges,
    }


# =========================================================================
# 2.  Build 20 enriched graphs
# =========================================================================

FRAMEWORKS = ["crewai", "langgraph", "autogen"]

enriched_graphs: list[dict] = []

for i in range(20):
    fw = FRAMEWORKS[i % 3]
    has_shared = (i % 3 == 0)  # every 3rd repo
    has_hub = (i % 4 == 0)     # every 4th repo
    has_fail = (i % 2 == 0)    # every other repo
    fragile = 2 if i % 5 == 0 else 0

    graph = _make_enriched_graph(
        repo_index=i,
        framework=fw,
        has_shared_state=has_shared,
        has_hub_spoke=has_hub,
        has_failures=has_fail,
        fragile_nodes=fragile,
    )
    enriched_graphs.append(graph)

# Add one extreme outlier for novel pattern detection
outlier = _make_enriched_graph(
    repo_index=99,
    framework="crewai",
    has_shared_state=True,
    has_hub_spoke=True,
    has_failures=True,
    fragile_nodes=4,
)
# Make the outlier have extreme latency and failure rates
for nid in outlier["nodes"]:
    beh = outlier["nodes"][nid]["behavioral"]
    beh["latency"]["p50"] = 9500.0
    beh["throughput"]["failure_rate"] = 0.85
    beh["throughput"]["failed"] = 20
    beh["error_behavior"]["errors_occurred"] = 25
    beh["error_behavior"]["propagated_downstream"] = 15
    beh["error_behavior"]["propagation_rate"] = 0.6
    beh["error_behavior"]["observed_error_handling"] = ["propagate"]
enriched_graphs.append(outlier)


# =========================================================================
# 3.  Run knowledge-base functions
# =========================================================================

print("=" * 80)
print("(a) MANIFESTATION PROBABILITIES")
print("=" * 80)

manifestation = compute_manifestation_probabilities(enriched_graphs)
for pc_id, data in manifestation.items():
    if data["sample_size"] > 0:
        print(f"\n  {pc_id}:")
        print(f"    probability          : {data['probability']}")
        print(f"    confidence_interval  : {data['confidence_interval']}")
        print(f"    sample_size          : {data['sample_size']}")
        print(f"    manifested_count     : {data.get('manifested_count', 'N/A')}")
        print(f"    severity             : {data['severity_when_manifested']}")

# Count how many preconditions had sample_size > 0
present_count = sum(1 for d in manifestation.values() if d["sample_size"] > 0)
print(f"\n  Total preconditions with data: {present_count} / {len(TAXONOMY_PRECONDITIONS)}")


print("\n" + "=" * 80)
print("(b) NOVEL PATTERNS DETECTED")
print("=" * 80)

novel = detect_novel_patterns(enriched_graphs)
if novel:
    for np_ in novel:
        print(f"\n  repo_id                 : {np_['repo_id']}")
        print(f"  anomaly_score           : {np_['anomaly_score']}")
        print(f"  most_anomalous_dimension: {np_['most_anomalous_dimension']}")
        print(f"  behavioral_fingerprint  : {json.dumps(np_['behavioral_fingerprint'], indent=4)}")
        print(f"  z_scores                : {json.dumps(np_['z_scores'], indent=4)}")
else:
    print("  (none detected)")


print("\n" + "=" * 80)
print("(c) FRAMEWORK COMPARISON")
print("=" * 80)

comparisons = compare_frameworks(enriched_graphs)
if comparisons:
    for comp in comparisons:
        print(f"\n  Motif: {comp['motif_name']}")
        print(f"  Frameworks compared: {comp['frameworks_compared']}")
        for fw, fw_data in comp["per_framework"].items():
            bd = fw_data["behavioral_distribution"]
            print(f"    {fw} ({fw_data['repos_count']} repos):")
            print(f"      failure_rate         : {bd['failure_rate']}")
            print(f"      confidence_interval  : {bd['confidence_interval_95']}")
            print(f"      avg_error_rate       : {bd.get('avg_error_rate', 'N/A')}")
            print(f"      avg_latency_p50_ms   : {bd.get('avg_latency_p50_ms', 'N/A')}")
else:
    print("  (no motifs found in 2+ frameworks)")


print("\n" + "=" * 80)
print("(d) FRAGILITY MAP")
print("=" * 80)

fragility_map = build_fragility_map(enriched_graphs)
if fragility_map:
    for entry in fragility_map:
        print(f"\n  Structural position: {entry['structural_position']}")
        print(f"    avg_tool_call_failure_rate : {entry['avg_tool_call_failure_rate']}")
        print(f"    max_tool_call_failure_rate : {entry['max_tool_call_failure_rate']}")
        print(f"    sensitivity_score          : {entry['sensitivity_score']}")
        print(f"    max_sensitivity_score      : {entry['max_sensitivity_score']}")
        print(f"    affected_repos_count       : {entry['affected_repos_count']}")
        print(f"    total_nodes_analyzed       : {entry['total_nodes_analyzed']}")
        print(f"    quality_dependent_rate     : {entry['quality_dependent_rate']}")
        print(f"    avg_retry_activations      : {entry['avg_retry_activations']}")
        print(f"    top_fragile_nodes (up to 3):")
        for node in entry["top_fragile_nodes"][:3]:
            print(f"      {node['repo_id']}/{node['node_id']}: "
                  f"sensitivity={node['sensitivity_score']}, "
                  f"tcfr={node['tool_call_failure_rate']}")
else:
    print("  (no fragility data)")


print("\n" + "=" * 80)
print("PATTERN KNOWLEDGE BASE (full)")
print("=" * 80)

pkb = build_pattern_knowledge_base(enriched_graphs)
for pat in pkb:
    print(f"\n  {pat['pattern_id']}:")
    print(f"    pattern_name : {pat['pattern_name']}")
    print(f"    prevalence   : {pat['prevalence']['repos_count']}/{pat['prevalence']['total_repos']} "
          f"({pat['prevalence']['prevalence_rate']})")
    print(f"    risk_level   : {pat['risk_assessment']['risk_level']} "
          f"(score={pat['risk_assessment']['risk_score']})")
    bd = pat["behavioral_distribution"]
    print(f"    failure_rate  : {bd['failure_rate']}  CI={bd['confidence_interval_95']}")


# =========================================================================
# (e) FINGERPRINTS & NORMALIZATION
# =========================================================================

print("\n" + "=" * 80)
print("(e) FINGERPRINTS & NORMALIZATION")
print("=" * 80)

from stratum_lab.query.fingerprint import (
    compute_graph_fingerprint,
    compute_normalization_constants,
    normalize_feature_vector,
)

# Compute fingerprints for all enriched graphs
fingerprints: dict[str, dict] = {}
for graph in enriched_graphs:
    rid = graph["repo_id"]
    fingerprints[rid] = compute_graph_fingerprint(graph)

print(f"  Fingerprints computed: {len(fingerprints)}")
if fingerprints:
    sample_fp = list(fingerprints.values())[0]
    print(f"  Feature vector length: {len(sample_fp['feature_vector'])}")

# Compute normalization constants
all_fps = list(fingerprints.values())
norm_constants = compute_normalization_constants(all_fps)
has_norm = bool(norm_constants.get("min")) and bool(norm_constants.get("max"))
print(f"  Normalization constants saved: {has_norm}")

if has_norm:
    print(f"  Min vector: {[round(v, 2) for v in norm_constants['min']]}")
    print(f"  Max vector: {[round(v, 2) for v in norm_constants['max']]}")

# Show sample fingerprints (first 3 repos)
for rid in list(fingerprints.keys())[:3]:
    fp = fingerprints[rid]
    raw_vec = fp["feature_vector"]
    if has_norm:
        norm_vec = normalize_feature_vector(raw_vec, norm_constants)
    else:
        norm_vec = raw_vec
    print(f"\n  Sample fingerprint ({rid}):")
    print(f"    motifs:         {fp['motifs']}")
    print(f"    topology_hash:  {fp['topology_hash'][:16]}...")
    print(f"    raw_vector:     {[round(v, 2) for v in raw_vec[:10]]}... ({len(raw_vec)} dims)")
    print(f"    normalized:     {[round(v, 2) for v in norm_vec[:10]]}... ({len(norm_vec)} dims)")

# =========================================================================
# (f) CROSS-PATTERN INTERACTIONS
# =========================================================================

print("\n" + "=" * 80)
print("(f) CROSS-PATTERN INTERACTIONS")
print("=" * 80)

from stratum_lab.knowledge.interactions import compute_interaction_matrix

interactions = compute_interaction_matrix(enriched_graphs)
interaction_list = interactions.get("interactions", [])
synergistic = interactions.get("synergistic_pairs", [])

print(f"  Total interaction pairs analyzed: {len(interaction_list)}")
print(f"  Synergistic pairs (>1.5x):       {len(synergistic)}")

if interactions.get("most_dangerous_combination"):
    combo = interactions["most_dangerous_combination"]
    print(f"\n  Most dangerous combination:")
    print(f"    {combo['precondition_a']} + {combo['precondition_b']}")
    print(f"    interaction_effect: {combo['interaction_effect']}")
    print(f"    P(fail|both):      {combo['p_fail_both']}")
    print(f"    co_occurrence:     {combo['co_occurrence_count']}")

if synergistic:
    print(f"\n  Synergistic pairs:")
    for pair in synergistic[:10]:
        print(f"    {pair['precondition_a']} + {pair['precondition_b']}: "
              f"effect={pair['interaction_effect']}, "
              f"P(both)={pair['p_fail_both']}, "
              f"n={pair['co_occurrence_count']}")

print("\nDone.")
