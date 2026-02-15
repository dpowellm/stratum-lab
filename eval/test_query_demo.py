"""Query layer demo — full product flow.

Creates a synthetic 4-agent crewAI system with shared state, a 3-deep
delegation chain, and no guardrails.  Runs it through the query layer
against a synthetic knowledge base.

Prints:
  (a) Fingerprint
  (b) Top 5 pattern matches with similarity scores
  (c) Predicted risks with probabilities and evidence
  (d) Full risk report in JSON format
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stratum_lab.knowledge.patterns import build_pattern_knowledge_base
from stratum_lab.knowledge.taxonomy import compute_manifestation_probabilities
from stratum_lab.knowledge.fragility import build_fragility_map
from stratum_lab.query.fingerprint import (
    compute_graph_fingerprint,
    compute_normalization_constants,
    normalize_feature_vector,
)
from stratum_lab.query.matcher import match_against_dataset
from stratum_lab.query.predictor import predict_risks
from stratum_lab.query.report import generate_risk_report

random.seed(42)

SEPARATOR = "=" * 80

# =========================================================================
# 1. Customer structural graph
# =========================================================================

customer_graph = {
    "repo_id": "customer_repo",
    "framework": "crewai",
    "taxonomy_preconditions": [
        "shared_state_no_arbitration",
        "no_timeout_on_delegation",
        "unhandled_tool_failure",
    ],
    "nodes": {
        "agent_manager": {
            "structural": {"node_type": "agent", "name": "Manager"},
            "behavioral": None,
        },
        "agent_researcher": {
            "structural": {
                "node_type": "agent", "name": "Researcher",
                "error_handling": {"strategy": "fail_silent"},
            },
            "behavioral": None,
        },
        "agent_writer": {
            "structural": {"node_type": "agent", "name": "Writer"},
            "behavioral": None,
        },
        "agent_reviewer": {
            "structural": {"node_type": "agent", "name": "Reviewer"},
            "behavioral": None,
        },
        "cap_web_search": {
            "structural": {"node_type": "capability", "name": "WebSearch", "kind": "tool"},
            "behavioral": None,
        },
        "cap_llm_call": {
            "structural": {"node_type": "capability", "name": "LLMCall", "kind": "llm"},
            "behavioral": None,
        },
        "ds_shared_memory": {
            "structural": {"node_type": "data_store", "name": "SharedMemory"},
            "behavioral": None,
        },
        "ext_web_api": {
            "structural": {"node_type": "external", "name": "WebAPI"},
            "behavioral": None,
        },
    },
    "edges": {
        "e1": {"structural": {"edge_type": "delegates_to", "source": "agent_manager", "target": "agent_researcher"}},
        "e2": {"structural": {"edge_type": "delegates_to", "source": "agent_researcher", "target": "agent_writer"}},
        "e3": {"structural": {"edge_type": "delegates_to", "source": "agent_writer", "target": "agent_reviewer"}},
        "e4": {"structural": {"edge_type": "uses", "source": "agent_researcher", "target": "cap_web_search"}},
        "e5": {"structural": {"edge_type": "uses", "source": "agent_researcher", "target": "cap_llm_call"}},
        "e6": {"structural": {"edge_type": "writes_to", "source": "agent_researcher", "target": "ds_shared_memory"}},
        "e7": {"structural": {"edge_type": "writes_to", "source": "agent_writer", "target": "ds_shared_memory"}},
        "e8": {"structural": {"edge_type": "reads_from", "source": "agent_reviewer", "target": "ds_shared_memory"}},
        "e9": {"structural": {"edge_type": "calls", "source": "cap_web_search", "target": "ext_web_api"}},
    },
}


# =========================================================================
# 2. Build a synthetic knowledge base (dataset of enriched graphs)
# =========================================================================

def _rand_behavioral(node_type="agent", has_errors=False, fragile=False):
    activation_count = random.randint(3, 30)
    total_runs = 3
    tasks = activation_count
    completed = tasks - random.randint(0, 2)
    failed = random.randint(1, 5) if has_errors else 0
    errors = random.randint(1, 8) if has_errors else 0
    propagated = random.randint(0, errors)
    handling = random.sample(["retry", "propagate", "fallback", "ignore"], k=random.randint(1, 2)) if has_errors else []
    failures = random.randint(2, 10) if fragile else random.randint(0, 1)
    total_tool = random.randint(max(failures, 5), failures + 15)
    retries = random.randint(1, 5) if fragile else 0
    p50 = round(random.uniform(50, 2000), 2)

    return {
        "activation_count": activation_count,
        "activation_rate": round(min(activation_count / total_runs, 1.0), 4),
        "throughput": {
            "tasks_received": tasks,
            "completed": max(completed, 0),
            "failed": failed,
            "failure_rate": round(failed / max(tasks, 1), 4),
        },
        "latency": {
            "p50": p50,
            "p95": round(p50 * random.uniform(1.3, 3.0), 2),
            "p99": round(p50 * random.uniform(2.0, 5.0), 2),
            "variance": round(random.uniform(10, 500), 2),
            "sample_count": random.randint(3, 30),
        },
        "error_behavior": {
            "errors_occurred": errors,
            "propagated_downstream": propagated,
            "swallowed": errors - propagated,
            "propagation_rate": round(propagated / max(errors, 1), 4),
            "default_values_used": random.randint(0, 2),
            "observed_error_handling": handling,
            "structural_prediction_match": random.choice([True, False, None]) if has_errors else None,
        },
        "decision_behavior": None,
        "model_sensitivity": {
            "tool_call_failures": failures,
            "tool_call_failure_rate": round(failures / max(total_tool, 1), 4),
            "retry_activations": retries,
            "quality_dependent": fragile,
        },
        "resource_usage": {
            "avg_tokens": round(random.uniform(200, 3000), 2),
            "avg_tool_calls": round(random.uniform(0, 5), 2),
            "avg_llm_calls": round(random.uniform(1, 8), 2),
            "avg_iterations": round(random.uniform(1, 4), 2),
        },
    }


def _make_dataset_graph(idx, framework, has_shared=False, has_hub=False, has_fail=False, fragile=0):
    repo_id = f"dataset_repo_{idx:03d}"
    nodes = {}
    edges = {}
    ec = 0

    # Agents
    manager_id = "agent_manager"
    nodes[manager_id] = {
        "structural": {"node_type": "agent", "name": "Manager"},
        "behavioral": _rand_behavioral("agent", has_errors=has_fail),
    }
    workers = ["Alpha", "Beta", "Gamma", "Delta"]
    worker_ids = [f"agent_{w.lower()}" for w in workers]
    for i, (wid, wname) in enumerate(zip(worker_ids, workers)):
        nodes[wid] = {
            "structural": {"node_type": "agent", "name": wname},
            "behavioral": _rand_behavioral("agent", has_errors=(has_fail and i == 0), fragile=(i < fragile)),
        }

    ds_id = "ds_shared_state"
    nodes[ds_id] = {
        "structural": {"node_type": "data_store", "name": "SharedState"},
        "behavioral": _rand_behavioral("data_store"),
    }
    ext_id = "ext_api"
    nodes[ext_id] = {
        "structural": {"node_type": "external", "name": "ExternalAPI"},
        "behavioral": _rand_behavioral("external"),
    }

    if has_hub:
        for wid in worker_ids:
            ec += 1
            edges[f"e{ec}"] = {
                "structural": {"edge_type": "delegates_to", "source": manager_id, "target": wid},
                "behavioral": {"traversal_count": random.randint(1, 5), "activation_rate": round(random.random(), 4),
                                "never_activated": False,
                                "data_flow": {"avg_data_size_bytes": 512.0, "schema_mismatch_count": 0, "null_data_count": 0},
                                "error_crossings": {"errors_traversed": 0, "error_types": [], "downstream_impact": 0},
                                "latency_contribution_ms": round(random.uniform(5, 50), 2), "conditional_behavior": None},
            }
    else:
        for i in range(min(3, len(worker_ids))):
            ec += 1
            src = manager_id if i == 0 else worker_ids[i - 1]
            edges[f"e{ec}"] = {
                "structural": {"edge_type": "delegates_to", "source": src, "target": worker_ids[i]},
                "behavioral": {"traversal_count": random.randint(1, 4), "activation_rate": round(random.random(), 4),
                                "never_activated": False,
                                "data_flow": {"avg_data_size_bytes": 256.0, "schema_mismatch_count": 0, "null_data_count": 0},
                                "error_crossings": {"errors_traversed": 0, "error_types": [], "downstream_impact": 0},
                                "latency_contribution_ms": round(random.uniform(5, 30), 2), "conditional_behavior": None},
            }

    if has_shared:
        for wid in worker_ids[:3]:
            ec += 1
            edges[f"e{ec}"] = {
                "structural": {"edge_type": "writes_to", "source": wid, "target": ds_id},
                "behavioral": {"traversal_count": random.randint(2, 6), "activation_rate": round(random.random(), 4),
                                "never_activated": False,
                                "data_flow": {"avg_data_size_bytes": 2048.0, "schema_mismatch_count": 0, "null_data_count": 0},
                                "error_crossings": {"errors_traversed": 0, "error_types": [], "downstream_impact": 0},
                                "latency_contribution_ms": round(random.uniform(1, 10), 2), "conditional_behavior": None},
            }

    ec += 1
    edges[f"e{ec}"] = {
        "structural": {"edge_type": "calls", "source": worker_ids[0], "target": ext_id},
        "behavioral": {"traversal_count": random.randint(0, 3), "activation_rate": round(random.random(), 4),
                        "never_activated": False,
                        "data_flow": {"avg_data_size_bytes": 0.0, "schema_mismatch_count": 0, "null_data_count": 0},
                        "error_crossings": {"errors_traversed": 0, "error_types": [], "downstream_impact": 0},
                        "latency_contribution_ms": round(random.uniform(20, 200), 2), "conditional_behavior": None},
    }

    preconditions = random.sample([
        "shared_state_no_arbitration", "no_timeout_on_delegation",
        "unhandled_tool_failure", "deep_delegation_chain",
        "no_human_checkpoint", "trust_boundary_no_validation",
    ], k=random.randint(2, 4))

    return {
        "repo_id": repo_id,
        "framework": framework,
        "total_runs": 3,
        "taxonomy_preconditions": preconditions,
        "nodes": nodes,
        "edges": edges,
    }


# Build 30 dataset graphs
dataset_graphs = []
for i in range(30):
    fw = ["crewai", "langgraph", "autogen"][i % 3]
    dataset_graphs.append(_make_dataset_graph(
        i, fw,
        has_shared=(i % 3 == 0),
        has_hub=(i % 4 == 0),
        has_fail=(i % 2 == 0),
        fragile=2 if i % 5 == 0 else 0,
    ))


# =========================================================================
# 3. Build knowledge base
# =========================================================================

tmpdir = tempfile.mkdtemp(prefix="stratum_query_demo_")
kb_dir = os.path.join(tmpdir, "knowledge_base")
os.makedirs(kb_dir, exist_ok=True)

# Patterns
patterns = build_pattern_knowledge_base(dataset_graphs)
with open(os.path.join(kb_dir, "patterns.json"), "w") as f:
    json.dump(patterns, f, indent=2, default=str)

# Taxonomy
taxonomy = compute_manifestation_probabilities(dataset_graphs)
with open(os.path.join(kb_dir, "taxonomy_probabilities.json"), "w") as f:
    json.dump(taxonomy, f, indent=2, default=str)

# Fragility
fragility = build_fragility_map(dataset_graphs)
with open(os.path.join(kb_dir, "fragility_map.json"), "w") as f:
    json.dump(fragility, f, indent=2, default=str)

# Framework comparisons
from stratum_lab.knowledge.patterns import compare_frameworks
fw_comp = compare_frameworks(dataset_graphs)
with open(os.path.join(kb_dir, "framework_comparisons.json"), "w") as f:
    json.dump(fw_comp, f, indent=2, default=str)

# Fingerprints + normalization
dataset_fps = {}
all_fps_list = []
for eg in dataset_graphs:
    fp = compute_graph_fingerprint(eg)
    dataset_fps[eg["repo_id"]] = fp
    all_fps_list.append(fp)

norm_constants = compute_normalization_constants(all_fps_list)
with open(os.path.join(kb_dir, "fingerprints.json"), "w") as f:
    json.dump(dataset_fps, f, indent=2, default=str)
with open(os.path.join(kb_dir, "normalization.json"), "w") as f:
    json.dump(norm_constants, f, indent=2, default=str)


# =========================================================================
# 4. Run the product query flow
# =========================================================================

print(SEPARATOR)
print("STRATUM QUERY DEMO — FULL PRODUCT FLOW")
print(SEPARATOR)
print(f"\nKnowledge base: {len(dataset_graphs)} enriched repos, {len(patterns)} patterns")
print(f"Customer graph: {customer_graph['repo_id']} ({customer_graph['framework']})")
print(f"  Nodes: {len(customer_graph['nodes'])}")
print(f"  Edges: {len(customer_graph['edges'])}")
print(f"  Preconditions: {customer_graph['taxonomy_preconditions']}")

# ---- (a) Fingerprint ----
print(f"\n{SEPARATOR}")
print("(a) GRAPH FINGERPRINT")
print(SEPARATOR)

fp = compute_graph_fingerprint(customer_graph)
print(f"  feature_vector (raw, {len(fp['feature_vector'])} elements):")
for i, val in enumerate(fp["feature_vector"]):
    print(f"    [{i:2d}] {val:.6f}")

# Normalize using dataset normalization constants
import json as _json
_norm_path = os.path.join(kb_dir, "normalization.json")
with open(_norm_path, "r") as _f:
    _norm_constants = _json.load(_f)
norm_vec = normalize_feature_vector(fp["feature_vector"], _norm_constants)
print(f"\n  feature_vector (normalized to [0,1], {len(norm_vec)} elements):")
for i, val in enumerate(norm_vec):
    print(f"    [{i:2d}] {val:.6f}")
print(f"\n  motifs: {fp['motifs']}")
print(f"  topology_hash: {fp['topology_hash']}")
print(f"\n  node_type_distribution: {json.dumps(fp['node_type_distribution'], indent=4)}")
print(f"\n  edge_type_distribution: {json.dumps(fp['edge_type_distribution'], indent=4)}")
print(f"\n  structural_metrics: {json.dumps(fp['structural_metrics'], indent=4)}")

# ---- (b) Pattern matches ----
print(f"\n{SEPARATOR}")
print("(b) TOP 5 PATTERN MATCHES")
print(SEPARATOR)

matches = match_against_dataset(fp, kb_dir, top_k=5)
for i, m in enumerate(matches):
    print(f"\n  Match {i + 1}:")
    print(f"    pattern_name:      {m.pattern_name}")
    print(f"    pattern_id:        {m.pattern_id}")
    print(f"    similarity_score:  {m.similarity_score:.4f}")
    print(f"    match_type:        {m.match_type}")
    print(f"    matched_repos:     {m.matched_repos}")
    if m.behavioral_summary:
        print(f"    behavioral_summary: {json.dumps(m.behavioral_summary, indent=6, default=str)}")

# ---- (c) Risk predictions ----
print(f"\n{SEPARATOR}")
print("(c) PREDICTED RISKS")
print(SEPARATOR)

prediction = predict_risks(
    customer_graph, matches,
    customer_graph["taxonomy_preconditions"],
    kb_dir,
)

print(f"\n  overall_risk_score: {prediction.overall_risk_score:.1f}")
print(f"  archetype:         {prediction.archetype}")
print(f"  archetype_prevalence: {prediction.archetype_prevalence:.2f}")
print(f"\n  Predicted risks ({len(prediction.predicted_risks)}):")
for risk in prediction.predicted_risks:
    print(f"\n    {risk.precondition_id}: {risk.precondition_name}")
    print(f"      manifestation_probability: {risk.manifestation_probability:.4f}")
    print(f"      confidence_interval:       {risk.confidence_interval}")
    print(f"      sample_size:               {risk.sample_size}")
    print(f"      severity_when_manifested:  {risk.severity_when_manifested}")
    print(f"      fragility_flag:            {risk.fragility_flag}")
    print(f"      behavioral_description:    {risk.behavioral_description}")
    print(f"      remediation:               {risk.remediation}")

print(f"\n  Positive signals ({len(prediction.positive_signals)}):")
for signal in prediction.positive_signals:
    print(f"    + {signal}")

print(f"\n  Structural-only risks ({len(prediction.structural_only_risks)}):")
for sr in prediction.structural_only_risks:
    print(f"    - {sr}")

print(f"\n  Dataset coverage: {json.dumps(prediction.dataset_coverage, indent=4, default=str)}")

# ---- (d) Semantic analysis ----
print(f"\n{SEPARATOR}")
print("(d) SEMANTIC ANALYSIS")
print(SEPARATOR)

sem = getattr(prediction, "semantic_analysis", {}) or {}
if sem:
    print(f"  semantic_risk_score:           {sem.get('semantic_risk_score', 0):.1f}")
    print(f"  unvalidated_handoff_fraction:  {sem.get('unvalidated_handoff_fraction', 0):.2f}")
    print(f"  semantic_chain_depth:          {sem.get('semantic_chain_depth', 0)}")
    print(f"  max_blast_radius:              {sem.get('max_blast_radius', 0)}")
    print(f"  classification_injection_points:{sem.get('classification_injection_points', 0)}")
    nondet = sem.get("nondeterministic_nodes", [])
    if nondet:
        print(f"  nondeterministic_nodes:        {', '.join(nondet)}")
    verdict = sem.get("verdict", "")
    if verdict:
        print(f"  verdict:                       {verdict}")
else:
    print("  (no semantic analysis — expected if no semantic_lineage on graph)")

# ---- (e) Full risk report (JSON) ----
print(f"\n{SEPARATOR}")
print("(e) FULL RISK REPORT (JSON)")
print(SEPARATOR)

report_json = generate_risk_report(prediction, customer_graph, output_format="json")
print(json.dumps(report_json, indent=2, default=str))

# ---- (f) Risk report (Markdown excerpt) ----
print(f"\n{SEPARATOR}")
print("(f) RISK REPORT (MARKDOWN)")
print(SEPARATOR)

report_md = generate_risk_report(prediction, customer_graph, output_format="markdown")
print(report_md)

# Cleanup
import shutil
shutil.rmtree(tmpdir, ignore_errors=True)

print(f"\n{SEPARATOR}")
print("QUERY DEMO COMPLETE")
print(SEPARATOR)
