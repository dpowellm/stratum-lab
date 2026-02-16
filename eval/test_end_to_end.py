#!/usr/bin/env python3
"""End-to-end integration test for the full stratum-lab pipeline.

Exercises the ENTIRE pipeline path using a realistic structural graph and
mock container execution. No synthetic _make_behavioral_record(). The
behavioral record is BUILT by the overlay from collected events, exactly
as it would be in production.

Output: eval/outputs/end-to-end-demo.txt
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure stratum_lab is importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "end-to-end-demo.txt"

buf = io.StringIO()


def tee(msg: str = "") -> None:
    print(msg)
    buf.write(msg + "\n")


# ============================================================================
# FIXTURE 1: Realistic structural graph
# ============================================================================

REPO_ID = "integration-test_crewai-research-crew"

STRUCTURAL_GRAPH = {
    "repo_id": REPO_ID,
    "repo_full_name": REPO_ID,
    "framework": "crewai",
    "scanner_version": "0.9.0",
    "nodes": {
        "agent_researcher": {
            "node_type": "agent", "node_name": "Researcher",
            "framework": "crewai", "file": "agents.py", "line": 10,
        },
        "agent_writer": {
            "node_type": "agent", "node_name": "Writer",
            "framework": "crewai", "file": "agents.py", "line": 25,
        },
        "agent_reviewer": {
            "node_type": "agent", "node_name": "Reviewer",
            "framework": "crewai", "file": "agents.py", "line": 40,
        },
        "cap_web_search": {
            "node_type": "capability", "node_name": "WebSearchTool",
            "file": "tools.py", "line": 5,
        },
        "cap_file_writer": {
            "node_type": "capability", "node_name": "FileWriterTool",
            "file": "tools.py", "line": 15,
        },
        "ds_shared_notes": {
            "node_type": "data_store", "node_name": "SharedNotes",
            "file": "state.py", "line": 1,
        },
    },
    "edges": {
        "e1": {"source": "agent_researcher", "target": "agent_writer", "edge_type": "delegates_to"},
        "e2": {"source": "agent_writer", "target": "agent_reviewer", "edge_type": "delegates_to"},
        "e3": {"source": "agent_researcher", "target": "cap_web_search", "edge_type": "uses"},
        "e4": {"source": "agent_writer", "target": "cap_file_writer", "edge_type": "uses"},
        "e5": {"source": "agent_researcher", "target": "ds_shared_notes", "edge_type": "writes_to"},
        "e6": {"source": "agent_writer", "target": "ds_shared_notes", "edge_type": "reads_from"},
    },
    "structural_metrics": {
        "total_agents": 3,
        "total_capabilities": 2,
        "total_data_stores": 1,
        "delegation_depth_max": 2,
        "has_human_checkpoint": 0,
        "shared_state_writer_count": 1,
        "observability_coverage_ratio": 0.0,
        "agent_to_capability_ratio": 1.5,
    },
    "preconditions": [
        "unbounded_delegation_depth",
        "shared_state_no_arbitration",
        "unhandled_tool_failure",
        "no_output_validation",
    ],
    "findings": [
        {"finding_id": "STRAT-DC-001", "finding_name": "Unsupervised delegation chain"},
        {"finding_id": "STRAT-OC-002", "finding_name": "Shared state contention"},
        {"finding_id": "STRAT-EA-001", "finding_name": "Unhandled tool failure"},
        {"finding_id": "STRAT-SI-004", "finding_name": "Missing output validation"},
    ],
    "runnability_score": 0.85,
    "topology_hash": "e2e_test_topo_001",
    "xcomp_findings": [],
}


# ============================================================================
# FIXTURE 2: Realistic event stream generator
# ============================================================================

def _make_node(node_type: str, node_id: str, node_name: str) -> dict:
    """Create a node reference dict matching patcher format."""
    return {"node_type": node_type, "node_id": node_id, "node_name": node_name}


def generate_realistic_events(run_id: str, input_hash: str) -> list[dict]:
    """Generate a realistic event stream as if a crewAI crew actually ran."""
    ts = 1700000000000000000  # base timestamp_ns
    events: list[dict] = []

    def evt(event_type, source=None, target=None, edge_type=None, payload=None, dt=1000000):
        nonlocal ts
        ts += dt
        e = {
            "event_id": f"evt_{len(events):06d}",
            "event_type": event_type,
            "timestamp_ns": ts,
            "run_id": run_id,
            "repo_id": REPO_ID,
            "source_node": source or {},
            "target_node": target or {},
            "edge_type": edge_type or "",
            "payload": payload or {},
            "parent_event_id": "",
            "stack_depth": 0,
        }
        events.append(e)
        return e

    researcher = _make_node("agent", "crewai:Researcher:agents.py:10", "Researcher")
    writer = _make_node("agent", "crewai:Writer:agents.py:25", "Writer")
    reviewer = _make_node("agent", "crewai:Reviewer:agents.py:40", "Reviewer")
    web_search = _make_node("capability", "cap_web_search", "WebSearchTool")
    file_writer = _make_node("capability", "cap_file_writer", "FileWriterTool")
    shared_notes = _make_node("data_store", "ds_shared_notes", "SharedNotes")

    # Execution start
    evt("execution.start", source=researcher, payload={"input_hash": input_hash})

    # Researcher agent starts
    evt("agent.task_start", source=researcher, payload={"task": "research"})

    # Researcher calls web search tool
    evt("tool.invoked", source=researcher, target=web_search, edge_type="uses")
    evt("tool.completed", source=researcher, target=web_search, edge_type="uses",
        payload={"status": "success", "output_hash": "abc123"})

    # Researcher makes LLM call
    start_evt = evt("llm.call_start", source=researcher, payload={"model": "gpt-4"})
    evt("llm.call_end", source=researcher,
        payload={"output_hash": "def456", "output_type": "structured_json", "tokens": 500},
        dt=50000000)

    # Researcher writes to shared notes
    evt("data.write", source=researcher, target=shared_notes, edge_type="writes_to",
        payload={"state_key": "research_notes", "data_hash": "ghi789"})

    # Researcher completes, delegates to Writer
    evt("agent.task_end", source=researcher, payload={"status": "success", "output_hash": "jkl012"})
    evt("delegation.initiated", source=researcher, target=writer, edge_type="delegates_to")

    # Writer starts
    evt("agent.task_start", source=writer, payload={"task": "write_article"})

    # Writer reads shared notes
    evt("data.read", source=writer, target=shared_notes, edge_type="reads_from",
        payload={"state_key": "research_notes"})

    # Writer uses file writer tool
    evt("tool.invoked", source=writer, target=file_writer, edge_type="uses")
    evt("tool.completed", source=writer, target=file_writer, edge_type="uses",
        payload={"status": "success", "output_hash": "mno345"})

    # Writer LLM call
    evt("llm.call_start", source=writer, payload={"model": "gpt-4"})
    evt("llm.call_end", source=writer,
        payload={"output_hash": "pqr678", "output_type": "long_text", "tokens": 1200},
        dt=80000000)

    # Writer delegates to Reviewer
    evt("agent.task_end", source=writer, payload={"status": "success", "output_hash": "stu901"})
    evt("delegation.initiated", source=writer, target=reviewer, edge_type="delegates_to")
    evt("delegation.completed", source=writer, target=reviewer, edge_type="delegates_to")

    # Reviewer starts
    evt("agent.task_start", source=reviewer, payload={"task": "review"})
    evt("llm.call_start", source=reviewer, payload={"model": "gpt-4"})
    evt("llm.call_end", source=reviewer,
        payload={"output_hash": "vwx234", "output_type": "classification", "tokens": 300},
        dt=30000000)
    evt("agent.task_end", source=reviewer, payload={"status": "success", "output_hash": "yza567"})

    # Execution end
    evt("execution.end", payload={"status": "success", "total_events": len(events)})

    return events


def generate_error_events(run_id: str, input_hash: str) -> list[dict]:
    """Generate events with a tool failure and error propagation (run_002)."""
    ts = 1700000000000000000
    events: list[dict] = []

    def evt(event_type, source=None, target=None, edge_type=None, payload=None, dt=1000000):
        nonlocal ts
        ts += dt
        e = {
            "event_id": f"evt_{len(events):06d}",
            "event_type": event_type,
            "timestamp_ns": ts,
            "run_id": run_id,
            "repo_id": REPO_ID,
            "source_node": source or {},
            "target_node": target or {},
            "edge_type": edge_type or "",
            "payload": payload or {},
            "parent_event_id": "",
            "stack_depth": 0,
        }
        events.append(e)
        return e

    researcher = _make_node("agent", "crewai:Researcher:agents.py:10", "Researcher")
    writer = _make_node("agent", "crewai:Writer:agents.py:25", "Writer")
    reviewer = _make_node("agent", "crewai:Reviewer:agents.py:40", "Reviewer")
    web_search = _make_node("capability", "cap_web_search", "WebSearchTool")
    shared_notes = _make_node("data_store", "ds_shared_notes", "SharedNotes")

    # Execution start
    evt("execution.start", source=researcher, payload={"input_hash": input_hash})

    # Researcher starts
    evt("agent.task_start", source=researcher, payload={"task": "research"})

    # Tool FAILURE on web search
    evt("tool.invoked", source=researcher, target=web_search, edge_type="uses")
    evt("tool.call_failure", source=researcher, target=web_search,
        payload={"tool_name": "WebSearchTool", "error_type": "connection_timeout",
                 "error_message": "Connection timed out after 30s"})

    # Error occurred and propagated
    evt("error.occurred", source=researcher,
        payload={"error_type": "tool_failure", "error_node_id": "cap_web_search",
                 "error_message": "WebSearchTool failed: connection timeout"})

    # LLM call still happens (researcher tries to continue)
    evt("llm.call_start", source=researcher, payload={"model": "gpt-4"})
    evt("llm.call_end", source=researcher,
        payload={"output_hash": "err_hash", "output_type": "text", "tokens": 200},
        dt=40000000)

    # Researcher writes partial data
    evt("data.write", source=researcher, target=shared_notes, edge_type="writes_to",
        payload={"state_key": "research_notes", "data_hash": "partial_data"})

    # Researcher finishes with error status
    evt("agent.task_end", source=researcher, payload={"status": "error", "output_hash": "err_out"})

    # Error propagated to writer
    evt("error.propagated", source=researcher, target=writer,
        payload={"error_type": "upstream_failure",
                 "propagation_path": ["crewai:Researcher:agents.py:10", "crewai:Writer:agents.py:25"]})

    # Delegation still happens
    evt("delegation.initiated", source=researcher, target=writer, edge_type="delegates_to")

    # Writer starts with degraded input
    evt("agent.task_start", source=writer, payload={"task": "write_article"})
    evt("data.read", source=writer, target=shared_notes, edge_type="reads_from",
        payload={"state_key": "research_notes"})
    evt("llm.call_start", source=writer, payload={"model": "gpt-4"})
    evt("llm.call_end", source=writer,
        payload={"output_hash": "w_err", "output_type": "text", "tokens": 800},
        dt=60000000)
    evt("agent.task_end", source=writer, payload={"status": "success", "output_hash": "w_out"})

    # Delegation to reviewer
    evt("delegation.initiated", source=writer, target=reviewer, edge_type="delegates_to")
    evt("delegation.completed", source=writer, target=reviewer, edge_type="delegates_to")

    # Reviewer
    evt("agent.task_start", source=reviewer, payload={"task": "review"})
    evt("llm.call_start", source=reviewer, payload={"model": "gpt-4"})
    evt("llm.call_end", source=reviewer,
        payload={"output_hash": "r_err", "output_type": "classification", "tokens": 250},
        dt=25000000)
    evt("agent.task_end", source=reviewer, payload={"status": "success", "output_hash": "r_out"})

    evt("execution.end", payload={"status": "completed_with_errors"})

    return events


# ============================================================================
# TEST RUNNER
# ============================================================================

def test_end_to_end():
    """Full pipeline integration test using realistic event streams."""

    tee("=" * 78)
    tee("STRATUM END-TO-END INTEGRATION TEST")
    tee("=" * 78)

    checks_passed = 0
    checks_failed = 0
    check_results: list[tuple[int, str, bool, str]] = []

    def check(num: int, label: str, condition: bool, detail: str = "") -> bool:
        nonlocal checks_passed, checks_failed
        if condition:
            checks_passed += 1
            check_results.append((num, label, True, detail))
        else:
            checks_failed += 1
            check_results.append((num, label, False, detail))
        return condition

    output_dir = Path(tempfile.mkdtemp(prefix="stratum_e2e_"))

    # ========== PHASE 1: SELECTION ==========
    # We skip actual selection (needs full pool) — just use our fixture directly
    tee(f"\n  PHASE 1: Selection ..................... 1 repo (fixture)")

    # ========== PHASE 2: EXECUTION (MOCKED) ==========
    # Write structural graph
    structural_dir = output_dir / "structural"
    structural_dir.mkdir(parents=True, exist_ok=True)
    graph_file = structural_dir / f"{REPO_ID.replace('/', '_')}.json"
    with open(graph_file, "w", encoding="utf-8") as f:
        json.dump(STRUCTURAL_GRAPH, f, indent=2)

    # Generate event files
    events_dir = output_dir / "raw_events"
    events_dir.mkdir(parents=True, exist_ok=True)

    run_configs = [
        ("run_001", "input_alpha", generate_realistic_events),
        ("run_002", "input_beta", generate_error_events),
        ("run_003", "input_gamma", generate_realistic_events),
        ("run_004", "input_alpha", generate_realistic_events),  # repeat
        ("run_005", "input_alpha", generate_realistic_events),  # repeat
    ]

    total_events_by_run: dict[str, int] = {}
    for run_id, input_hash, gen_fn in run_configs:
        events = gen_fn(run_id, input_hash)
        total_events_by_run[run_id] = len(events)
        events_file = events_dir / f"{run_id}.jsonl"
        with open(events_file, "w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev, default=str) + "\n")

    tee(f"  PHASE 2: Execution (mocked) ........... 5 runs generated")

    # ========== PHASE 3: COLLECTION ==========
    from stratum_lab.collection.cli import run_collection
    runs_file = str(output_dir / "runs.json")

    # Suppress rich console output during collection
    import logging
    logging.disable(logging.WARNING)

    run_collection(str(events_dir), runs_file)

    logging.disable(logging.NOTSET)

    # Load collection output
    with open(runs_file, "r", encoding="utf-8") as f:
        collection_data = json.load(f)

    run_records = collection_data.get("run_records", [])
    tee(f"  PHASE 3: Collection ................... {len(run_records)} run records parsed")

    # ========== PHASE 4: OVERLAY ==========
    from stratum_lab.overlay.cli import run_overlay
    enriched_dir = str(output_dir / "enriched_graphs")

    run_overlay(str(structural_dir), str(events_dir), enriched_dir)

    # Load enriched graph
    enriched_files = list(Path(enriched_dir).glob("*.json"))
    enriched_graph = None
    if enriched_files:
        with open(enriched_files[0], "r", encoding="utf-8") as f:
            enriched_graph = json.load(f)

    tee(f"  PHASE 4: Overlay ...................... enriched graph produced")

    # ========== PHASE 5: KNOWLEDGE BASE ==========
    from stratum_lab.knowledge.cli import run_knowledge_build
    kb_dir = str(output_dir / "knowledge_base")

    run_knowledge_build(enriched_dir, kb_dir)

    tee(f"  PHASE 5: Knowledge base ............... built (1 repo)")

    # ========== PHASE 6: FEEDBACK EXPORT ==========
    # Build behavioral record from enriched graph, then export
    from stratum_lab.output.behavioral_record import build_behavioral_record, validate_behavioral_record
    from stratum_lab.feedback.exporter import export_feedback

    behavioral_records = []
    if enriched_graph:
        record = build_behavioral_record(
            repo_full_name=enriched_graph.get("repo_id", REPO_ID),
            execution_metadata={
                "framework": enriched_graph.get("framework", "crewai"),
                "num_runs": enriched_graph.get("total_runs", 5),
            },
            edge_validation=enriched_graph.get("edge_validation", {}),
            emergent_edges=enriched_graph.get("emergent_edges", []),
            node_activation=enriched_graph.get("node_activation", {}),
            error_propagation=enriched_graph.get("error_propagation", []),
            failure_modes=enriched_graph.get("failure_modes", []),
            monitoring_baselines=enriched_graph.get("monitoring_baselines", []),
        )
        behavioral_records.append(record)

    feedback_dir = str(output_dir / "feedback")
    files_written = export_feedback(behavioral_records, feedback_dir)

    tee(f"  PHASE 6: Feedback export .............. {len(files_written)} files written")

    # ========================================================================
    # ASSERTIONS
    # ========================================================================
    tee("")

    # E2E CHECK 1: Collection produced 5 run records
    check(1, "Collection produced 5 run records",
          len(run_records) == 5,
          f"got {len(run_records)}")

    # E2E CHECK 2: Run records have correct event counts
    run_001_record = next((r for r in run_records if r.get("run_id") == "run_001"), None)
    run_001_count = run_001_record.get("total_events", 0) if run_001_record else 0
    check(2, "Run records have correct event counts",
          run_001_count >= 20,
          f"run_001 has {run_001_count} events")

    # E2E CHECK 3: Run record for run_002 has error_summary.total_errors >= 1
    run_002_record = next((r for r in run_records if r.get("run_id") == "run_002"), None)
    run_002_errors = run_002_record.get("error_summary", {}).get("total_errors", 0) if run_002_record else 0
    check(3, "Run record for run_002 has errors",
          run_002_errors >= 1,
          f"total_errors={run_002_errors}")

    # E2E CHECK 4: Overlay produced enriched graph with edge activation rates
    has_edge_validation = enriched_graph is not None and "edge_validation" in enriched_graph
    check(4, "Overlay produced enriched graph with edge activation rates",
          has_edge_validation,
          f"edge_validation present: {has_edge_validation}")

    # E2E CHECK 5: Edge e1 (researcher->writer) has activation_rate > 0
    ev = enriched_graph.get("edge_validation", {}) if enriched_graph else {}
    per_edge = ev.get("per_edge", [])
    e1_entry = next((e for e in per_edge if e.get("edge_id") == "e1"), None)
    e1_rate = e1_entry.get("activation_rate", 0) if e1_entry else 0
    check(5, "Edge e1 (researcher->writer) has activation_rate > 0",
          e1_rate > 0,
          f"activation_rate={e1_rate}")

    # E2E CHECK 6: Edge e3 (researcher->web_search) has activation_rate > 0
    e3_entry = next((e for e in per_edge if e.get("edge_id") == "e3"), None)
    e3_rate = e3_entry.get("activation_rate", 0) if e3_entry else 0
    check(6, "Edge e3 (researcher->web_search) has activation_rate > 0",
          e3_rate > 0,
          f"activation_rate={e3_rate}")

    # E2E CHECK 7: At least 1 error propagation trace exists
    error_traces = enriched_graph.get("error_propagation", []) if enriched_graph else []
    check(7, "At least 1 error propagation trace exists",
          len(error_traces) >= 1,
          f"traces={len(error_traces)}")

    # E2E CHECK 8: Error trace has canonical schema
    trace_ok = False
    if error_traces:
        t = error_traces[0]
        trace_ok = (
            "error_source_node" in t
            and "structural_predicted_path" in t
            and "actual_observed_path" in t
            and isinstance(t.get("downstream_impact"), dict)
            and "stop_mechanism" in t
        )
    check(8, "Error trace has canonical schema",
          trace_ok,
          f"fields present: {trace_ok}")

    # E2E CHECK 9: Node activation classifies agents as always_active
    na = enriched_graph.get("node_activation", {}) if enriched_graph else {}
    always_active_ids = [n.get("node_id") for n in na.get("always_active", [])]
    agents_active = sum(1 for nid in always_active_ids if nid.startswith("agent_"))
    check(9, "Node activation classifies agents as always_active",
          agents_active >= 3,
          f"always_active agents: {agents_active}, ids: {always_active_ids}")

    # E2E CHECK 10: Monitoring baselines computed for at least 3 metrics
    mb = enriched_graph.get("monitoring_baselines", []) if enriched_graph else []
    check(10, "Monitoring baselines computed for at least 3 metrics",
          len(mb) >= 3,
          f"metrics={len(mb)}")

    # E2E CHECK 11: same_input_activation_consistency is non-null
    # Check determinism on any enriched node
    consistency_found = False
    if enriched_graph:
        for nid, ndata in enriched_graph.get("nodes", {}).items():
            det = ndata.get("behavioral", {}).get("determinism")
            if det and det.get("same_input_activation_consistency") is not None:
                consistency_found = True
                break
    check(11, "same_input_activation_consistency is non-null",
          consistency_found,
          f"found={consistency_found}")

    # E2E CHECK 12: Failure mode classification finds at least 1 STRAT- finding
    fm = enriched_graph.get("failure_modes", []) if enriched_graph else []
    strat_findings = [f for f in fm if f.get("finding_id", "").startswith("STRAT-")]
    check(12, "Failure mode classification finds at least 1 STRAT- finding",
          len(strat_findings) >= 1,
          f"STRAT findings: {len(strat_findings)}")

    # E2E CHECK 13: Feedback export produced failure_mode_catalog.json with >=1 entry
    fmc_path = Path(feedback_dir) / "failure_mode_catalog.json"
    fmc_entries = 0
    if fmc_path.exists():
        with open(fmc_path, "r", encoding="utf-8") as f:
            fmc = json.load(f)
        fmc_entries = len(fmc.get("catalog", fmc.get("findings", [])))
    check(13, "Feedback: failure_mode_catalog.json has >=1 catalog entry",
          fmc_entries >= 1,
          f"entries={fmc_entries}")

    # E2E CHECK 14: Feedback export produced monitoring_baselines.json with >=1 baseline
    mb_path = Path(feedback_dir) / "monitoring_baselines.json"
    mb_entries = 0
    if mb_path.exists():
        with open(mb_path, "r", encoding="utf-8") as f:
            mb_data = json.load(f)
        mb_entries = len(mb_data.get("baselines", []))
    check(14, "Feedback: monitoring_baselines.json has >=1 baseline",
          mb_entries >= 1,
          f"baselines={mb_entries}")

    # E2E CHECK 15: All feedback files have model_context with model_tier
    feedback_path = Path(feedback_dir)
    feedback_files = [
        "emergent_heuristics.json",
        "edge_confidence_weights.json",
        "failure_mode_catalog.json",
        "monitoring_baselines.json",
        "prediction_match_report.json",
    ]
    all_have_mc = True
    for fname in feedback_files:
        fp = feedback_path / fname
        if fp.exists():
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            mc = data.get("model_context", {})
            if not isinstance(mc, dict) or "model_tier" not in mc:
                all_have_mc = False
                break
        else:
            all_have_mc = False
            break
    check(15, "All feedback files have model_context with model_tier",
          all_have_mc)

    # E2E CHECK 16: Behavioral record passes validate_behavioral_record()
    br_valid = False
    br_errors: list[str] = []
    if behavioral_records:
        br_valid, br_errors = validate_behavioral_record(behavioral_records[0])
    check(16, "Behavioral record passes validate_behavioral_record()",
          br_valid,
          f"errors: {br_errors[:3]}" if br_errors else "")

    # E2E CHECK 17: Schema version is 'v6'
    schema_version = behavioral_records[0].get("schema_version") if behavioral_records else None
    check(17, "Schema version is 'v6'",
          schema_version == "v6",
          f"got: {schema_version}")

    # E2E CHECK 18: Feedback catalog uses FINDING_NAMES (human-readable)
    from stratum_lab.config import FINDING_NAMES
    catalog_names_ok = True
    if fmc_path.exists():
        with open(fmc_path, "r", encoding="utf-8") as f:
            fmc = json.load(f)
        for entry in fmc.get("catalog", fmc.get("findings", [])):
            fname = entry.get("finding_name", "")
            fid = entry.get("finding_id", "")
            # finding_name should NOT be the STRAT-xxx ID itself
            if fname.startswith("STRAT-") or len(fname) <= 5:
                catalog_names_ok = False
                break
    check(18, "Feedback catalog uses FINDING_NAMES (human-readable)",
          catalog_names_ok)

    # E2E CHECK 19: Feedback monitoring baselines use METRIC_TO_FINDING mapping
    from stratum_lab.config import METRIC_TO_FINDING
    baselines_mapping_ok = True
    if mb_path.exists():
        with open(mb_path, "r", encoding="utf-8") as f:
            mb_data = json.load(f)
        for baseline in mb_data.get("baselines", []):
            metric = baseline.get("metric", "")
            fid = baseline.get("finding_id", "")
            if metric in METRIC_TO_FINDING:
                expected_fid = METRIC_TO_FINDING[metric]
                if fid != expected_fid:
                    baselines_mapping_ok = False
                    break
    check(19, "Feedback monitoring baselines use METRIC_TO_FINDING mapping",
          baselines_mapping_ok)

    # E2E CHECK 20: Feedback monitoring baselines use SCANNER_METRIC_NAMES
    from stratum_lab.config import SCANNER_METRIC_NAMES
    scanner_names_ok = True
    if mb_path.exists():
        with open(mb_path, "r", encoding="utf-8") as f:
            mb_data = json.load(f)
        for baseline in mb_data.get("baselines", []):
            metric = baseline.get("metric", "")
            scanner_metric = baseline.get("scanner_metric", "")
            if metric in SCANNER_METRIC_NAMES:
                expected = SCANNER_METRIC_NAMES[metric]
                if scanner_metric != expected:
                    scanner_names_ok = False
                    break
            # Also check it doesn't have scanner_ prefix
            if scanner_metric.startswith("scanner_"):
                scanner_names_ok = False
                break
    check(20, "Scanner metric names are production-ready",
          scanner_names_ok)

    # E2E CHECK 21: Finding names match taxonomy categories
    from stratum_lab.config import FINDING_NAMES as FN
    taxonomy_ok = True
    taxonomy_violations: list[str] = []
    TAXONOMY_KEYWORDS = {
        "DC": {"decision", "delegation", "oversight", "checkpoint", "reversib", "guardrail", "circular"},
        "OC": {"objective", "incentive", "conflict", "shared", "competing", "contention", "overlap", "coordination"},
        "SI": {"signal", "integrity", "error", "propagat", "laundering", "schema", "validation", "sanitiz", "ordering", "semantic", "classification", "data"},
        "EA": {"authority", "scope", "escalat", "permission", "role", "boundar", "bottleneck", "concentration"},
        "AB": {"aggregate", "volume", "monitor", "population", "monoculture", "amplification", "unbounded"},
    }
    for fid, fname in FN.items():
        if fid.startswith("STRAT-XCOMP"):
            continue
        parts = fid.split("-")
        if len(parts) < 3:
            continue
        category = parts[1]  # DC, OC, SI, EA, AB
        keywords = TAXONOMY_KEYWORDS.get(category, set())
        name_lower = fname.lower()
        if not any(kw in name_lower for kw in keywords):
            taxonomy_ok = False
            taxonomy_violations.append(f"{fid}: '{fname}' missing {category} keyword")
    check(21, "Finding names match taxonomy categories",
          taxonomy_ok,
          f"violations: {taxonomy_violations[:3]}" if taxonomy_violations else "")

    # E2E CHECK 22: Error propagation trace has depth >= 2 (path length >= 3)
    deep_trace_found = False
    max_depth = 0
    if enriched_graph:
        for trace in enriched_graph.get("error_propagation", []):
            path_len = len(trace.get("actual_observed_path", []))
            depth = trace.get("downstream_impact", {}).get("cascade_depth", 0)
            max_depth = max(max_depth, depth)
            if path_len >= 3:
                deep_trace_found = True
    check(22, "Error propagation has cascade depth >= 2",
          deep_trace_found,
          f"max_depth={max_depth}")

    # ========================================================================
    # RESULTS
    # ========================================================================
    tee("")
    for num, label, passed, detail in check_results:
        status = "[PASS]" if passed else "[FAIL]"
        msg = f"  E2E CHECK {num:2d}: {status} {label}"
        if not passed and detail:
            msg += f" -- {detail}"
        tee(msg)

    tee("")
    tee("=" * 78)
    tee(f"  Results: {checks_passed} passed, {checks_failed} failed, {checks_passed + checks_failed} total")
    if checks_failed == 0:
        tee("  ALL CHECKS PASSED")
    else:
        tee(f"  {checks_failed} CHECK(S) FAILED")
    tee("=" * 78)

    # Save output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"\nOutput saved to {OUTPUT_FILE}")

    assert checks_failed == 0, f"{checks_failed} check(s) failed"


if __name__ == "__main__":
    test_end_to_end()
    sys.exit(0)
