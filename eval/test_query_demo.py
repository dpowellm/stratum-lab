"""Validation check 9: Per-repo behavioral record matches stratum-graph schema.

Builds a complete behavioral record from synthetic data using
build_behavioral_record(), validates it against the stratum-graph v6
schema via validate_behavioral_record(), and verifies every top-level
and nested field.

Run as a standalone script:
    cd stratum-lab
    python eval/test_query_demo.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup -- stratum_lab is NOT an installed package
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from stratum_lab.output.behavioral_record import (
    build_behavioral_record,
    validate_behavioral_record,
)


# ---------------------------------------------------------------------------
# Output setup
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "query-demo.txt"

console = Console(record=True, width=100)

pass_count = 0
fail_count = 0


def check(label: str, condition: bool, detail: str = "") -> bool:
    """Log a pass/fail check and update counters."""
    global pass_count, fail_count
    if condition:
        pass_count += 1
        console.print(f"  [green][+][/green] {label}")
    else:
        fail_count += 1
        msg = f"  [red][X][/red] {label}"
        if detail:
            msg += f"  -- {detail}"
        console.print(msg)
    return condition


# =========================================================================
# 1. Build synthetic data for the behavioral record
# =========================================================================

EXECUTION_METADATA = {
    "total_runs": 3,
    "successful_runs": 2,
    "failed_runs": 1,
    "framework": "crewai",
    "stratum_lab_version": "0.1.0",
    "execution_timestamp": "2026-02-15T12:00:00Z",
    "docker_image": "stratum-runner:latest",
    "timeout_seconds": 300,
    "model_provider": "openai",
    "model_name": "gpt-4",
}

EDGE_VALIDATION = {
    "dead_edges": [
        {
            "edge_id": "e10",
            "source": "cap_web_search_tool",
            "target": "ext_web_api",
            "edge_type": "calls",
            "runs_observed": 3,
            "possible_reasons": ["tool never invoked web API directly"],
        },
    ],
    "activation_rates": {
        "e1": {"traversal_count": 6, "activation_rate": 1.0, "never_activated": False},
        "e2": {"traversal_count": 5, "activation_rate": 0.833, "never_activated": False},
        "e3": {"traversal_count": 3, "activation_rate": 0.5, "never_activated": False},
        "e10": {"traversal_count": 0, "activation_rate": 0.0, "never_activated": True},
    },
}

EMERGENT_EDGES = [
    {
        "edge_id": "emergent_001",
        "source_node_id": "agent_reviewer",
        "target_node_id": "cap_web_search_tool",
        "edge_type": "uses",
        "traversal_count": 2,
        "activation_rate": 0.333,
        "discovery_type": "runtime_interaction",
        "detection_heuristic": "event_pair_matching",
        "significance": "medium",
        "trigger_condition": "Reviewer needed additional data on run 3",
    },
    {
        "edge_id": "emergent_002",
        "source_node_id": "agent_researcher",
        "target_node_id": "agent_reviewer",
        "edge_type": "delegates_to",
        "traversal_count": 1,
        "activation_rate": 0.167,
        "discovery_type": "runtime_interaction",
        "detection_heuristic": "event_pair_matching",
        "significance": "low",
        "trigger_condition": "Researcher directly delegated to Reviewer bypassing Writer",
    },
]

NODE_ACTIVATION = {
    "always_active": [
        "agent_researcher",
        "agent_writer",
        "cap_llm_call",
        "ds_shared_memory",
    ],
    "conditional": [
        "agent_reviewer",
        "cap_web_search_tool",
        "guard_output",
    ],
    "never_active": [
        "cap_summarize_tool",
    ],
}

ERROR_PROPAGATION = [
    {
        "trace_id": "err_trace_001",
        "error_source_node": "agent_writer",
        "error_type": "ValueError",
        "structural_predicted_path": ["agent_writer", "ds_shared_memory", "agent_reviewer"],
        "actual_observed_path": ["agent_writer", "ds_shared_memory", "agent_reviewer"],
        "propagation_stopped_by": "",
        "stop_mechanism": "propagated_through",
        "downstream_impact": {
            "nodes_affected": 2,
            "downstream_errors": 1,
            "downstream_tasks_failed": 0,
            "cascade_depth": 2,
        },
        "structural_prediction_match": True,
        "run_id": "run_002",
    },
    {
        "trace_id": "err_trace_002",
        "error_source_node": "agent_researcher",
        "error_type": "TimeoutError",
        "structural_predicted_path": ["agent_researcher"],
        "actual_observed_path": ["agent_researcher"],
        "propagation_stopped_by": "agent_researcher",
        "stop_mechanism": "retry",
        "downstream_impact": {
            "nodes_affected": 0,
            "downstream_errors": 0,
            "downstream_tasks_failed": 0,
            "cascade_depth": 0,
        },
        "structural_prediction_match": True,
        "run_id": "run_001",
    },
]

FAILURE_MODES = [
    {
        "finding_id": "STRAT-OC-002",
        "finding_name": "Shared state contention",
        "manifestation_observed": True,
        "precondition_id": "shared_state_no_arbitration",
        "description": "Multiple agents write to SharedMemory without locking",
        "severity": "high",
        "evidence_runs": ["run_001", "run_002"],
        "evidence_events": ["evt_000042", "evt_000078"],
    },
    {
        "finding_id": "STRAT-DC-002",
        "finding_name": "Missing delegation timeout",
        "manifestation_observed": False,
        "precondition_id": "no_timeout_on_delegation",
        "description": "Delegation chain completed within timeout in all runs",
        "severity": "none",
        "evidence_runs": [],
        "evidence_events": [],
    },
    {
        "finding_id": "STRAT-EA-001",
        "finding_name": "Unhandled tool failure",
        "manifestation_observed": True,
        "precondition_id": "unhandled_tool_failure",
        "description": "WebSearch tool failure not caught by Researcher agent",
        "severity": "medium",
        "evidence_runs": ["run_002"],
        "evidence_events": ["evt_000105"],
    },
]

MONITORING_BASELINES = [
    {
        "metric": "agent_task_latency_p95_ms",
        "observed_baseline": 2450.0,
        "threshold": 5000.0,
        "unit": "milliseconds",
        "node_scope": "agent_researcher",
    },
    {
        "metric": "error_propagation_rate",
        "observed_baseline": 0.33,
        "threshold": 0.5,
        "unit": "ratio",
        "node_scope": "global",
    },
    {
        "metric": "delegation_chain_depth_max",
        "observed_baseline": 3,
        "threshold": 5,
        "unit": "count",
        "node_scope": "global",
    },
    {
        "metric": "tool_call_failure_rate",
        "observed_baseline": 0.08,
        "threshold": 0.2,
        "unit": "ratio",
        "node_scope": "cap_web_search_tool",
    },
]


# =========================================================================
# 2. Build the behavioral record
# =========================================================================

def main() -> None:
    global pass_count, fail_count

    console.rule("[bold]CHECK 9: Behavioral Record Matches stratum-graph Schema[/bold]")
    console.print()

    record = build_behavioral_record(
        repo_full_name="test-org/crewai-research-crew",
        execution_metadata=EXECUTION_METADATA,
        edge_validation=EDGE_VALIDATION,
        emergent_edges=EMERGENT_EDGES,
        node_activation=NODE_ACTIVATION,
        error_propagation=ERROR_PROPAGATION,
        failure_modes=FAILURE_MODES,
        monitoring_baselines=MONITORING_BASELINES,
    )

    # =====================================================================
    # 3. Validate via validate_behavioral_record()
    # =====================================================================

    console.print("[bold]3. validate_behavioral_record()[/bold]")
    console.print()

    is_valid, missing = validate_behavioral_record(record)
    check("validate_behavioral_record returns True", is_valid,
          f"missing fields: {missing}" if missing else "")
    check("No missing fields", len(missing) == 0,
          f"missing: {missing}" if missing else "")
    console.print()

    # =====================================================================
    # 4. Top-level key verification
    # =====================================================================

    console.print("[bold]4. Top-Level Key Verification[/bold]")
    console.print()

    EXPECTED_TOP_KEYS = {
        "repo_full_name",
        "schema_version",
        "execution_metadata",
        "edge_validation",
        "emergent_edges",
        "node_activation",
        "error_propagation",
        "failure_modes",
        "monitoring_baselines",
    }

    actual_keys = set(record.keys())
    check(
        f"Record has exactly {len(EXPECTED_TOP_KEYS)} top-level keys",
        actual_keys == EXPECTED_TOP_KEYS,
        f"extra={actual_keys - EXPECTED_TOP_KEYS}, missing={EXPECTED_TOP_KEYS - actual_keys}",
    )

    for key in sorted(EXPECTED_TOP_KEYS):
        check(f"Key present: {key}", key in record)

    check(
        "repo_full_name is a string",
        isinstance(record["repo_full_name"], str),
    )
    check(
        "schema_version == 'v6'",
        record["schema_version"] == "v6",
        f"got '{record.get('schema_version')}'",
    )
    console.print()

    # =====================================================================
    # 5. Nested structure: edge_validation
    # =====================================================================

    console.print("[bold]5. edge_validation structure[/bold]")
    console.print()

    ev = record["edge_validation"]
    check("edge_validation is a dict", isinstance(ev, dict))
    check("edge_validation has 'dead_edges' key", "dead_edges" in ev)
    check("edge_validation has 'activation_rates' key", "activation_rates" in ev)
    check("dead_edges is a list", isinstance(ev.get("dead_edges"), list))
    check("activation_rates is a dict", isinstance(ev.get("activation_rates"), dict))

    if ev["dead_edges"]:
        de = ev["dead_edges"][0]
        check("dead_edges[0] has 'edge_id'", "edge_id" in de)
        check("dead_edges[0] has 'source'", "source" in de)
        check("dead_edges[0] has 'target'", "target" in de)

    if ev["activation_rates"]:
        first_rate_key = next(iter(ev["activation_rates"]))
        ar = ev["activation_rates"][first_rate_key]
        check(
            f"activation_rates['{first_rate_key}'] has traversal_count",
            "traversal_count" in ar,
        )
        check(
            f"activation_rates['{first_rate_key}'] has activation_rate",
            "activation_rate" in ar,
        )
    console.print()

    # =====================================================================
    # 6. Nested structure: emergent_edges
    # =====================================================================

    console.print("[bold]6. emergent_edges structure[/bold]")
    console.print()

    ee = record["emergent_edges"]
    check("emergent_edges is a list", isinstance(ee, list))
    check(f"emergent_edges has {len(ee)} entries", len(ee) > 0)

    for i, entry in enumerate(ee):
        check(f"emergent_edges[{i}] has 'discovery_type'", "discovery_type" in entry)
        check(f"emergent_edges[{i}] has 'detection_heuristic'", "detection_heuristic" in entry)
        check(
            f"emergent_edges[{i}] discovery_type is a string",
            isinstance(entry.get("discovery_type"), str),
        )
        check(
            f"emergent_edges[{i}] detection_heuristic is a string",
            isinstance(entry.get("detection_heuristic"), str),
        )
    console.print()

    # =====================================================================
    # 7. Nested structure: node_activation
    # =====================================================================

    console.print("[bold]7. node_activation structure[/bold]")
    console.print()

    na = record["node_activation"]
    check("node_activation is a dict", isinstance(na, dict))
    check("node_activation has 'always_active' list", isinstance(na.get("always_active"), list))
    check("node_activation has 'conditional' list", isinstance(na.get("conditional"), list))
    check("node_activation has 'never_active' list", isinstance(na.get("never_active"), list))
    check(
        f"always_active has {len(na['always_active'])} entries",
        len(na["always_active"]) > 0,
    )
    check(
        f"conditional has {len(na['conditional'])} entries",
        len(na["conditional"]) > 0,
    )
    check(
        f"never_active has {len(na['never_active'])} entries",
        len(na["never_active"]) > 0,
    )

    # All entries should be strings (node IDs)
    all_na_entries = na["always_active"] + na["conditional"] + na["never_active"]
    check(
        "All node_activation entries are strings",
        all(isinstance(x, str) for x in all_na_entries),
    )
    console.print()

    # =====================================================================
    # 8. Nested structure: error_propagation
    # =====================================================================

    console.print("[bold]8. error_propagation structure[/bold]")
    console.print()

    ep = record["error_propagation"]
    check("error_propagation is a list", isinstance(ep, list))
    check(f"error_propagation has {len(ep)} traces", len(ep) > 0)

    for i, trace in enumerate(ep):
        check(f"error_propagation[{i}] is a dict", isinstance(trace, dict))
        check(f"error_propagation[{i}] has 'error_source_node'", "error_source_node" in trace)
        check(f"error_propagation[{i}] has 'structural_predicted_path'", "structural_predicted_path" in trace)
        check(
            f"error_propagation[{i}] structural_predicted_path is a list",
            isinstance(trace.get("structural_predicted_path"), list),
        )
    console.print()

    # =====================================================================
    # CHECK 26: Rich error propagation fields
    # =====================================================================

    console.print("[bold]CHECK 26. Rich error propagation fields[/bold]")
    console.print()

    # Check that at least one trace has the rich schema fields (strict — no fallbacks)
    has_rich_ep = False
    for trace in ep:
        has_spp = isinstance(trace.get("structural_predicted_path"), list)
        has_aop = isinstance(trace.get("actual_observed_path"), list)
        has_sm = isinstance(trace.get("stop_mechanism"), str)
        has_di = isinstance(trace.get("downstream_impact"), dict) and "nodes_affected" in trace["downstream_impact"]
        if has_spp and has_aop and has_sm and has_di:
            has_rich_ep = True
            break

    check("At least one error_propagation trace has rich schema fields", has_rich_ep,
          "Expected structural_predicted_path, actual_observed_path, stop_mechanism, downstream_impact")
    console.print()

    # =====================================================================
    # 9. Nested structure: failure_modes
    # =====================================================================

    console.print("[bold]9. failure_modes structure[/bold]")
    console.print()

    fm = record["failure_modes"]
    check("failure_modes is a list", isinstance(fm, list))
    check(f"failure_modes has {len(fm)} entries", len(fm) > 0)

    for i, finding in enumerate(fm):
        check(f"failure_modes[{i}] has 'finding_id'", "finding_id" in finding)
        check(f"failure_modes[{i}] has 'manifestation_observed'", "manifestation_observed" in finding)
        check(
            f"failure_modes[{i}] finding_id is a string",
            isinstance(finding.get("finding_id"), str),
        )
        check(
            f"failure_modes[{i}] manifestation_observed is a bool",
            isinstance(finding.get("manifestation_observed"), bool),
        )
    console.print()

    # =====================================================================
    # 10. Nested structure: monitoring_baselines
    # =====================================================================

    console.print("[bold]10. monitoring_baselines structure[/bold]")
    console.print()

    mb = record["monitoring_baselines"]
    check("monitoring_baselines is a list", isinstance(mb, list))
    check(f"monitoring_baselines has {len(mb)} entries", len(mb) > 0)

    for i, baseline in enumerate(mb):
        check(f"monitoring_baselines[{i}] has 'metric'", "metric" in baseline)
        check(f"monitoring_baselines[{i}] has 'observed_baseline'", "observed_baseline" in baseline)
        check(f"monitoring_baselines[{i}] has 'threshold'", "threshold" in baseline)
        check(
            f"monitoring_baselines[{i}] metric is a string",
            isinstance(baseline.get("metric"), str),
        )
        check(
            f"monitoring_baselines[{i}] observed_baseline is numeric",
            isinstance(baseline.get("observed_baseline"), (int, float)),
        )
        check(
            f"monitoring_baselines[{i}] threshold is numeric",
            isinstance(baseline.get("threshold"), (int, float)),
        )
    console.print()

    # =====================================================================
    # 11. JSON serialization
    # =====================================================================

    console.print("[bold]11. JSON Serialization[/bold]")
    console.print()

    try:
        serialized = json.dumps(record, indent=2, default=str)
        check("json.dumps(record) succeeds", True)
        check(
            f"Serialized JSON is {len(serialized)} bytes",
            len(serialized) > 0,
        )

        # Round-trip: deserialize and re-validate
        deserialized = json.loads(serialized)
        rt_valid, rt_missing = validate_behavioral_record(deserialized)
        check("Round-trip deserialized record is still valid", rt_valid,
              f"missing after round-trip: {rt_missing}" if rt_missing else "")
    except (TypeError, ValueError) as exc:
        check(f"json.dumps(record) succeeds", False, str(exc))

    console.print()

    # =====================================================================
    # 12. Print the complete behavioral record structure
    # =====================================================================

    console.print("[bold]12. Complete Behavioral Record[/bold]")
    console.print()

    # Top-level summary table
    summary_table = Table(title="Behavioral Record Summary", show_lines=True)
    summary_table.add_column("Field", style="cyan", no_wrap=True)
    summary_table.add_column("Type", style="magenta")
    summary_table.add_column("Value / Summary", style="white")

    summary_table.add_row("repo_full_name", "str", record["repo_full_name"])
    summary_table.add_row("schema_version", "str", record["schema_version"])
    summary_table.add_row(
        "execution_metadata", "dict",
        f"{len(record['execution_metadata'])} keys: {', '.join(record['execution_metadata'].keys())}",
    )
    summary_table.add_row(
        "edge_validation", "dict",
        f"dead_edges: {len(ev['dead_edges'])}, activation_rates: {len(ev['activation_rates'])} edges",
    )
    summary_table.add_row(
        "emergent_edges", "list",
        f"{len(ee)} emergent edges discovered",
    )
    summary_table.add_row(
        "node_activation", "dict",
        f"always={len(na['always_active'])}, conditional={len(na['conditional'])}, never={len(na['never_active'])}",
    )
    summary_table.add_row(
        "error_propagation", "list",
        f"{len(ep)} propagation traces",
    )
    summary_table.add_row(
        "failure_modes", "list",
        f"{len(fm)} findings ({sum(1 for f in fm if f['manifestation_observed'])} manifested)",
    )
    summary_table.add_row(
        "monitoring_baselines", "list",
        f"{len(mb)} baselines defined",
    )

    console.print(summary_table)
    console.print()

    # Print each section in detail
    console.print(Panel(
        json.dumps(record["execution_metadata"], indent=2),
        title="execution_metadata",
        border_style="blue",
    ))
    console.print()

    console.print(Panel(
        json.dumps(record["edge_validation"], indent=2),
        title="edge_validation",
        border_style="blue",
    ))
    console.print()

    console.print(Panel(
        json.dumps(record["emergent_edges"], indent=2),
        title="emergent_edges",
        border_style="blue",
    ))
    console.print()

    console.print(Panel(
        json.dumps(record["node_activation"], indent=2),
        title="node_activation",
        border_style="blue",
    ))
    console.print()

    console.print(Panel(
        json.dumps(record["error_propagation"], indent=2),
        title="error_propagation",
        border_style="blue",
    ))
    console.print()

    console.print(Panel(
        json.dumps(record["failure_modes"], indent=2),
        title="failure_modes",
        border_style="blue",
    ))
    console.print()

    console.print(Panel(
        json.dumps(record["monitoring_baselines"], indent=2),
        title="monitoring_baselines",
        border_style="blue",
    ))
    console.print()

    # =====================================================================
    # Final results
    # =====================================================================

    console.rule("[bold]Results[/bold]")
    console.print()
    console.print(f"  Passed: [green]{pass_count}[/green]")
    console.print(f"  Failed: [red]{fail_count}[/red]")
    console.print(f"  Total:  {pass_count + fail_count}")
    console.print()

    if fail_count == 0:
        console.print("[bold green]  ALL CHECKS PASSED[/bold green]")
    else:
        console.print(f"[bold red]  {fail_count} CHECK(S) FAILED[/bold red]")

    console.print()

    # =====================================================================
    # Write output file
    # =====================================================================

    output_text = console.export_text()
    OUTPUT_FILE.write_text(output_text, encoding="utf-8")
    console.print(f"Output saved to: {OUTPUT_FILE}")

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
