#!/usr/bin/env python3
"""Phase 4b: Per-repo enrichment computation.

Reads behavioral record + data topology scan + event streams.
Writes enrichments.json to results_dir and patches behavioral_record.json.

Usage:
    python compute_enrichments.py <results_dir>
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from glob import glob
from pathlib import Path

# Ensure stratum_lab is importable
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from stratum_lab.privacy import compute_privacy_topology
from stratum_lab.permissions import compute_permission_blast_radius
from stratum_lab.cost_risk import compute_cost_risk
from stratum_lab.audit_readiness import compute_audit_readiness

logger = logging.getLogger(__name__)


def load_events(results_dir: str) -> dict:
    """Load events_run_N.jsonl files from results_dir/raw_events/ or results_dir.

    Returns {run_name: [event_dict, ...]}.
    """
    events_by_run: dict[str, list[dict]] = {}

    # Try raw_events/ subdirectory first, then results_dir itself
    search_dirs = [
        os.path.join(results_dir, "raw_events"),
        results_dir,
    ]

    for search_dir in search_dirs:
        pattern = os.path.join(search_dir, "events_run_*.jsonl")
        for fpath in sorted(glob(pattern)):
            run_name = Path(fpath).stem
            events: list[dict] = []
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            try:
                                events.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
            except OSError:
                continue
            if events:
                events_by_run[run_name] = events

        if events_by_run:
            break

    return events_by_run


def load_behavioral_record(results_dir: str) -> dict:
    """Load behavioral_record.json from results_dir. Return parsed dict."""
    path = os.path.join(results_dir, "behavioral_record.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def load_data_topology(results_dir: str) -> dict:
    """Load data_topology.json from results_dir. Return parsed dict or empty dict."""
    path = os.path.join(results_dir, "data_topology.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def extract_runtime_tool_calls(events_by_run: dict) -> dict:
    """From all tool.call_end events across runs, build {node_id: [tool_names]}."""
    result: dict[str, list[str]] = {}
    for events in events_by_run.values():
        for evt in events:
            if evt.get("event_type") in ("tool.call_end", "tool.invoked"):
                nid = evt.get("node_id", "")
                tool = evt.get("tool_name", "")
                if nid and tool:
                    result.setdefault(nid, []).append(tool)
    return result


def compute_all_enrichments(results_dir: str) -> dict:
    """Orchestrate all enrichment computations for one repo."""
    behavioral_record = load_behavioral_record(results_dir)
    data_topology = load_data_topology(results_dir)
    events_by_run = load_events(results_dir)

    # Extract nodes and edges from behavioral record topology
    topology = behavioral_record.get("edge_validation", {})
    nodes_raw = []
    edges_raw = []

    # Try multiple sources for topology data
    if isinstance(topology, dict):
        for edge_key, edge_data in topology.items():
            if isinstance(edge_data, dict):
                src = edge_data.get("source", "")
                tgt = edge_data.get("target", "")
                if src and tgt:
                    edges_raw.append({"edge_id": edge_key, "source": src, "target": tgt})

    # Also try emergent_edges
    emergent = behavioral_record.get("emergent_edges", [])
    if isinstance(emergent, list):
        for edge in emergent:
            if isinstance(edge, dict):
                edges_raw.append(edge)

    # Extract unique node IDs from edges and node_activation
    node_ids: set[str] = set()
    for edge in edges_raw:
        node_ids.add(edge.get("source", ""))
        node_ids.add(edge.get("target", ""))
    node_activation = behavioral_record.get("node_activation", {})
    if isinstance(node_activation, dict):
        node_ids.update(node_activation.keys())
    node_ids.discard("")

    nodes_raw = [{"node_id": nid} for nid in sorted(node_ids)]

    # Extract tool_registrations from data_topology
    tool_registrations = data_topology.get("tool_registrations", {})

    # Extract state.access events
    state_access_events: list[dict] = []
    for events in events_by_run.values():
        for evt in events:
            if evt.get("event_type") == "state.access":
                state_access_events.append(evt)

    # Extract runtime tool calls
    runtime_tool_calls = extract_runtime_tool_calls(events_by_run)

    # Extract findings from behavioral record
    findings = behavioral_record.get("failure_modes", [])
    if not isinstance(findings, list):
        findings = []

    # Run all 4 enrichments with error isolation
    enrichments: dict = {}
    completed = 0
    failed = 0

    for name, func, args in [
        ("privacy_topology", compute_privacy_topology,
         (nodes_raw, edges_raw, tool_registrations, state_access_events)),
        ("permission_blast_radius", compute_permission_blast_radius,
         (nodes_raw, edges_raw, tool_registrations, runtime_tool_calls)),
        ("cost_risk", compute_cost_risk,
         (list(events_by_run.values()), events_by_run)),
        ("audit_readiness", compute_audit_readiness,
         (events_by_run, nodes_raw, edges_raw, tool_registrations, findings)),
    ]:
        try:
            enrichments[name] = func(*args)
            completed += 1
        except Exception as exc:
            logger.warning("Enrichment %s failed: %s", name, exc)
            enrichments[name] = {"error": str(exc)}
            failed += 1

    enrichments["enrichment_timestamp"] = datetime.now(timezone.utc).isoformat()
    enrichments["enrichments_completed"] = completed
    enrichments["enrichments_failed"] = failed

    return enrichments


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python compute_enrichments.py <results_dir>", file=sys.stderr)
        sys.exit(1)

    results_dir = sys.argv[1]
    if not os.path.isdir(results_dir):
        print(f"Error: {results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    enrichments = compute_all_enrichments(results_dir)

    # Write enrichments.json
    output_path = os.path.join(results_dir, "enrichments.json")
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(enrichments, fh, indent=2)

    # Patch behavioral_record.json if it exists
    record_path = os.path.join(results_dir, "behavioral_record.json")
    if os.path.exists(record_path):
        try:
            with open(record_path, "r", encoding="utf-8") as fh:
                record = json.load(fh)
            record["research_enrichments"] = enrichments
            with open(record_path, "w", encoding="utf-8") as fh:
                json.dump(record, fh, indent=2)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to patch behavioral record: %s", exc)

    print(f"Enrichments written to {output_path}", file=sys.stderr)
    print(f"Completed: {enrichments['enrichments_completed']}/4, "
          f"Failed: {enrichments['enrichments_failed']}/4", file=sys.stderr)


if __name__ == "__main__":
    main()
