"""Failure mode classification against finding taxonomy.

Maps reliability scanner findings to behavioral signals observed at runtime.
Goal: 10-15 qualitative examples per finding type, NOT statistical estimates.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from stratum_lab.config import FINDING_NAMES


FINDING_BEHAVIORAL_SIGNALS = {
    "STRAT-DC-001": {
        "name": FINDING_NAMES["STRAT-DC-001"],
        "signals": ["delegation_depth_>=3", "no_human_gate_activation", "decision_without_checkpoint"],
        "expected_failure": "Cascading errors through unsupervised delegation chain",
    },
    "STRAT-DC-002": {
        "name": FINDING_NAMES["STRAT-DC-002"],
        "signals": ["irreversible_action_no_gate", "write_after_error"],
        "expected_failure": "Irreversible action taken without checkpoint after error",
    },
    "STRAT-SI-001": {
        "name": FINDING_NAMES["STRAT-SI-001"],
        "signals": ["error_swallowed", "silent_error_handling", "downstream_continues_after_error"],
        "expected_failure": "Error silently swallowed, downstream agents process corrupted data",
    },
    "STRAT-SI-004": {
        "name": FINDING_NAMES["STRAT-SI-004"],
        "signals": ["schema_mismatch_error", "type_error_at_boundary"],
        "expected_failure": "Schema mismatch at agent boundary causes silent data corruption",
    },
    "STRAT-EA-001": {
        "name": FINDING_NAMES["STRAT-EA-001"],
        "signals": ["deep_delegation_chain", "capability_access_beyond_direct"],
        "expected_failure": "Agent gains transitive access to capabilities not directly assigned",
    },
    "STRAT-OC-002": {
        "name": FINDING_NAMES["STRAT-OC-002"],
        "signals": ["concurrent_state_writes", "state_access_conflict"],
        "expected_failure": "Multiple agents write to shared state without coordination",
    },
    "STRAT-AB-001": {
        "name": FINDING_NAMES["STRAT-AB-001"],
        "signals": ["high_token_usage_no_limit", "unbounded_iterations"],
        "expected_failure": "Unbounded resource consumption without monitoring",
    },
}


def classify_failure_modes(
    findings: List[Dict],
    events: List[Dict],
    structural_graph: Dict,
    error_traces: List[Dict],
) -> List[Dict]:
    """Classify failure modes from runtime behavioral evidence."""
    results = []

    finding_ids = {f.get("finding_id", ""): f for f in findings}

    for finding_id, signals_def in FINDING_BEHAVIORAL_SIGNALS.items():
        if finding_id not in finding_ids and not _events_suggest_finding(finding_id, events):
            continue

        occurrences = _detect_signal_occurrences(
            finding_id, signals_def, events, structural_graph, error_traces
        )

        manifestation_observed = len(occurrences) > 0

        downstream_impact = any(
            occ.get("downstream_impact", False) for occ in occurrences
        )

        results.append({
            "finding_id": finding_id,
            "finding_name": signals_def["name"],
            "manifestation_observed": manifestation_observed,
            "failure_type": signals_def["expected_failure"],
            "failure_description": _describe_manifestation(finding_id, occurrences),
            "occurrences": len(occurrences),
            "occurrence_details": occurrences[:15],
            "downstream_impact_observed": downstream_impact,
        })

    return results


def _events_suggest_finding(finding_id: str, events: List[Dict]) -> bool:
    """Check if events suggest a finding even if not in the finding list."""
    if finding_id == "STRAT-SI-001":
        return any(
            e.get("payload", {}).get("error_handling") in ("caught_silent", "caught_default", "fail_silent")
            for e in events if e.get("event_type", "").startswith("error.")
        )
    if finding_id == "STRAT-DC-001":
        delegation_count = sum(
            1 for e in events if e.get("event_type") == "delegation.initiated"
        )
        return delegation_count >= 3
    return False


def _detect_signal_occurrences(
    finding_id: str,
    signals_def: Dict,
    events: List[Dict],
    structural_graph: Dict,
    error_traces: List[Dict],
) -> List[Dict]:
    """Detect occurrences of behavioral signals for a finding."""
    occurrences = []

    if finding_id == "STRAT-SI-001":
        for trace in error_traces:
            if trace.get("swallowed") or trace.get("stop_mechanism") in ("swallowed_at_source", "error_handler"):
                occurrences.append({
                    "signal": "error_swallowed",
                    "node": trace.get("error_source_node", ""),
                    "description": f"Error at {trace.get('error_source_node', '')} was {trace.get('stop_mechanism', 'swallowed')}",
                    "downstream_impact": trace.get("downstream_impact", {}).get("downstream_errors", 0) > 0,
                })

        for e in events:
            if e.get("event_type", "").startswith("error."):
                handling = e.get("payload", {}).get("error_handling", "")
                if handling in ("caught_silent", "caught_default", "fail_silent"):
                    node = e.get("source_node", {}).get("node_name", "unknown")
                    occurrences.append({
                        "signal": "silent_error_handling",
                        "node": node,
                        "description": f"Error at {node} handled with '{handling}'",
                        "downstream_impact": False,
                    })

    elif finding_id == "STRAT-DC-001":
        delegations = [e for e in events if e.get("event_type") == "delegation.initiated"]
        if len(delegations) >= 3:
            chain = []
            for d in delegations:
                src = d.get("source_node", {}).get("node_name", "")
                tgt = d.get("target_node", {}).get("node_name", "")
                chain.append(f"{src}->{tgt}")
            occurrences.append({
                "signal": "delegation_depth_>=3",
                "node": "chain",
                "description": f"Delegation chain of depth {len(delegations)}: {', '.join(chain[:5])}",
                "downstream_impact": any(t.get("downstream_impact", {}).get("downstream_errors", 0) > 0 for t in error_traces),
            })

        human_gates = [
            e for e in events
            if e.get("event_type") == "guardrail.triggered"
            and "human" in str(e.get("payload", {})).lower()
        ]
        if not human_gates and delegations:
            occurrences.append({
                "signal": "no_human_gate_activation",
                "node": "system",
                "description": "No human approval gate activated during delegation chain",
                "downstream_impact": False,
            })

    elif finding_id == "STRAT-OC-002":
        state_writes = [
            e for e in events
            if e.get("event_type") in ("data.write", "state.access")
            and e.get("payload", {}).get("access_type", "") in ("write", "read_write", "")
        ]
        by_target: Dict[str, List] = defaultdict(list)
        for e in state_writes:
            target = (e.get("target_node", {}).get("node_name", "")
                     or e.get("payload", {}).get("state_key", ""))
            if target:
                by_target[target].append(e)
        for target, writes in by_target.items():
            writers = set()
            for w in writes:
                src = w.get("source_node", {}).get("node_name", "")
                if not src:
                    src = w.get("payload", {}).get("node_id", "")
                if src:
                    writers.add(src)
            if len(writers) >= 2:
                occurrences.append({
                    "signal": "concurrent_state_writes",
                    "node": target,
                    "description": f"Multiple agents ({', '.join(writers)}) write to '{target}'",
                    "downstream_impact": True,
                })

    elif finding_id in ("STRAT-SI-004", "STRAT-EA-001", "STRAT-DC-002",
                        "STRAT-DC-003", "STRAT-AB-001"):
        for trace in error_traces:
            if trace.get("error_type") == "schema_mismatch" and finding_id == "STRAT-SI-004":
                occurrences.append({
                    "signal": "schema_mismatch_error",
                    "node": trace.get("error_source_node", ""),
                    "description": f"Schema mismatch at {trace.get('error_source_node', '')}",
                    "downstream_impact": trace.get("downstream_impact", {}).get("downstream_errors", 0) > 0,
                })

    return occurrences


def _describe_manifestation(finding_id: str, occurrences: List[Dict]) -> str:
    """Generate a human-readable description of manifestation."""
    if not occurrences:
        return "No behavioral manifestation observed"

    signals = set(o.get("signal", "") for o in occurrences)
    return f"Observed {len(occurrences)} instances across signals: {', '.join(signals)}"
