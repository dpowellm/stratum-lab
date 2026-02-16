"""Monitoring baseline extraction from behavioral observations.

For each observable finding, compute metrics that can serve as
monitoring thresholds in production.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List

from stratum_lab.config import METRIC_TO_FINDING, SCANNER_METRIC_NAMES


MONITORING_METRICS = {
    "STRAT-DC-001": {
        "metric": "max_delegation_depth",
        "unit": "count",
        "threshold_method": "mean_plus_2sd",
    },
    "STRAT-SI-001": {
        "metric": "error_swallow_rate",
        "unit": "ratio",
        "threshold_method": "absolute_cap_1.0",
    },
    "STRAT-AB-001": {
        "metric": "total_llm_calls_per_run",
        "unit": "count",
        "threshold_method": "mean_plus_2sd",
    },
    "STRAT-OC-002": {
        "metric": "concurrent_state_write_rate",
        "unit": "ratio",
        "threshold_method": "absolute_cap_1.0",
    },
    "STRAT-SI-004": {
        "metric": "schema_mismatch_rate",
        "unit": "ratio",
        "threshold_method": "absolute_cap_1.0",
    },
    "STRAT-EA-001": {
        "metric": "tool_call_failure_rate",
        "unit": "ratio",
        "threshold_method": "mean_plus_2sd",
    },
    "STRAT-DC-002": {
        "metric": "delegation_latency_p95_ms",
        "unit": "milliseconds",
        "threshold_method": "mean_plus_2sd",
    },
}


def extract_monitoring_baselines(
    findings: List[Dict],
    events: List[Dict],
) -> List[Dict]:
    """Extract monitoring baselines from behavioral observations."""
    results = []

    # Group events by run
    by_run: Dict[str, List[Dict]] = defaultdict(list)
    for e in events:
        rid = e.get("run_id", "")
        by_run.setdefault(rid, []).append(e)

    run_count = sum(1 for rid in by_run if rid)

    for finding_id, metric_def in MONITORING_METRICS.items():
        metric_name = metric_def["metric"]
        values = []

        for run_id, run_events in by_run.items():
            if not run_id:
                continue
            value = _compute_metric(metric_name, run_events, run_count)
            if value is not None:
                values.append(value)

        if not values:
            continue

        mean_val = sum(values) / len(values)
        stddev = math.sqrt(sum((v - mean_val) ** 2 for v in values) / max(len(values), 1))

        # Apply threshold method
        if metric_def["threshold_method"] == "absolute_cap_1.0":
            suggested_threshold = min(mean_val + 2 * stddev, 1.0)
        else:
            suggested_threshold = mean_val + 2 * stddev

        confidence = "high" if len(values) >= 10 else "medium" if len(values) >= 3 else "low"

        scanner_metric = SCANNER_METRIC_NAMES.get(metric_name, metric_name)

        results.append({
            "finding_id": finding_id,
            "metric": metric_name,
            "scanner_metric": scanner_metric,
            "observed_baseline": round(mean_val, 4),
            "observed_stddev": round(stddev, 4),
            "suggested_threshold": round(suggested_threshold, 4),
            "confidence": confidence,
            "sample_size": len(values),
            "unit": metric_def["unit"],
            "model_tier_note": "Baseline from weak model. Threshold ratios transfer; absolute values may differ.",
        })

    return results


def _compute_metric(metric_name: str, run_events: List[Dict], run_count: int) -> float | None:
    """Dispatch to the appropriate metric computation."""
    if metric_name == "max_delegation_depth":
        return _compute_max_delegation_depth(run_events)
    elif metric_name == "error_swallow_rate":
        return _compute_error_swallow_rate(run_events)
    elif metric_name == "total_llm_calls_per_run":
        return _compute_llm_calls_per_run(run_events)
    elif metric_name == "concurrent_state_write_rate":
        return _compute_state_write_conflicts(run_events)
    elif metric_name == "schema_mismatch_rate":
        return _compute_schema_mismatch_rate(run_events)
    elif metric_name == "tool_call_failure_rate":
        return _compute_tool_failure_rate(run_events)
    elif metric_name == "delegation_latency_p95_ms":
        return _compute_delegation_latency_p95(run_events)
    return None


def _compute_max_delegation_depth(events: List[Dict]) -> float:
    """Max nesting depth of delegation.initiated chains in this run."""
    return float(sum(1 for e in events if e.get("event_type") == "delegation.initiated"))


def _compute_error_swallow_rate(events: List[Dict]) -> float:
    """Ratio of errors swallowed to total errors. Returns 0.0-1.0."""
    errors = [e for e in events if e.get("event_type", "").startswith("error.")]
    if not errors:
        return 0.0
    swallowed = sum(
        1 for e in errors
        if e.get("payload", {}).get("error_handling") in ("caught_silent", "caught_default", "fail_silent")
        or e.get("payload", {}).get("swallowed", False)
    )
    return min(swallowed / len(errors), 1.0)


def _compute_llm_calls_per_run(events: List[Dict]) -> float:
    """Count of llm.call_start events in this run."""
    return float(sum(1 for e in events if e.get("event_type") == "llm.call_start"))


def _compute_state_write_conflicts(events: List[Dict]) -> float:
    """Ratio of conflicting writes to total writes. Returns 0.0-1.0."""
    writes = [
        e for e in events
        if e.get("event_type") in ("data.write", "state.access")
        and e.get("payload", {}).get("access_type", "") in ("write", "read_write", "")
    ]
    if not writes:
        return 0.0

    # Group by state_key/target
    by_key: Dict[str, List[Dict]] = defaultdict(list)
    for w in writes:
        key = (w.get("target_node", {}).get("node_name", "")
               or w.get("payload", {}).get("state_key", "default"))
        by_key[key].append(w)

    # Count keys with multiple distinct writers
    conflict_writes = 0
    for key, key_writes in by_key.items():
        writers = set()
        for w in key_writes:
            src = (w.get("source_node", {}).get("node_name", "")
                   or w.get("payload", {}).get("node_id", ""))
            if src:
                writers.add(src)
        if len(writers) >= 2:
            conflict_writes += len(key_writes)

    return min(conflict_writes / len(writes), 1.0)


def _compute_schema_mismatch_rate(events: List[Dict]) -> float:
    """Ratio of schema_mismatch errors to total agent.task_end. Returns 0.0-1.0."""
    task_ends = [e for e in events if e.get("event_type") == "agent.task_end"]
    if not task_ends:
        return 0.0
    errors = [e for e in events if e.get("event_type", "").startswith("error.")]
    mismatches = sum(
        1 for e in errors
        if e.get("payload", {}).get("error_type") == "schema_mismatch"
    )
    return min(mismatches / len(task_ends), 1.0)


def _compute_tool_failure_rate(events: List[Dict]) -> float:
    """Ratio: tool failures / (tool completions + tool failures). Returns 0.0-1.0."""
    completed = sum(1 for e in events if e.get("event_type") == "tool.completed")
    failures = sum(1 for e in events if e.get("event_type") == "tool.call_failure")
    total = completed + failures
    if total == 0:
        return 0.0
    return min(failures / total, 1.0)


def _compute_delegation_latency_p95(events: List[Dict]) -> float:
    """Match delegation.initiated -> delegation.completed, compute p95 latency in ms."""
    starts: Dict[str, int] = {}
    latencies: List[float] = []

    for e in events:
        etype = e.get("event_type", "")
        eid = e.get("event_id", "")
        ts = e.get("timestamp_ns", 0)

        if etype == "delegation.initiated":
            starts[eid] = ts
        elif etype == "delegation.completed":
            parent = e.get("parent_event_id", "")
            if parent and parent in starts:
                diff_ns = ts - starts[parent]
                if diff_ns >= 0:
                    latencies.append(diff_ns / 1_000_000)

    if not latencies:
        return 0.0

    latencies.sort()
    idx = int(len(latencies) * 0.95)
    idx = min(idx, len(latencies) - 1)
    return latencies[idx]
