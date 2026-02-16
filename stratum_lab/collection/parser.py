"""JSONL event parser and run record builder.

Reads JSONL event files produced by the stratum-patcher instrumentation,
validates and parses individual events, and builds structured RunRecord
summaries suitable for the graph overlay phase.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required fields for a valid event
# ---------------------------------------------------------------------------
REQUIRED_EVENT_FIELDS = {"event_id", "timestamp_ns", "run_id", "repo_id", "event_type"}

# Event types that represent specific categories
# These match the dot-notation event types emitted by stratum_patcher
AGENT_START_TYPES = {"agent.task_start", "execution.start", "reply.generation_start"}
AGENT_END_TYPES = {"agent.task_end", "execution.end", "reply.generated"}
TOOL_START_TYPES = {"tool.invoked"}
TOOL_END_TYPES = {"tool.completed"}
TOOL_ERROR_TYPES = {"tool.call_failure"}
LLM_START_TYPES = {"llm.call_start"}
LLM_END_TYPES = {"llm.call_end"}
ERROR_EVENT_TYPES = {"error.occurred", "error.propagated", "error.cascade", "tool.call_failure"}
DELEGATION_EVENT_TYPES = {"delegation.initiated", "delegation.completed"}
MESSAGE_EVENT_TYPES = {"message.received", "message.receive_error"}
EDGE_EVENT_TYPES = {"edge.traversed"}
DATA_EVENT_TYPES = {"data.read", "data.write"}
EXTERNAL_EVENT_TYPES = {"external.call"}
FILE_EVENT_TYPES = {"file.read", "file.write"}
SPEAKER_EVENT_TYPES = {"speaker.selected"}
STATE_ACCESS_EVENT_TYPES = {"state.access"}
ROUTING_DECISION_EVENT_TYPES = {"routing.decision"}


# ---------------------------------------------------------------------------
# Event validation
# ---------------------------------------------------------------------------

def validate_event(event_dict: dict[str, Any]) -> bool:
    """Validate that an event dict has all required fields.

    Parameters
    ----------
    event_dict:
        A dict parsed from a single JSONL line.

    Returns
    -------
    True if the event has all required fields, False otherwise.
    """
    for field in REQUIRED_EVENT_FIELDS:
        if field not in event_dict:
            return False
    return True


# ---------------------------------------------------------------------------
# JSONL file parsing
# ---------------------------------------------------------------------------

def parse_events_file(filepath: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file line by line and return a list of EventRecord dicts.

    Malformed JSON lines are logged as warnings and skipped.  Events that
    fail validation are also skipped with a warning.

    Parameters
    ----------
    filepath:
        Path to a ``.jsonl`` file containing one JSON event per line.

    Returns
    -------
    List of validated event dicts, sorted by ``timestamp_ns``.
    """
    filepath = Path(filepath)
    events: list[dict[str, Any]] = []
    skipped = 0

    with open(filepath, "r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue

            # Parse JSON
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                skipped += 1
                if skipped <= 10:
                    logger.warning(
                        "Malformed JSON at %s:%d — %s", filepath.name, line_num, exc
                    )
                continue

            if not isinstance(event, dict):
                skipped += 1
                logger.warning(
                    "Non-dict JSON value at %s:%d — skipping", filepath.name, line_num
                )
                continue

            # Validate
            if not validate_event(event):
                skipped += 1
                if skipped <= 10:
                    logger.warning(
                        "Event missing required fields at %s:%d — skipping",
                        filepath.name,
                        line_num,
                    )
                continue

            events.append(event)

    if skipped > 10:
        logger.warning(
            "%s: skipped %d total malformed/invalid lines (showed first 10 warnings)",
            filepath.name,
            skipped,
        )
    elif skipped > 0:
        logger.warning("%s: skipped %d malformed/invalid lines", filepath.name, skipped)

    # Sort by timestamp for consistent ordering
    events.sort(key=lambda e: e.get("timestamp_ns", 0))
    return events


# ---------------------------------------------------------------------------
# Run record construction
# ---------------------------------------------------------------------------

def build_run_record(
    events: list[dict[str, Any]],
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a structured RunRecord from parsed events and optional metadata.

    Parameters
    ----------
    events:
        List of validated event dicts from a single run (same ``run_id``).
    run_metadata:
        Optional dict with metadata fields (repo_id, run_id, framework, etc.).
        If not supplied, these values are inferred from the events.

    Returns
    -------
    A structured RunRecord dict containing:
      - total_events_by_type
      - execution_timeline (start, end, duration_ns)
      - agent_activations
      - tool_invocations
      - llm_calls
      - error_summary
      - delegation_chains
    """
    if not events:
        return _empty_run_record(run_metadata)

    # Infer basic identifiers from events if metadata not provided
    first = events[0]
    meta = run_metadata or {}
    repo_id = meta.get("repo_id") or first.get("repo_id", "unknown")
    run_id = meta.get("run_id") or first.get("run_id", "unknown")
    framework = meta.get("framework") or first.get("framework", "unknown")

    # ---- Total events by type ----
    type_counts: Counter[str] = Counter()
    for ev in events:
        type_counts[ev["event_type"]] += 1

    # ---- Execution timeline ----
    timestamps = [ev["timestamp_ns"] for ev in events]
    start_ns = min(timestamps)
    end_ns = max(timestamps)
    duration_ns = end_ns - start_ns

    # ---- Agent activations ----
    agent_activations: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"activation_count": 0, "start_events": 0, "end_events": 0, "error_events": 0}
    )
    for ev in events:
        etype = ev["event_type"]
        source = ev.get("source_node") or {}
        agent_name = source.get("node_name", "") or source.get("node_id", "") or "unknown_agent"
        if etype in AGENT_START_TYPES:
            agent_activations[agent_name]["start_events"] += 1
            agent_activations[agent_name]["activation_count"] += 1
        elif etype in AGENT_END_TYPES:
            agent_activations[agent_name]["end_events"] += 1
        elif etype in ERROR_EVENT_TYPES:
            agent_activations[agent_name]["error_events"] += 1

    # ---- Tool invocations ----
    tool_invocations: dict[str, dict[str, int]] = defaultdict(
        lambda: {"invocation_count": 0, "success_count": 0, "failure_count": 0}
    )
    for ev in events:
        etype = ev["event_type"]
        payload = ev.get("payload", {})
        tool_name = payload.get("tool_name") or "unknown_tool"
        if etype in TOOL_START_TYPES:
            tool_invocations[tool_name]["invocation_count"] += 1
        elif etype in TOOL_END_TYPES:
            status = payload.get("status", "success")
            if status in ("success",):
                tool_invocations[tool_name]["success_count"] += 1
            else:
                tool_invocations[tool_name]["failure_count"] += 1
        elif etype in TOOL_ERROR_TYPES:
            tool_invocations[tool_name]["failure_count"] += 1

    # ---- LLM calls ----
    llm_count = 0
    total_tokens = 0
    llm_latencies_ns: list[int] = []
    llm_starts: dict[str, int] = {}

    for ev in events:
        etype = ev["event_type"]
        if etype in LLM_START_TYPES:
            llm_count += 1
            eid = ev.get("event_id", "")
            llm_starts[eid] = ev["timestamp_ns"]
        elif etype in LLM_END_TYPES:
            payload = ev.get("payload", {})
            input_t = payload.get("input_tokens", 0) or 0
            output_t = payload.get("output_tokens", 0) or 0
            total_tokens += input_t + output_t
            parent = ev.get("parent_event_id")
            if parent and parent in llm_starts:
                latency = ev["timestamp_ns"] - llm_starts[parent]
                llm_latencies_ns.append(latency)

    avg_latency_ms = 0.0
    if llm_latencies_ns:
        avg_latency_ms = (sum(llm_latencies_ns) / len(llm_latencies_ns)) / 1_000_000

    # ---- Error summary ----
    error_types: Counter[str] = Counter()
    max_propagation_depth = 0
    for ev in events:
        if ev["event_type"] in ERROR_EVENT_TYPES:
            payload = ev.get("payload", {})
            error_type = payload.get("error_type") or ev["event_type"]
            error_types[error_type] += 1
            depth = ev.get("stack_depth", 0)
            if depth > max_propagation_depth:
                max_propagation_depth = depth

    # ---- Delegation chains ----
    delegation_chains: list[dict[str, str]] = []
    for ev in events:
        if ev["event_type"] in DELEGATION_EVENT_TYPES:
            source = ev.get("source_node") or {}
            target = ev.get("target_node") or {}
            delegation_chains.append({
                "delegator": source.get("node_name", "") or source.get("node_id", "") or "unknown",
                "delegate": target.get("node_name", "") or target.get("node_id", "") or "unknown",
                "event_type": ev["event_type"],
                "timestamp_ns": ev["timestamp_ns"],
            })

    # ---- State access events (v6) ----
    state_access_count = sum(1 for ev in events if ev["event_type"] in STATE_ACCESS_EVENT_TYPES)

    # ---- Routing decision events (v6) ----
    routing_decision_count = sum(1 for ev in events if ev["event_type"] in ROUTING_DECISION_EVENT_TYPES)

    return {
        "repo_id": repo_id,
        "run_id": run_id,
        "framework": framework,
        "total_events": len(events),
        "total_events_by_type": dict(type_counts),
        "execution_timeline": {
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_ns": duration_ns,
            "duration_ms": duration_ns / 1_000_000,
        },
        "agent_activations": dict(agent_activations),
        "tool_invocations": dict(tool_invocations),
        "llm_calls": {
            "count": llm_count,
            "total_tokens": total_tokens,
            "avg_latency_ms": round(avg_latency_ms, 2),
            "latency_count": len(llm_latencies_ns),
        },
        "error_summary": {
            "total_errors": sum(error_types.values()),
            "error_types": dict(error_types),
            "max_propagation_depth": max_propagation_depth,
        },
        "delegation_chains": delegation_chains,
        "state_access_count": state_access_count,
        "routing_decision_count": routing_decision_count,
        "metadata": meta,
    }


def _empty_run_record(meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return an empty run record structure when no events are present."""
    meta = meta or {}
    return {
        "repo_id": meta.get("repo_id", "unknown"),
        "run_id": meta.get("run_id", "unknown"),
        "framework": meta.get("framework", "unknown"),
        "total_events": 0,
        "total_events_by_type": {},
        "execution_timeline": {
            "start_ns": 0,
            "end_ns": 0,
            "duration_ns": 0,
            "duration_ms": 0.0,
        },
        "agent_activations": {},
        "tool_invocations": {},
        "llm_calls": {
            "count": 0,
            "total_tokens": 0,
            "avg_latency_ms": 0.0,
            "latency_count": 0,
        },
        "error_summary": {
            "total_errors": 0,
            "error_types": {},
            "max_propagation_depth": 0,
        },
        "delegation_chains": [],
        "state_access_count": 0,
        "routing_decision_count": 0,
        "metadata": meta,
    }


# ---------------------------------------------------------------------------
# Multi-run aggregation
# ---------------------------------------------------------------------------

def aggregate_run_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate multiple run records for the same repo into a repo-level summary.

    Parameters
    ----------
    records:
        List of RunRecord dicts, all sharing the same ``repo_id``.

    Returns
    -------
    A repo-level summary dict with aggregated statistics across all runs.
    """
    if not records:
        return {
            "repo_id": "unknown",
            "framework": "unknown",
            "num_runs": 0,
            "total_events": 0,
            "avg_events_per_run": 0.0,
            "total_events_by_type": {},
            "execution_timeline": {
                "min_duration_ms": 0.0,
                "max_duration_ms": 0.0,
                "avg_duration_ms": 0.0,
            },
            "agent_activations": {},
            "tool_invocations": {},
            "llm_calls": {
                "total_count": 0,
                "total_tokens": 0,
                "avg_latency_ms": 0.0,
            },
            "error_summary": {
                "total_errors": 0,
                "error_types": {},
                "max_propagation_depth": 0,
                "runs_with_errors": 0,
            },
            "delegation_chains": [],
        }

    repo_id = records[0].get("repo_id", "unknown")
    framework = records[0].get("framework", "unknown")
    num_runs = len(records)

    # ---- Aggregate event type counts ----
    agg_type_counts: Counter[str] = Counter()
    total_events = 0
    for rec in records:
        total_events += rec.get("total_events", 0)
        for etype, count in rec.get("total_events_by_type", {}).items():
            agg_type_counts[etype] += count

    # ---- Execution timeline ----
    durations = [
        rec.get("execution_timeline", {}).get("duration_ms", 0.0) for rec in records
    ]
    durations_nonzero = [d for d in durations if d > 0] or [0.0]

    # ---- Agent activations aggregate ----
    agg_agents: dict[str, dict[str, int]] = defaultdict(
        lambda: {"activation_count": 0, "start_events": 0, "end_events": 0, "error_events": 0}
    )
    for rec in records:
        for agent_name, stats in rec.get("agent_activations", {}).items():
            for key in ("activation_count", "start_events", "end_events", "error_events"):
                agg_agents[agent_name][key] += stats.get(key, 0)

    # ---- Tool invocations aggregate ----
    agg_tools: dict[str, dict[str, int]] = defaultdict(
        lambda: {"invocation_count": 0, "success_count": 0, "failure_count": 0}
    )
    for rec in records:
        for tool_name, stats in rec.get("tool_invocations", {}).items():
            for key in ("invocation_count", "success_count", "failure_count"):
                agg_tools[tool_name][key] += stats.get(key, 0)

    # ---- LLM calls aggregate ----
    total_llm_count = sum(rec.get("llm_calls", {}).get("count", 0) for rec in records)
    total_llm_tokens = sum(rec.get("llm_calls", {}).get("total_tokens", 0) for rec in records)
    weighted_latency_sum = sum(
        rec.get("llm_calls", {}).get("avg_latency_ms", 0.0)
        * rec.get("llm_calls", {}).get("latency_count", 0)
        for rec in records
    )
    total_latency_count = sum(
        rec.get("llm_calls", {}).get("latency_count", 0) for rec in records
    )
    avg_llm_latency = (
        weighted_latency_sum / total_latency_count if total_latency_count > 0 else 0.0
    )

    # ---- Error summary aggregate ----
    agg_error_types: Counter[str] = Counter()
    total_errors = 0
    max_depth = 0
    runs_with_errors = 0
    for rec in records:
        err = rec.get("error_summary", {})
        rec_errors = err.get("total_errors", 0)
        total_errors += rec_errors
        if rec_errors > 0:
            runs_with_errors += 1
        for etype, count in err.get("error_types", {}).items():
            agg_error_types[etype] += count
        depth = err.get("max_propagation_depth", 0)
        if depth > max_depth:
            max_depth = depth

    # ---- Delegation chains (collect all) ----
    all_delegations: list[dict[str, str]] = []
    for rec in records:
        all_delegations.extend(rec.get("delegation_chains", []))

    return {
        "repo_id": repo_id,
        "framework": framework,
        "num_runs": num_runs,
        "total_events": total_events,
        "avg_events_per_run": round(total_events / num_runs, 2),
        "total_events_by_type": dict(agg_type_counts),
        "execution_timeline": {
            "min_duration_ms": round(min(durations_nonzero), 2),
            "max_duration_ms": round(max(durations_nonzero), 2),
            "avg_duration_ms": round(sum(durations_nonzero) / len(durations_nonzero), 2),
        },
        "agent_activations": dict(agg_agents),
        "tool_invocations": dict(agg_tools),
        "llm_calls": {
            "total_count": total_llm_count,
            "total_tokens": total_llm_tokens,
            "avg_latency_ms": round(avg_llm_latency, 2),
        },
        "error_summary": {
            "total_errors": total_errors,
            "error_types": dict(agg_error_types),
            "max_propagation_depth": max_depth,
            "runs_with_errors": runs_with_errors,
        },
        "delegation_chains": all_delegations,
    }
