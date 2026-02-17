"""Load events JSONL and reconstruct ExecutionContext dataclasses."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LLMCall:
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    output_preview: str = ""


@dataclass
class AgentExecution:
    agent_name: str = ""
    task_description: str = ""
    output_text: str = ""
    output_hash: str = ""
    output_type: str = ""
    llm_calls: list[LLMCall] = field(default_factory=list)
    token_count_in: int = 0
    token_count_out: int = 0
    duration_seconds: float = 0.0
    incomplete: bool = False


@dataclass
class ExecutionContext:
    repo_url: str = ""
    framework: str = ""
    agents: list[AgentExecution] = field(default_factory=list)
    delegation_chain: list[tuple[str, str]] = field(default_factory=list)
    total_events: int = 0
    total_duration_seconds: float = 0.0
    status: str = ""


# ---------------------------------------------------------------------------
# Event loading
# ---------------------------------------------------------------------------

def load_events(events_path: Path | str) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of event dicts, sorted by timestamp."""
    events: list[dict[str, Any]] = []
    path = Path(events_path)
    if not path.exists():
        return events
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    # Sort by timestamp_ns (preferred) then by timestamp string
    events.sort(key=lambda e: (e.get("timestamp_ns", 0), e.get("timestamp", "")))
    return events


def _get_payload(evt: dict) -> dict:
    """Extract payload dict from event, handling both nested and flat formats."""
    p = evt.get("payload")
    return p if isinstance(p, dict) else {}


def _get_node_name(evt: dict) -> str:
    """Extract agent/node name from event."""
    sn = evt.get("source_node")
    if isinstance(sn, dict):
        return sn.get("node_name", "")
    return evt.get("agent_name", evt.get("node_name", ""))


def _timestamp_seconds(evt: dict) -> float:
    """Extract timestamp as seconds (float). Tries timestamp_ns first."""
    ns = evt.get("timestamp_ns")
    if ns is not None:
        try:
            return float(ns) / 1e9
        except (TypeError, ValueError):
            pass
    # Fallback: parse ISO timestamp
    ts = evt.get("timestamp", "")
    if ts:
        try:
            from datetime import datetime, timezone
            # Handle Z suffix
            ts = ts.replace("Z", "+00:00")
            dt = datetime.fromisoformat(ts)
            return dt.timestamp()
        except (ValueError, TypeError):
            pass
    return 0.0


# ---------------------------------------------------------------------------
# Context reconstruction
# ---------------------------------------------------------------------------

def reconstruct_context(
    events: list[dict[str, Any]],
    status: str = "",
    repo_url: str = "",
) -> ExecutionContext:
    """Reconstruct an ExecutionContext from a list of events.

    Handles event types: execution.start, execution.end, agent.task_start,
    agent.task_end, llm.call_start, llm.call_end, delegation.initiated,
    patcher.status, file.read, file.write.
    """
    ctx = ExecutionContext(repo_url=repo_url, status=status)
    ctx.total_events = len(events)

    # Track agent spans: agent_name → {start_ts, task_description, llm_calls, ...}
    agent_spans: dict[str, dict] = {}
    # Track LLM call starts: keyed by event_id or source_node
    llm_starts: dict[str, dict] = {}

    exec_start_ts = 0.0
    exec_end_ts = 0.0

    for evt in events:
        etype = evt.get("event_type", "")
        payload = _get_payload(evt)

        if etype == "execution.start":
            exec_start_ts = _timestamp_seconds(evt)
            ctx.repo_url = ctx.repo_url or payload.get("repo_url", "")
            ctx.framework = payload.get("framework", "")

        elif etype == "execution.end":
            exec_end_ts = _timestamp_seconds(evt)

        elif etype == "patcher.status":
            # Extract framework from patcher status if not yet set
            if not ctx.framework:
                patches = payload.get("patches_ok", [])
                if isinstance(patches, list) and patches:
                    ctx.framework = patches[0]

        elif etype == "agent.task_start":
            name = _get_node_name(evt)
            if not name:
                name = payload.get("agent_role", payload.get("agent_name", f"agent_{len(agent_spans)}"))
            agent_spans[name] = {
                "start_ts": _timestamp_seconds(evt),
                "task_description": payload.get("task", payload.get("task_description", "")),
                "llm_calls": [],
            }

        elif etype == "agent.task_end":
            name = _get_node_name(evt)
            if name in agent_spans:
                span = agent_spans[name]
                span["end_ts"] = _timestamp_seconds(evt)
                span["output_text"] = payload.get("output_preview", payload.get("preview", ""))
                span["output_hash"] = payload.get("output_hash", payload.get("hash", ""))
                span["output_type"] = payload.get("output_type", payload.get("type", ""))

        elif etype == "llm.call_start":
            eid = evt.get("event_id", id(evt))
            llm_starts[eid] = {
                "agent_name": _get_node_name(evt),
                "model": payload.get("model", payload.get("model_requested", "")),
                "ts": _timestamp_seconds(evt),
            }

        elif etype == "llm.call_end":
            agent_name = _get_node_name(evt)
            llm_call = LLMCall(
                model=payload.get("model_actual", payload.get("model", "")),
                input_tokens=payload.get("input_tokens", 0) or 0,
                output_tokens=payload.get("output_tokens", 0) or 0,
                latency_ms=payload.get("latency_ms", 0) or 0,
                output_preview=payload.get("output_preview", payload.get("preview", "")),
            )
            # Attach to agent span
            if agent_name in agent_spans:
                agent_spans[agent_name]["llm_calls"].append(llm_call)

        elif etype == "delegation.initiated":
            src = _get_node_name(evt)
            tn = evt.get("target_node")
            if isinstance(tn, dict):
                tgt = tn.get("node_name", "")
            else:
                tgt = payload.get("delegate", "")
            if src and tgt:
                ctx.delegation_chain.append((src, tgt))

    # Build AgentExecution objects from spans
    for name, span in agent_spans.items():
        start_ts = span.get("start_ts", 0.0)
        end_ts = span.get("end_ts", 0.0)
        llm_calls = span.get("llm_calls", [])

        # Handle incomplete agents (missing task_end)
        incomplete = "end_ts" not in span
        output_text = span.get("output_text", "")
        if incomplete and llm_calls and not output_text:
            # Use last LLM call output as partial
            output_text = llm_calls[-1].output_preview

        agent = AgentExecution(
            agent_name=name,
            task_description=span.get("task_description", ""),
            output_text=output_text,
            output_hash=span.get("output_hash", ""),
            output_type=span.get("output_type", ""),
            llm_calls=llm_calls,
            token_count_in=sum(c.input_tokens for c in llm_calls),
            token_count_out=sum(c.output_tokens for c in llm_calls),
            duration_seconds=(end_ts - start_ts) if end_ts > start_ts else 0.0,
            incomplete=incomplete,
        )
        ctx.agents.append(agent)

    # Total duration
    if exec_end_ts > exec_start_ts:
        ctx.total_duration_seconds = exec_end_ts - exec_start_ts
    elif ctx.agents:
        ctx.total_duration_seconds = sum(a.duration_seconds for a in ctx.agents)

    return ctx


def load_execution_context(
    events_path: Path | str,
    status: str = "",
    repo_url: str = "",
) -> ExecutionContext:
    """Load events JSONL and return a fully reconstructed ExecutionContext."""
    events = load_events(events_path)
    return reconstruct_context(events, status=status, repo_url=repo_url)
